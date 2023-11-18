import argparse
import os.path as osp
import pickle as pkl
import time
from typing import Optional

import nni
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree, to_undirected

from lp_models.BaseLP import BaseLp
from pGRACE.dataset import get_dataset
from pGRACE.eval import link_prediction
from pGRACE.functional import (degree_drop_weights, drop_edge_weighted,
                               feature_drop_weights,
                               feature_drop_weights_dense)
from pGRACE.model import GRACE, Encoder
from pGRACE.utils import generate_split, get_activation, get_base_model
from simple_param.sp import SimpleParam
from utils.evaluation import LPEvaluator


#The following is borrowed from https://github.com/RinneSz/CLGA
class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(
                    2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]


class GRACE(torch.nn.Module):
    '''
    GRACE model is a semi-supervised link prediction model, which is based on the GNN model,fisrtly, it uses the GNN model to learn the node representation, and then uses the node representation to predict the link.
    GRACE is proposed in the paper "GRACE: Graph Representation Learning via Adaptive Contrastive Estimation" (https://arxiv.org/abs/2006.04131).
    Parameters
    ----------
    encoder: Encoder
        The encoder of the GRACE model.
    num_hidden: int
        The dimension of the node representation.
    num_proj_hidden: int
        The dimension of the projected node representation.
    tau: float
        The temperature parameter of the GRACE model.
    '''
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        def f(x): return torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        def f(x): return torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: Optional[int] = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


class LogReg(nn.Module):
    '''
    Logistic Regression model for node classification.
    Parameters
    ----------
    ft_in: int
        The input feature dimension.
    nb_classes: int
        The output dimension.
    '''
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class GCA(torch.nn.Module):
    '''
    GCA: Graph Contrastive Learning with Adaptive Graph Augmentation, ICLR 2021
    This is the implementation of GCA model.
    Parameters
    ----------
    data: Data
        The data object.
    device: torch.device
        The device to run the model.
    embedding_dim: int
        The dimension of node embedding.
    '''
    def __init__(self, data,device,embedding_dim):
        super(GCA, self).__init__()
        self.device=device
        self.data=data
        self.embedding_dim=embedding_dim
        self.z=torch.tensor(self.get_z(),dtype=torch.float32,requires_grad=False).to(self.device)
        self.projecter = LogReg(embedding_dim, embedding_dim).to(self.device)
    def get_z(self):
        encoder = Encoder(self.data.x.shape[1],self.embedding_dim,get_activation('prelu'),base_model=get_base_model('GCNConv'), k=2).to(self.device)
        model = GRACE(encoder, self.embedding_dim, 32, 0.4).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-5)
        num_epochs=100
        drop_weights = degree_drop_weights(self.data.train_pos_edge_index)
        self.train_gca(model,optimizer,num_epochs,drop_weights)
        model.eval()
        z = model(self.data.x.to(self.device), self.data.train_pos_edge_index.to(self.device)).detach()
        return z
    def train_gca(self,model,optimizer,num_epochs,drop_weights):
        model.train()
        x_1 = self.data.x.to(self.device)
        x_2 = self.data.x.to(self.device)
        def drop_edge(idx: int):
                return drop_edge_weighted(self.data.train_pos_edge_index, drop_weights, p=0.1 if idx==1 else 0, threshold=0.7)

        edge_index_1 = drop_edge(1).to(self.device)
        edge_index_2 = drop_edge(2).to(self.device)
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1,
                                                torch.ones(edge_index_1.shape[1]).to(self.device), [self.data.num_nodes, self.data.num_nodes]).to(self.device)
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2,
                                                torch.ones(edge_index_2.shape[1]).to(self.device), [self.data.num_nodes, self.data.num_nodes]).to(self.device)
        edge_adj_1 = edge_sp_adj_1.to_dense()
        edge_adj_2 = edge_sp_adj_2.to_dense()
        for epoch in tqdm.tqdm(range(num_epochs)):
            optimizer.zero_grad()
            z1 = model(x_1, edge_adj_1.nonzero().T.to(self.device))
            z2 = model(x_2, edge_adj_2.nonzero().T.to(self.device))
            loss = model.loss(z1, z2)
            loss.backward()
            optimizer.step()
    def forward(self,x=None,edge_index=None):
        return self.projecter(self.z)



class GCA_LP(BaseLp):
    '''
    GCA_LP is based on GCA model,used for link prediction.
    Parameters
    ----------
    data: Data
        The data object.
    device: torch.device
        The device to run the model.
    '''
    #初始化
    def __init__(self, data, device):
        super(GCA_LP, self).__init__()
        self.data = data
        self.device = device
        self.model = GCA(data, device,256)

