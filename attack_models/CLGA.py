import time

import torch
from torch.nn.modules.module import Module
from torch_geometric.utils import degree, dense_to_sparse, to_undirected
import numpy  as np
import networkx as nx
from differentiable_models.gcn import GCN
from differentiable_models.model import GRACE
from pGRACE.functional import (degree_drop_weights, drop_edge_weighted,
                               drop_feature, drop_feature_weighted,
                               feature_drop_weights)
from pGRACE.utils import get_activation
import tqdm

class Metacl(Module):
    def __init__(self, args, data, device):
        super(Metacl, self).__init__()
        self.model = None
        self.optimizer = None
        self.args = args
        self.device = device
        self.data = data.to(device)
        self.drop_weights = None
        self.feature_weights = None

    def drop_edge(self, p):
        return drop_edge_weighted(self.data.edge_index, self.drop_weights, p=p,threshold=0.7)

    def train_gcn(self):
        self.model.train()
        self.optimizer.zero_grad()
        edge_index_1 = self.drop_edge(0.3)
        edge_index_2 = self.drop_edge(0.4)
        x_1 = drop_feature(self.data.x, 0.1)
        x_2 = drop_feature(self.data.x, 0)
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1,
                                                 torch.ones(edge_index_1.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2,
                                                 torch.ones(edge_index_2.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        x_1 = drop_feature_weighted(self.data.x, self.feature_weights, 0.1)
        x_2 = drop_feature_weighted(self.data.x, self.feature_weights, 0)
        z1 = self.model(x_1, edge_sp_adj_1, sparse=True)
        z2 = self.model(x_2, edge_sp_adj_2, sparse=True)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_drop_weights(self):
        self.drop_weights = degree_drop_weights(self.data.edge_index).to(self.device)
        edge_index_ = to_undirected(self.data.edge_index)
        node_deg = degree(edge_index_[1])
        self.feature_weights = feature_drop_weights(self.data.x, node_c=node_deg).to(self.device)

    def inner_train(self):
        encoder = GCN(self.data.x.shape[1], 256, get_activation('prelu'))
        self.model = GRACE(encoder, 256, 32, 0.4).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.01,
            weight_decay=1e-5
        )
        self.compute_drop_weights()
        for epoch in tqdm.tqdm(range(1, 1000 + 1)):
            loss = self.train_gcn()

    def compute_gradient(self, pe1, pe2, pf1, pf2):
        self.model.eval()
        self.compute_drop_weights()
        edge_index_1 = self.drop_edge(pe1)
        edge_index_2 = self.drop_edge(pe2)
        x_1 = drop_feature(self.data.x, pf1)
        x_2 = drop_feature(self.data.x, pf2)
        edge_sp_adj_1 = torch.sparse.FloatTensor(edge_index_1,
                                                 torch.ones(edge_index_1.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        edge_sp_adj_2 = torch.sparse.FloatTensor(edge_index_2,
                                                 torch.ones(edge_index_2.shape[1]).to(self.device),
                                                 [self.data.num_nodes, self.data.num_nodes])
        x_1 = drop_feature_weighted(self.data.x, self.feature_weights, pf1)
        x_2 = drop_feature_weighted(self.data.x, self.feature_weights, pf2)
        edge_adj_1 = edge_sp_adj_1.to_dense()
        edge_adj_2 = edge_sp_adj_2.to_dense()
        edge_adj_1.requires_grad = True
        edge_adj_2.requires_grad = True
        z1 = self.model(x_1, edge_adj_1, sparse=False)
        z2 = self.model(x_2, edge_adj_2, sparse=False)
        loss = self.model.loss(z1, z2, batch_size=None)
        loss.backward()
        return edge_adj_1.grad, edge_adj_2.grad

    def attack(self):
        perturbed_edges = []
        num_total_edges = self.data.num_edges
        posioned_edge_index=torch.concat([self.data.train_pos_edge_index,self.data.val_pos_edge_index],axis=1)
        adj_sp = torch.sparse.FloatTensor(posioned_edge_index, torch.ones(posioned_edge_index.shape[1]).to(self.device),
                                          [self.data.num_nodes, self.data.num_nodes])
        adj = adj_sp.to_dense()

        print('Begin perturbing.....')
        # save three poisoned adj when the perturbation rate reaches 1%, 5%, 10%
        while len(perturbed_edges) < int(self.args.attack_rate* num_total_edges):
            if len(perturbed_edges)%20==0: 
                self.inner_train()
            adj_1_grad, adj_2_grad = self.compute_gradient(0.3, 0.4, 0.1, 0)
            grad_sum = adj_1_grad + adj_2_grad
            grad_sum_1d = grad_sum.view(-1)
            grad_sum_1d_abs = torch.abs(grad_sum_1d)
            values, indices = grad_sum_1d_abs.sort(descending=True)
            i = -1
            while True:
                i += 1
                index = int(indices[i])
                row = int(index / self.data.num_nodes)
                column = index % self.data.num_nodes
                if [row, column] in perturbed_edges:
                    continue
                if grad_sum_1d[index] < 0 and adj[row, column] == 1:
                    adj[row, column] = 0
                    adj[column, row] = 0
                    perturbed_edges.append([row, column])
                    perturbed_edges.append([column, row])
                    break
                elif grad_sum_1d[index] > 0 and adj[row, column] == 0:
                    adj[row, column] = 1
                    adj[column, row] = 1
                    perturbed_edges.append([row, column])
                    perturbed_edges.append([column, row])
                    break
            self.data.edge_index = dense_to_sparse(adj)[0]
            end = time.time()
        output_adj = adj.to('cpu')

        posioned_edge_index=torch.tensor(output_adj).nonzero().T

        
        G = nx.Graph()
        # 添加所有节点
        for i in range(self.data.num_nodes):
            G.add_node(i)
        # 添加训练集所有边
        edge_index = np.array(posioned_edge_index.cpu())
        for j in range(edge_index.shape[1]):
            G.add_edge(edge_index[0][j], edge_index[1][j])
        
        #edge_index = np.array(self.data.val_pos_edge_index)
        #for j in range(edge_index.shape[1]):
            #G.add_edge(edge_index[0][j], edge_index[1][j])
        edge_index = np.array(self.data.test_pos_edge_index.cpu())
        for j in range(edge_index.shape[1]):
            G.add_edge(edge_index[0][j], edge_index[1][j])
        adj_matrix = nx.adjacency_matrix(G).todense()
        adj_matrix=torch.tensor(adj_matrix)
        return adj_matrix