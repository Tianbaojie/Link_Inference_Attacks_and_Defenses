import copy

import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, structured_negative_sampling


from lp_models_for_large_dataset.BaseLP import BaseLp
from utils.evaluation import LPEvaluator


class GAT(torch.nn.Module):
    '''
    GAT model
    The model has two layers of GATConv,onr layer of Linear,every layer has a ReLU activation function.
    ----------------
    Parameters:
    embedding_dim: int
        The number of embedding dimension.
    '''
    def __init__(self, embedding_dim_in,embedding_dim_out):
        super(GAT, self).__init__()
        self.conv1 = GATConv(embedding_dim_in, embedding_dim_out)
        self.conv2 = GATConv(embedding_dim_out, embedding_dim_out)
        self.act = nn.ReLU()
        self.fc=nn.Linear(embedding_dim_out,embedding_dim_out)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        return self.fc(x)
    
    
#基于GAT的链接预测模型
class GAT_LP(BaseLp):
    '''
    GCN_LP model used for link prediction based on GAT, and the output of the last layer is normalized.

    Parameters:
    embedding_dim: int
        The number of embedding dimension.
    device: torch.device
        The device to run the model.
    '''
    def __init__(self,embedding_dim_in,embedding_dim_out,device):
        super(GAT_LP,self).__init__()
        self.device=device
        self.model=GAT(embedding_dim_in,embedding_dim_out).to(device)
