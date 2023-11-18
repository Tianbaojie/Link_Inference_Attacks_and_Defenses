import copy

import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch_geometric.nn import GATConv, GCNConv,GCN2Conv
from torch_geometric.utils import add_self_loops, structured_negative_sampling

from lp_models.BaseLP import BaseLp
from utils.evaluation import LPEvaluator

class GCNII(torch.nn.Module):
    def __init__(self, embedding_dim_in,embedding_dim_out, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = nn.Linear(embedding_dim_in, embedding_dim_out)
    

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(embedding_dim_out, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins(x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)

        return x
    
#基于GCNII的链接预测模型
class GCNII_LP(BaseLp):
    '''
    GCN_LP model used for link prediction based on GCN, and the output of the last layer is normalized.
    The model trains the GCN model and then uses the output of the last layer as the node representation.
    The loss function is the negative log likelihood loss.

    Parameters:
    embedding_dim: int
        The number of embedding dimension.
    device: torch.device
        The device to run the model.
    '''
    def __init__(self,embedding_dim_in,embedding_dim_out,device):
        super(GCNII_LP,self).__init__()
        self.device=device
        self.model=GCNII(embedding_dim_in,embedding_dim_out,num_layers=5, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6).to(device)
