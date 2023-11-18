import copy

import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, structured_negative_sampling

from lp_models_for_large_dataset.BaseLP import BaseLp
from utils.evaluation import LPEvaluator


class GCN(torch.nn.Module):
    '''
    GCN model has two layers of GCNConv,one layer of Linear,every layer has a ReLU activation function.
    ----------------
    Parameters:
    embedding_dim: int
        The number of embedding dimension.
    '''
    def __init__(self, embedding_dim_in,embedding_dim_out):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(embedding_dim_in, embedding_dim_out)
        self.conv2 = GCNConv(embedding_dim_out, embedding_dim_out)
        self.fc=nn.Linear(embedding_dim_out,embedding_dim_out)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        # x: 节点特征矩阵，shape为 [num_nodes, input_dim]
        # edge_index: 边的索引矩阵，shape为 [2, num_edges]

        # 进行第一层卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 进行第二层卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 返回每个节点的表示
        return self.fc(x)
    
#基于GCN的链接预测模型
class GCN_LP(BaseLp):
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
        super(GCN_LP,self).__init__()
        self.device=device
        self.model=GCN(embedding_dim_in,embedding_dim_out).to(device)
