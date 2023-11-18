import copy

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from gensim.models import Word2Vec
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, structured_negative_sampling

from deeprobust.graph.defense import Node2Vec as node2vec
from lp_models.BaseLP import BaseLp
from utils.evaluation import LPEvaluator


class Node2Vec(torch.nn.Module):
    def __init__(self,graph,embedding_dim,device) -> None:
        super(Node2Vec, self).__init__()
        self.adj=nx.adjacency_matrix(graph).todense()
        temp=node2vec()
        temp.fit(self.adj,embedding_dim=embedding_dim)
        self.embedding=torch.tensor(temp.embedding,dtype=torch.float32,requires_grad=False).to(device)

        self.fc=nn.Linear(embedding_dim,embedding_dim)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self,x=None,edge_index=None):
        return self.fc(self.embedding)

#基于Node2Vec的链接预测模型
class Node2Vec_LP(BaseLp):
    def __init__(self,data,embedding_dim,device):
        super(Node2Vec_LP,self).__init__()
        self.device=device
        self.graph = self.bulid_graph(data)
        self.embedding_dim = embedding_dim
        self.model=Node2Vec(self.graph,self.embedding_dim,device).to(self.device)
    def bulid_graph(self,data):
        G = nx.Graph()
        # 添加所有节点
        for i in range(data.num_nodes):
            G.add_node(i)
        edge_index =  np.array(data.train_pos_edge_index.cpu())
        for j in range(edge_index.shape[1]):
            G.add_edge(edge_index[0][j], edge_index[1][j])
        return G