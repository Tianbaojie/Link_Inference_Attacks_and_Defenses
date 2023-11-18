import copy
import datetime
import gc
import itertools
import os
import pickle
import queue
import random
import time
import warnings
from dataclasses import dataclass

import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dgl import shortest_dist
from dgl.base import NID
from dgl.convert import to_heterogeneous, to_homogeneous
from dgl.data import AsGraphPredDataset
from dgl.data.utils import save_graphs
from dgl.dataloading import GraphDataLoader
from dgl.nn import HeteroEmbedding, PathEncoder
from dgl.random import choice
from dgl.sampling import random_walk
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator, collate_dgl
from ogb.graphproppred.mol_encoder import AtomEncoder
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SparseAdam
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import datetime
import gc
import itertools
import os
import pickle
import queue
import random
import time
import warnings
from dataclasses import dataclass

import dgl
import dgl.nn as dglnn
import dgl.sparse as dglsp
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch as th
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from dgl import shortest_dist
from dgl.base import NID
from dgl.convert import to_heterogeneous, to_homogeneous
from dgl.data import AsGraphPredDataset
from dgl.data.utils import save_graphs
from dgl.dataloading import GraphDataLoader
from dgl.nn import HeteroEmbedding, PathEncoder
from dgl.random import choice
from dgl.sampling import random_walk
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator, collate_dgl
from ogb.graphproppred.mol_encoder import AtomEncoder
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SparseAdam
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.build_graph import build_graph_dgl

from utils.build_graph import build_graph_dgl

os.environ['DGLBACKEND'] = 'pytorch'
warnings.filterwarnings('ignore')



class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class MetaPathTransformer(nn.Module):
    def __init__(self, max_length, num_head, embedding_size, device):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_length, embedding_size)
        self.nodetype_embeddings = nn.Embedding(4, embedding_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, batch_first=True, nhead=num_head)
        self.position_embeddings_index = torch.tensor(list(range(max_length))).to(device)
        self.max_length = max_length
        self.embedding_size = embedding_size
        self.device = device

    def forward(self, node_embeddings, node_types_link, src_mask):
        embedding = node_embeddings + \
            self.position_embeddings(self.position_embeddings_index)+self.nodetype_embeddings(node_types_link)
        embedding = embedding.view(-1, self.max_length, self.embedding_size)
        return self.encoder_layer(embedding, src_key_padding_mask=src_mask.view(-1, self.max_length))


class DotProductSimilarity(nn.Module):
    def __init__(self):
        super(DotProductSimilarity, self).__init__()

    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        return result


class PathGather(nn.Module):
    def __init__(self, embedding_size=128, max_paths=128):
        super(PathGather, self).__init__()
        self.dot = DotProductSimilarity()
        self.embedding_size = embedding_size
        self.max_paths = max_paths

    def forward(self, link_user_embeddings, link_answer_embeddings):
        score = self.dot(link_user_embeddings, link_answer_embeddings).view(-1, self.max_paths)
        attention = F.softmax(score)
        attention = attention.unsqueeze(-1)
        link_user_path_embeddings = torch.sum(
            attention*(link_user_embeddings.view(-1, self.max_paths, self.embedding_size)), 1)
        link_answer_path_embeddings = torch.sum(
            attention*(link_answer_embeddings.view(-1, self.max_paths, self.embedding_size)), 1)
        return self.dot(link_user_path_embeddings, link_answer_path_embeddings)


class ExplainableGraphTransformer(nn.Module):
    def __init__(self, embeddingDict, embedding_size, etypes, max_length, num_head, device='cuda:0', max_paths=10):
        super().__init__()
        self.embeddings = HeteroEmbedding(embeddingDict, embedding_size)
        self.rgcn = RGCN(embedding_size, embedding_size, embedding_size, etypes)
        self.metapathencoder = MetaPathTransformer(max_length, num_head, embedding_size, device)
        self.pathgather = PathGather(embedding_size, max_paths)

        self.device = device
        self.id2nodetype={0:'user',1:'answer',2:'topic',3:'question'}
    def forward(self, g, node_paths_link, type_paths_link, masks_link, answer_position_link):
        # node_types=[[nodetype2id[nodetype] for nodetype in path] for path in type_paths]
        h = self.embeddings({key: g.nodes(key) for key in g.ntypes})
        x = self.rgcn(g, h)
        node_embeddings = torch.stack([torch.stack([torch.stack([x[self.id2nodetype[key]][value] for value, key in zip(node_path, type_path)])
                                      for node_path, type_path in zip(node_paths, type_paths)]) for node_paths, type_paths in zip(node_paths_link, type_paths_link.tolist())])
        # node_types_link=torch.tensor([[[nodetype2id[nodetype] for nodetype in path] for path in type_paths] for type_paths in type_paths_link])
        link_path_embeddings = self.metapathencoder(node_embeddings.to(
            self.device), type_paths_link.to(self.device), masks_link.to(self.device))
        index = answer_position_link.view(-1)



        #-----------------------消融实验----------------------------------
        link_user_embeddings = link_path_embeddings[:, 0, :]
        link_answer_embeddings = torch.stack([link_path_embeddings[num, i, :]
                                             for num, i in zip(range(len(index)), index)])
        return self.pathgather(link_user_embeddings, link_answer_embeddings)

class RGCNClassfier(nn.Module):
    def __init__(self, embeddingDict, embedding_size, etypes,max_length=5,max_paths=10):
        super().__init__()
        self.embeddings = HeteroEmbedding(embeddingDict, embedding_size)
        self.rgcn = RGCN(embedding_size, embedding_size, embedding_size, etypes)
        self.dot=DotProductSimilarity()
        self.max_paths=max_paths
        self.max_length=max_length
        self.embedding_size=embedding_size
        self.id2nodetype={0:'user',1:'answer',2:'topic',3:'question'}
    def forward(self, g, node_paths_link, type_paths_link, masks_link, answer_position_link):
        # node_types=[[nodetype2id[nodetype] for nodetype in path] for path in type_paths]
        h = self.embeddings({key: g.nodes(key) for key in g.ntypes})
        x = self.rgcn(g, h)
        link_path_embeddings = torch.stack([torch.stack([torch.stack([x[self.id2nodetype[key]][value] for value, key in zip(node_path, type_path)])
                                      for node_path, type_path in zip(node_paths, type_paths)]) for node_paths, type_paths in zip(node_paths_link, type_paths_link.tolist())])
        link_path_embeddings=link_path_embeddings.view(-1, self.max_length, self.embedding_size)
        index = answer_position_link.view(-1)
        link_user_embeddings = link_path_embeddings[:, 0, :]
        link_answer_embeddings = torch.stack([link_path_embeddings[num, i, :]
                                             for num, i in zip(range(len(index)), index)])
        score = self.dot(link_user_embeddings, link_answer_embeddings).view(-1, self.max_paths)
        return score[:,0]


