import os
import os.path as osp
import sys
import warnings
from pathlib import Path
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

os.environ['DGLBACKEND'] = 'pytorch'
warnings.filterwarnings('ignore')


import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (Amazon, CitationFull, Coauthor,
                                      Planetoid, PolBlogs, TUDataset, WikiCS)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges
from torch.utils import data
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent
warnings.filterwarnings("ignore")

def get_dataset(path,name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'PolBlogs', 'Coauthor-CS', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo']
    name = 'dblp' if name == 'DBLP' else name

    if name == 'Proteins':
        return TUDataset(root=path, name='PROTEINS', transform=T.NormalizeFeatures())

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    if name == 'PolBlogs':
        return PolBlogs(root=path, transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(path, 'Citation'), name, transform=T.NormalizeFeatures())

class QADADataset(data.Dataset):
    def __init__(self, node_paths_link,type_paths_link,masks_link,answer_position_link,label=None):
        self.node_paths_link=torch.tensor(node_paths_link)
        self.type_paths_link=torch.tensor(type_paths_link)
        self.masks_link=torch.tensor(masks_link)
        self.answer_position_link=torch.tensor(answer_position_link)
        self.label=torch.tensor(label)
    def __getitem__(self, idx):
        return self.node_paths_link[idx],self.type_paths_link[idx],self.masks_link[idx],self.answer_position_link[idx]#,self.label[idx]
    def __len__(self):
        return len(self.node_paths_link)