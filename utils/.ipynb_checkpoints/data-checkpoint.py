import os
import os.path as osp
import sys
import warnings
from pathlib import Path

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (Amazon, CitationFull, Coauthor,
                                      Planetoid, PolBlogs, TUDataset, WikiCS)
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges

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

