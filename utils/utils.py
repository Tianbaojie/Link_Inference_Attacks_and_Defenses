import copy
import math
import random
from os import path as osp

import numpy as np
#设置各种包的随机数，使得结果可以复现
import torch
import torch_geometric
from torch_geometric.deprecation import deprecated
from torch_geometric.utils import to_undirected
from pGRACE.utils import generate_split
from deeprobust.graph.data.pyg_dataset import random_coauthor_amazon_splits
from utils.data import get_dataset
def train_test_split_edges_clga(data,val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed=42
):
    set_random_seed(seed)
    bidirected_edge_index = data.edge_index.cpu().numpy()
    index = np.where(bidirected_edge_index[0]<bidirected_edge_index[1])[0]
    undirected_edge_index = torch.Tensor(bidirected_edge_index[:, index]).long()
    train_mask, test_mask, val_mask = generate_split(int(undirected_edge_index.shape[1]), train_ratio=1-val_ratio-test_ratio, val_ratio=val_ratio)

    train_edge_index = to_undirected(undirected_edge_index[:, train_mask])
    test_edge_index = to_undirected(undirected_edge_index[:, test_mask])
    val_edge_index = to_undirected(undirected_edge_index[:, val_mask])
    data.train_pos_edge_index=train_edge_index
    data.val_pos_edge_index=val_edge_index
    data.test_pos_edge_index=test_edge_index

    observed_edge_sp_adj = torch.sparse.FloatTensor(data.edge_index,
                                                    torch.ones(data.edge_index.shape[1]),
                                                    [data.num_nodes, data.num_nodes])
    observed_edge_adj = observed_edge_sp_adj.to_dense()
    negative_edges = 1 - observed_edge_adj - torch.eye(data.num_nodes)
    negative_edge_index = torch.nonzero(negative_edges)
    negative_edge_index=negative_edge_index[torch.randperm(negative_edge_index.shape[0])].t()
    data.train_neg_edge_index=negative_edge_index[:,-data.train_pos_edge_index.shape[1]:]
    data.val_neg_edge_index=negative_edge_index[:,:data.val_pos_edge_index.shape[1]]
    data.test_neg_edge_index=negative_edge_index[:,:data.test_pos_edge_index.shape[1]]
    return data


def train_test_split_edges(
    data: 'torch_geometric.data.Data',
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
    seed=42
):
    assert 'batch' not in data  # No batch-mode.
    set_random_seed(seed)
    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    
    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    if edge_attr is not None:
        edge_attr = edge_attr[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.val_pos_edge_attr = edge_attr[:n_v]

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        data.test_pos_edge_attr = edge_attr[n_v:n_v + n_t]

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    if edge_attr is not None:
        out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
        data.train_pos_edge_index, data.train_pos_edge_attr = out
    else:
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def set_random_seed(seed: int):
    """设置随机数种子

    Args:
        seed (int): 随机数种子
    """
    # 设置PyTorch的随机数种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 设置NumPy的随机数种子
    np.random.seed(seed)

    # 设置Python自带的随机数种子
    random.seed(seed)

    # 设置torch_geometric的随机数种子
    torch_geometric.seed.seed_everything(seed)

#生成数据集
def generate_dataset(dataset_name):
    #for dataset_name in ['Cora','CiteSeer','PolBlogs','PubMed']:
    path = osp.expanduser('dataset')
    path = osp.join(path, dataset_name)
    dataset = get_dataset(path, dataset_name)
    #如果是Coauthor-CS或者Coauthor-Phy或者Amazon-Computers或者Amazon-Photo，就使用random_coauthor_amazon_splits函数
    if dataset_name in ['Coauthor-CS','PolBlogs', 'Coauthor-Phy','Amazon-Computers', 'Amazon-Photo']:
        random_coauthor_amazon_splits(dataset,dataset.data.y.max()+1,None)
    data = dataset[0]
    data = train_test_split_edges_clga(data, val_ratio=0.1, test_ratio=0.2,seed=42)
    if dataset_name in ['PolBlogs']:
        data.x=torch.randn((data.num_nodes,30))
    torch.save(data, osp.join(path, 'data.pt'))

#将modifiedAdj转化为data
#data的格式为Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], val_pos_edge_index=[2, 527], test_pos_edge_index=[2, 527], train_pos_edge_index=[2, 8448], train_neg_adj_mask=[2708, 2708], val_neg_edge_index=[2, 527], test_neg_edge_index=[2, 527])

def modifiedAdj2data(modifiedAdj,data):
    new_data=copy.deepcopy(data)
    new_data.edge_index= modifiedAdj.nonzero().T
    new_data.num_edges=new_data.edge_index.shape[1]

    edges=[]
    for edge in data.val_pos_edge_index.T.tolist():
        if edge in new_data.edge_index.T.tolist():
            edges.append(edge)
    new_data.val_pos_edge_index=torch.tensor(edges).T
    #新的test_pos_edge_index是原有的data.test_pos_edge_index和new_data的edge_index的交集
    edges=[]
    for edge in data.test_pos_edge_index.T.tolist():
        if edge in new_data.edge_index.T.tolist():
            edges.append(edge)
    new_data.test_pos_edge_index=torch.tensor(edges).T
    #新的val_neg_edge_index是原有的data.val_neg_edge_index和new_data的edge_index的交集
    edges=[]
    for edge in data.val_neg_edge_index.T.tolist():
        if edge not in new_data.edge_index.T.tolist():
            edges.append(edge)
    new_data.val_neg_edge_index=torch.tensor(edges).T
    #新的test_neg_edge_index是原有的data.test_neg_edge_index和new_data的edge_index的交集
    edges=[]
    for edge in data.test_neg_edge_index.T.tolist():
        if edge not in new_data.edge_index.T.tolist():
            edges.append(edge)
    new_data.test_neg_edge_index=torch.tensor(edges).T

    edges=[]
    for edge in new_data.edge_index.T.tolist():
        if (edge not in new_data.val_pos_edge_index.T.tolist()) and (edge not in new_data.test_pos_edge_index.T.tolist()):
            if ([edge[1],edge[0]] not in new_data.val_pos_edge_index.T.tolist()) and ([edge[1],edge[0]] not in new_data.test_pos_edge_index.T.tolist()):
                edges.append(edge)
    new_data.train_pos_edge_index=torch.tensor(edges).T
    return new_data
def modifiedAdj2data_small(modifiedAdj,data):
    new_data=copy.deepcopy(data)
    new_data.edge_index= modifiedAdj.nonzero().T
    print('modifiedAdj2data_small')
    new_edge_index=set([(a,b) for a,b in zip(new_data.edge_index[0].tolist(),new_data.edge_index[1].tolist())])
    old_edge_index=set([(a,b) for a,b in zip(data.edge_index[0].tolist(),data.edge_index[1].tolist())])
    old_train_pos_edge_index=set([(a,b) for a,b in zip(data.train_pos_edge_index[0].tolist(),data.train_pos_edge_index[1].tolist())])
    old_val_pos_edge_index=set([(a,b) for a,b in zip(data.val_pos_edge_index[0].tolist(),data.val_pos_edge_index[1].tolist())])
    old_val_neg_edge_index=set([(a,b) for a,b in zip(data.val_neg_edge_index[0].tolist(),data.val_neg_edge_index[1].tolist())])
    old_test_pos_edge_index=set([(a,b) for a,b in zip(data.test_pos_edge_index[0].tolist(),data.test_pos_edge_index[1].tolist())])
    old_test_neg_edge_index=set([(a,b) for a,b in zip(data.test_neg_edge_index[0].tolist(),data.test_neg_edge_index[1].tolist())])
    
    new_val_pos_edge_index=old_val_pos_edge_index.intersection(new_edge_index)
    new_test_pos_edge_index=old_test_pos_edge_index.intersection(new_edge_index)

    new_val_neg_edge_index=old_val_neg_edge_index.difference(new_edge_index)
    new_test_neg_edge_index=old_test_neg_edge_index.difference(new_edge_index)

    new_train_pos_edge_index=new_edge_index.difference(new_val_pos_edge_index).difference(new_test_pos_edge_index)
    new_data.num_edges=new_data.edge_index.shape[1]
    new_data.train_pos_edge_index= torch.tensor(np.array(list(new_train_pos_edge_index)).T)
    new_data.val_pos_edge_index= torch.tensor(np.array(list(new_val_pos_edge_index)).T)
    new_data.val_neg_edge_index= torch.tensor(np.array(list(new_val_neg_edge_index)).T)
    new_data.test_pos_edge_index= torch.tensor(np.array(list(new_test_pos_edge_index)).T)
    new_data.test_neg_edge_index= torch.tensor(np.array(list(new_test_neg_edge_index)).T)
    return new_data
def modifiedAdj2data_large(modifiedAdj,data):
    new_data=copy.deepcopy(data)
    noneZero=modifiedAdj.nonzero()
    new_data.edge_index= torch.tensor(np.array([noneZero[0],noneZero[1]]))
    print('modifiedAdj2data_large')
    new_edge_index=set([(a,b) for a,b in zip(noneZero[0],noneZero[1])])
    old_edge_index=set([(a,b) for a,b in zip(data.edge_index[0].tolist(),data.edge_index[1].tolist())])
    old_train_pos_edge_index=set([(a,b) for a,b in zip(data.train_pos_edge_index[0].tolist(),data.train_pos_edge_index[1].tolist())])
    old_val_pos_edge_index=set([(a,b) for a,b in zip(data.val_pos_edge_index[0].tolist(),data.val_pos_edge_index[1].tolist())])
    old_val_neg_edge_index=set([(a,b) for a,b in zip(data.val_neg_edge_index[0].tolist(),data.val_neg_edge_index[1].tolist())])
    old_test_pos_edge_index=set([(a,b) for a,b in zip(data.test_pos_edge_index[0].tolist(),data.test_pos_edge_index[1].tolist())])
    old_test_neg_edge_index=set([(a,b) for a,b in zip(data.test_neg_edge_index[0].tolist(),data.test_neg_edge_index[1].tolist())])
    new_val_pos_edge_index=old_val_pos_edge_index.difference(new_edge_index)
    new_test_pos_edge_index=old_test_pos_edge_index.difference(new_edge_index)
    new_val_neg_edge_index=old_val_neg_edge_index.difference(new_edge_index)
    new_test_neg_edge_index=old_train_pos_edge_index.intersection(new_edge_index) | new_edge_index.difference(old_edge_index)
    new_train_pos_edge_index=new_edge_index.difference(old_val_pos_edge_index).difference(old_test_pos_edge_index)
    new_data.num_edges=new_data.edge_index.shape[1]
    new_data.train_pos_edge_index= torch.tensor(np.array(list(new_train_pos_edge_index)).T)
    new_data.val_pos_edge_index= torch.tensor(np.array(list(new_val_pos_edge_index)).T)
    new_data.val_neg_edge_index= torch.tensor(np.array(list(new_val_neg_edge_index)).T)
    new_data.test_pos_edge_index= torch.tensor(np.array(list(new_test_pos_edge_index)).T)
    new_data.test_neg_edge_index= torch.tensor(np.array(list(new_test_neg_edge_index)).T)
    return new_data