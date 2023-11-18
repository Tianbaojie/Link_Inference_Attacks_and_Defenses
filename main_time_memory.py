import argparse
import copy
import datetime
import logging
import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import sys
import os.path as osp
import networkx as nx
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RemoveDuplicatedEdges
import deeprobust
from attack_models.baseline_attacks import baseline_attacks
from attack_models.integrity import AALP_Integrity
from attack_models.availability import AALP_Availability
from attack_models.CLGA import Metacl as CLGA
from attack_models.viking import get_attacked_graph_viking as viking
from lp_models.DeepWalk import DeepWalk_LP
from lp_models.GAT import GAT_LP
from lp_models.GCA import GCA_LP
from lp_models.GCN import GCN_LP
from lp_models.MetaModel import MetaModel_LP
from lp_models.Node2Vec import Node2Vec_LP
from lp_models.VGNAE import VGNAE_LP
from utils.io import (load_clean_model, load_dataset, load_modifiedAdj,
                      save_args, save_best_scores, save_best_test_result,
                      save_best_val_result, save_results)
from utils.utils import (modifiedAdj2data_large,modifiedAdj2data_small,
                         set_random_seed, train_test_split_edges_clga)
import subprocess
from utils.io import (load_clean_model, load_dataset, load_modifiedAdj,
                      save_args, save_best_scores, save_best_test_result,
                      save_best_val_result, save_results)
from utils.utils import (modifiedAdj2data_large,modifiedAdj2data_small,
                         set_random_seed, train_test_split_edges_clga)
from deeprobust.graph.global_attack import NodeEmbeddingAttack
from scipy.sparse import csr_matrix
from process_isolated_nodes import process_isolated_nodes, restore_isolated_ndoes,restore_isolated_ndoes_int
from torch_geometric.utils import contains_isolated_nodes
def run_clean_exp(args,data):
    print('run clean exp on :')
    print(data)
    #查看lp_model是不是符合规定,如果不符合规定就报错，并提示错误信息
    assert args.lp_model in ['deepwalk','node2vec','gcn', 'gat', 'gca','vgnae','metamodel'] 
    best_val_result,best_test_result,best_scores=None,None,None
    #根据args.lp_model的值来选择不同的模型
    if args.lp_model == 'deepwalk':
        model=DeepWalk_LP(data,128,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=0.01)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,1000)
        #保存model, best_val_result, best_test_result, best_scores, args, data
        save_results(model, best_val_result, best_test_result, best_scores, args, data)

    elif args.lp_model == 'node2vec':
        model=Node2Vec_LP(data,128,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=0.01)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,1000)
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
        
    elif args.lp_model == 'gcn':
        model=GCN_LP(data.x.shape[1],128,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=0.01)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,1000)
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
        
    elif args.lp_model == 'gat':
        model=GAT_LP(data.x.shape[1],128,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=0.01)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,1000)
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
        
    elif args.lp_model == 'gca':
        model=GCA_LP(data,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=0.001)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,5000)
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
    elif args.lp_model == 'metamodel':
        model=MetaModel_LP(data,device=args.device)
        optimizer = None
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,1)
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
        
    elif args.lp_model == 'vgnae':
        model=VGNAE_LP(data,128,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=0.01)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,300)
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
    return best_val_result,best_test_result,best_scores
def run_clean_exp_on_large_datasset(args,data):
    from lp_models_for_large_dataset.DeepWalk import DeepWalk_LP
    from lp_models_for_large_dataset.GAT import GAT_LP
    from lp_models_for_large_dataset.GCN import GCN_LP
    from lp_models_for_large_dataset.VGNAE import VGNAE_LP
    print('run clean exp on :')
    print(data)
    #查看lp_model是不是符合规定,如果不符合规定就报错，并提示错误信息
    assert args.lp_model in ['deepwalk','gcn', 'gat','vgnae'] 
    best_val_result,best_test_result,best_scores=None,None,None
    #根据args.lp_model的值来选择不同的模型
    if args.lp_model == 'deepwalk':
        model=DeepWalk_LP(data,128,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=1e-5)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,30)
        #保存model, best_val_result, best_test_result, best_scores, args, data
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
    elif args.lp_model == 'gcn':
        model=GCN_LP(data.x.shape[1],128,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=1e-5)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,30)
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
        
    elif args.lp_model == 'gat':
        model=GAT_LP(data.x.shape[1],128,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=1e-5)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,30)
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
        
    elif args.lp_model == 'vgnae':
        model=VGNAE_LP(data,128,device=args.device)
        optimizer = optim.Adam(model.model.parameters(), lr=1e-5)
        best_val_result,best_test_result,best_scores=model.get_result(data,optimizer,30)
        save_results(model, best_val_result, best_test_result, best_scores, args, data)
    return best_val_result,best_test_result,best_scores
@profile
def run_poisoning_exp_integrity(args,data_o):
    data=copy.deepcopy(data_o)
    if args.dataset=='zhihu' or args.dataset=='quora':
        data=RemoveDuplicatedEdges()(data)
    
    #The validation set is used to test the overall performance, 
    # while the test set is used to test the system's performance on the target edge set.
    posioned_edge_index=torch.concat([data_o.train_pos_edge_index,data_o.val_pos_edge_index],axis=1)
    adj=sp.csr_matrix((np.ones(posioned_edge_index.shape[1]),
        (posioned_edge_index[0], posioned_edge_index[1])), shape=(data.num_nodes, data.num_nodes))
    idx_train=np.array(range(data.num_nodes))[data.train_mask]
    idx_val=np.array(range(data.num_nodes))[data.val_mask]
    idx_test=np.array(range(data.num_nodes))[data.test_mask]
    idx_unlabeled = np.union1d(idx_val, idx_test)
    surrogate=None
    if args.attack_method in ['metattack', 'minmax', 'pgd']:
    # Setup Surrogate model
        surrogate = deeprobust.graph.defense.GCN(nfeat=data.x.shape[1], nclass=data.y.max().item() + 1,
                        nhid=16, dropout=0, with_relu=False, with_bias=False, device=args.device).to(args.device)
        surrogate.fit(data.x, adj, data.y, idx_train, idx_val, patience=30)
    # Setup Attack Model
    #n_perturbation  is followed by the paper of CLGA,/2是因为是对称矩阵
    n_perturbation = int(args.attack_rate * data.num_edges/2)
    if args.dataset=='zhihu' or args.dataset=='quora':
        n_perturbation = int(args.attack_rate * data.train_pos_edge_index.shape[1]/2)

    modified_adj=None
    if args.attack_method in ['metattack', 'minmax', 'pgd', 'random', 'dice']:
        attack_model = baseline_attacks(args, surrogate,adj, data.x, data.y, n_perturbation,args.device,idx_train,idx_unlabeled)
        modified_adj = attack_model.modified_adj
    elif args.attack_method == 'clga':
        model = CLGA(args=args,data=data,device=args.device)
        modified_adj = model.attack()
    elif args.attack_method == 'aalp':
        model = AALP_Integrity(args=args,data=data)
        modified_adj = model.attack()
    elif args.attack_method == 'viking':
        labels=np.array(data.y)
        L = (labels == np.unique(labels)[:, None]).astype(int).T
        modified_adj = viking(adj,attack='our', n_flips=n_perturbation, dim=32, window_size=5, L=L)
        modified_adj=torch.Tensor(modified_adj.todense())
    elif args.attack_method == 'nodeembeddingattack':
        modified_adj=run_poisoning_exp_integrity_nodeattack(args,data_o)
    if args.attack_method in ['random', 'dice']:
        if args.dataset!='zhihu' and args.dataset!='quora':
            modified_adj = torch.Tensor(modified_adj.todense())
        else:
            data=modifiedAdj2data_large(modified_adj,data_o)
            with open(os.path.join(args.outputs,'adj/{}_{}_{}_{}_modifiedData.pkl'.format(args.dataset,args.attack_method,args.attack_goal,args.attack_rate)),'wb') as f:
                pickle.dump(data,f)
    
    try:
        modified_adj.cpu()
    except:
        pass
    posioned_edge_index=torch.tensor(modified_adj).nonzero().T

        
    G = nx.Graph()
    for i in range(data_o.num_nodes):
        G.add_node(i)
    edge_index = np.array(posioned_edge_index.cpu())
    for j in range(edge_index.shape[1]):
        G.add_edge(edge_index[0][j], edge_index[1][j])
    edge_index = np.array(data_o.test_pos_edge_index.cpu())
    for j in range(edge_index.shape[1]):
        G.add_edge(edge_index[0][j], edge_index[1][j])
    adj_matrix = nx.adjacency_matrix(G).todense()
    adj_matrix=torch.tensor(adj_matrix)
    #使用pickle将modified_adj保存到args.outputs文件夹下,文件名由args里面的dataset、attack_method、attack_rate、modifiedAdj组成
    with open(os.path.join(args.outputs,'adj/{}_{}_{}_{}_modifiedAdj.pkl'.format(args.dataset,args.attack_method,args.attack_goal,args.attack_rate)),'wb') as f:
        pickle.dump(adj_matrix,f)
def run_poisoning_exp_integrity_nodeattack(args,data):
    device=args.device
    #n_perturbation  is followed by the paper of CLGA,/2是因为是对称矩阵
    n_perturbation = int(args.attack_rate * data.num_edges/2)
    posioned_edge_index=torch.concat([data.train_pos_edge_index,data.val_pos_edge_index,data.test_pos_edge_index],axis=1)
    new_edge_index, mapping, mask_all = process_isolated_nodes(posioned_edge_index)
    test_pos_edge_index=new_edge_index[:,-data.test_pos_edge_index.shape[1]:]
    new_edge_index=new_edge_index[:,:-data.test_pos_edge_index.shape[1]]
    new_edge_index, mapping, mask = process_isolated_nodes(new_edge_index)
    new_num_nodes = int(new_edge_index.max() + 1)
    edge_sp_adj = torch.sparse.FloatTensor(new_edge_index.to(device),
                                           torch.ones(new_edge_index.shape[1]).to(device),
                                           [new_num_nodes, new_num_nodes])
    edge_adj = edge_sp_adj.to_dense().cpu().numpy()
    adj = csr_matrix(edge_adj)
    model = NodeEmbeddingAttack()
    model.attack(adj, attack_type='remove', n_perturbations=n_perturbation)
    modified_adj = torch.Tensor(model.modified_adj.todense())
    edge_index = modified_adj.nonzero().T
    posioned_edge_index = restore_isolated_ndoes_int(edge_index, mask)
    posioned_edge_index = restore_isolated_ndoes_int(posioned_edge_index, mask_all)
    test_pos_edge_index =restore_isolated_ndoes_int(test_pos_edge_index,mask_all)
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
    edge_index = np.array(posioned_edge_index.cpu())
    for j in range(edge_index.shape[1]):
        G.add_edge(edge_index[0][j], edge_index[1][j])
    adj_matrix = nx.adjacency_matrix(G).todense()
    adj_matrix=torch.tensor(adj_matrix)
    
    return adj_matrix
    
def run_poisoning_exp_availability_nodeattack(args,data):
    device=args.device
    #n_perturbation  is followed by the paper of CLGA,/2是因为是对称矩阵
    n_perturbation = int(args.attack_rate * data.num_edges/2)
    new_edge_index, mapping, mask = process_isolated_nodes(data.edge_index)
    new_num_nodes = int(new_edge_index.max() + 1)
    edge_sp_adj = torch.sparse.FloatTensor(new_edge_index.to(device),
                                           torch.ones(new_edge_index.shape[1]).to(device),
                                           [new_num_nodes, new_num_nodes])
    edge_adj = edge_sp_adj.to_dense().cpu().numpy()
    adj = csr_matrix(edge_adj)
    model = NodeEmbeddingAttack()
    model.attack(adj, attack_type='remove', n_perturbations=n_perturbation)
    modified_adj = torch.Tensor(model.modified_adj.todense())
    edge_index = modified_adj.nonzero().T
    restored_edge_index = restore_isolated_ndoes_int(edge_index, mask)
    edge_sp_adj = torch.sparse.FloatTensor(restored_edge_index.to(device),
                                           torch.ones(restored_edge_index.shape[1]).to(device),
                                           [data.num_nodes, data.num_nodes])
    return edge_sp_adj
@profile
def run_poisoning_exp_availability(args,data_o):
    data=copy.deepcopy(data_o)
    if args.dataset=='zhihu' or args.dataset=='quora':
        data=RemoveDuplicatedEdges()(data)
    adj=sp.csr_matrix((np.ones(data.edge_index.shape[1]),
        (data.edge_index[0], data.edge_index[1])), shape=(data.num_nodes, data.num_nodes))
    
    idx_train=np.array(range(data.num_nodes))[data.train_mask]
    idx_val=np.array(range(data.num_nodes))[data.val_mask]
    idx_test=np.array(range(data.num_nodes))[data.test_mask]
    idx_unlabeled = np.union1d(idx_val, idx_test)
    surrogate=None
    if args.attack_method in ['metattack', 'minmax', 'pgd']:
    # Setup Surrogate model
        surrogate = deeprobust.graph.defense.GCN(nfeat=data.x.shape[1], nclass=data.y.max().item() + 1,
                        nhid=16, dropout=0, with_relu=False, with_bias=False, device=args.device).to(args.device)
        surrogate.fit(data.x, adj, data.y, idx_train, idx_val, patience=30)
    # Setup Attack Model
    #n_perturbation  is followed by the paper of CLGA,/2是因为是对称矩阵
    n_perturbation = int(args.attack_rate * data.num_edges/2)
    if args.dataset=='zhihu' or args.dataset=='quora':
        n_perturbation = int(args.attack_rate * data.train_pos_edge_index.shape[1]/2)

    modified_adj=None
    if args.attack_method in ['metattack', 'minmax', 'pgd', 'random', 'dice']:
        attack_model = baseline_attacks(args, surrogate,adj, data.x, data.y, n_perturbation,args.device,idx_train,idx_unlabeled)
        modified_adj = attack_model.modified_adj
    elif args.attack_method == 'clga':
        model = CLGA(args=args,data=data,device=args.device)
        modified_adj = model.attack()
    elif args.attack_method == 'aalp':
        model = AALP_Availability(args=args,data=data)
        modified_adj = model.attack()
    elif args.attack_method == 'viking':
        labels=np.array(data.y)
        L = (labels == np.unique(labels)[:, None]).astype(int).T
        modified_adj = viking(adj,attack='our', n_flips=n_perturbation, dim=32, window_size=5, L=L)
        modified_adj=torch.Tensor(modified_adj.todense())
    elif args.attack_method == 'nodeembeddingattack':
        modified_adj=run_poisoning_exp_availability_nodeattack(args,data_o)
        modified_adj=modified_adj.to_dense()
    if args.attack_method in ['random', 'dice']:
        if args.dataset!='zhihu' and args.dataset!='quora':
            modified_adj = torch.Tensor(modified_adj.todense())
        else:
            data=modifiedAdj2data_large(modified_adj,data_o)
            with open(os.path.join(args.outputs,'adj/{}_{}_{}_{}_modifiedData.pkl'.format(args.dataset,args.attack_method,args.attack_goal,args.attack_rate)),'wb') as f:
                pickle.dump(data,f)
    
    try:
        modified_adj.cpu()
    except:
        pass
    #使用pickle将modified_adj保存到args.outputs文件夹下,文件名由args里面的dataset、attack_method、attack_rate、modifiedAdj组成
    with open(os.path.join(args.outputs,'adj/{}_{}_{}_{}_modifiedAdj.pkl'.format(args.dataset,args.attack_method,args.attack_goal,args.attack_rate)),'wb') as f:
        pickle.dump(modified_adj,f)

if __name__ == '__main__':
    print('-'*30,'NEW EXP','-'*30)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Cora',help="['Cora','CiteSeer','PubMed']")
    parser.add_argument('--exp_type', type=str, default='poisoning',help="[clean,evasion,poisoning]")
    parser.add_argument('--lp_model', type=str, default='gcn',help="""['deepwalk','node2vec','gcn', 'gat', 'gca','vgnae']""")
    parser.add_argument('--attack_method', type=str, default='noattack',help="""['noattack','random', 'dice','metattack', 'minmax', 'pgd','clga','aalp']""")
    parser.add_argument('--attack_goal', type=str, default='integrity',help="""['integrity','availability']""")
    parser.add_argument('--attack_rate', type=float, default=0.05)
    parser.add_argument('--outputs', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default='32')
    parser.add_argument('--datasetsDir', type=str, default='../datasets')

    args = parser.parse_args()
    filename = '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method,args.attack_goal, str(args.attack_rate), str(args.seed), 'cuda'])
    child_process=subprocess.Popen(['python', './utils/gpu_logger.py',filename])
    print('--------------------ddd------------------')


    #设置随机数种子,只能保证数据集的划分是一致的，但是模型的初始化参数是随机的，所以模型的初始化参数是不一致的
    set_random_seed(args.seed)
    data=load_dataset(args.dataset)
    print(args)
    print(data)
    best_val_result,best_test_result,best_scores=None,None,None
    if args.exp_type=='clean':
        if args.dataset=='zhihu' or args.dataset=='quora':
            best_val_result,best_test_result,best_score=run_clean_exp_on_large_datasset(args,data)
        else:
            best_val_result,best_test_result,best_score=run_clean_exp(args,data)
    elif args.exp_type=='poisoning':
        if args.attack_goal=='integrity':
            run_poisoning_exp_integrity(args,data)
        elif args.attack_goal=='availability':
            run_poisoning_exp_availability(args,data)
    else:
        raise ValueError('exp_type should be clean, evasion or poisoning')
    print('args:',args)
    child_process.kill()

