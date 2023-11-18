from utils.utils import (set_random_seed)
import argparse
import logging
import os
import copy
import datetime
import os
import pickle
import warnings
from lp_models.MetaTransformer import ExplainableGraphTransformer,RGCNClassfier
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from utils.io import load_data
import tqdm
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

def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()
def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
def calculate_mrr(target, target_rank):
    mrr = 0.0
    for i in range(len(target)):
        if target[i] in target_rank[i]:
            rank = target_rank[i].index(target[i]) + 1
            mrr += 1.0 / rank
    mrr /= len(target)
    return mrr
def train(model, g, train_pos_loader, train_neg_loader, Answers_dev_dataloader,Answers_test_dataloader,optimizer, epoch):
    best_mrr,best_auc=0,0
    best_mrr_test,best_auc_test=0,0
    for epoch in range(epoch):
        model.train()
        total_loss, total_acc = 0, 0
        # 做training
        with tqdm(total=len(train_pos_loader), desc="training") as pbar:
            for pos_inputs,neg_inputs in zip(train_pos_loader,train_neg_loader):
                optimizer.zero_grad() 
                outputs_pos = model(g,*pos_inputs)
                outputs_neg = model(g,*neg_inputs).view(outputs_pos.shape[0],-1) 
                loss = compute_loss(outputs_pos, outputs_neg)
                loss.backward()
                optimizer.step() 
                total_loss += loss.item()
                pbar.set_postfix({'loss':'{:.3f}'.format(loss.item())})
                pbar.update(1)
        
        # 做validation
        mrr_dev,auc_dev=test(model, g, Answers_dev_dataloader)
        print('epoch:{},loss:{},mrr_dev:{},auc_dev:{}'.format(epoch,total_loss,mrr_dev,auc_dev))
        logging.info('epoch:{},loss:{},mrr_dev:{},auc_dev:{}'.format(epoch,total_loss,mrr_dev,auc_dev))

        #做test
        mrr_test,auc_test=test(model, g, Answers_test_dataloader)
        print('epoch:{},loss:{},mrr_test:{},auc_test:{}'.format(epoch,total_loss,mrr_test,auc_test))
        logging.info('epoch:{},loss:{},mrr_test:{},auc_test:{}'.format(epoch,total_loss,mrr_test,auc_test))

        if mrr_dev>best_mrr:
            best_mrr=mrr_dev
            best_auc=auc_dev
            best_mrr_test=mrr_test
            best_auc_test=auc_test
            torch.save(model.state_dict(), os.path.join(args.outputs, f'best_mrr_model_{args.dataset}.pt'))
        
    return best_mrr,best_auc,best_mrr_test,best_auc_test


def test(model, g, Answers_dev_dataloader):
    predicts=[]
    with torch.no_grad():
        model.eval()
        for inputs in Answers_dev_dataloader:
            outputs = model(g,*inputs)
            predicts.append(outputs.detach().cpu())
    predicts=torch.concat(predicts).view(-1,5)
    #MRR
    mrr=calculate_mrr([0]*len(predicts),predicts.argsort(dim=1,descending=True).tolist())

    #AUC
    labels=torch.zeros_like(predicts)
    labels[:,0]=1
    auc=roc_auc_score(labels.view(-1).numpy(),predicts.view(-1).numpy())
    return mrr,auc

if __name__ == '__main__':
    logging.basicConfig(filename='log_attack.txt', level=logging.CRITICAL)
    print('-'*30, 'NEW EXP', '-'*30)
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='zhihu', help="['zhihu','quora']")
    parser.add_argument('--exp_type', type=str, default='attack', help="[attack,poisoning]")
    parser.add_argument('--lp_model', type=str, default='metatransformer',
                        help="""['deepwalk','node2vec','gcn', 'gat', 'gca','vgnae','metatransformer']""")
    parser.add_argument('--poisoning_method', type=str, default='noattack',
                        help="""['noattack','random', 'dice','metattack', 'minmax', 'pgd','clga','aalp','viking']""")
    parser.add_argument('--poisoning_goal', type=str, default='integrity', help="""['integrity','availability']""")
    parser.add_argument('--poisoning_rate', type=float, default=0.05)
    parser.add_argument('--outputs', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default='32')
    parser.add_argument('--datasetsDir', type=str, default='datasets')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset_cache', type=str, default='zhihu_cahe', help="['zhihu_cahe','quora_cahe']")

    #epoch
    parser.add_argument('--epoch', type=int, default=6)
    args = parser.parse_args()
    set_random_seed(args.seed)
    args.dataset_cache=args.dataset+'_cache'
    best_auc,best_mrr,best_auc_test,best_mrr_test=0,0,0,0
    if args.lp_model == 'metatransformer' or args.lp_model == 'rgcn':

        if not os.path.isfile(f"./dataset/{args.dataset_cache}/train_pos_dataset.pkl"):
            g,train_pos_dataset,train_neg_dataset,Answers_dev_dataset,Answers_test_dataset=load_data(args)

            with open(f"./dataset/{args.dataset_cache}/g.pkl", 'wb') as f:
                pickle.dump(g, f)
            with open(f"./dataset/{args.dataset_cache}/train_pos_dataset.pkl", 'wb') as f:
                pickle.dump(train_pos_dataset, f)
            with open(f"./dataset/{args.dataset_cache}/train_neg_dataset.pkl", 'wb') as f:
                pickle.dump(train_neg_dataset, f)
            with open(f"./dataset/{args.dataset_cache}/Answers_dev_dataset.pkl", 'wb') as f:
                pickle.dump(Answers_dev_dataset, f)
            with open(f"./dataset/{args.dataset_cache}/Answers_test_dataset.pkl", 'wb') as f:
                pickle.dump(Answers_test_dataset, f)
        else:
            with open(f"./dataset/{args.dataset_cache}/g.pkl", 'rb') as f:
                g = pickle.load(f)
            with open(f"./dataset/{args.dataset_cache}/train_pos_dataset.pkl", 'rb') as f:
                train_pos_dataset = pickle.load(f)
            with open(f"./dataset/{args.dataset_cache}/train_neg_dataset.pkl", 'rb') as f:
                train_neg_dataset = pickle.load(f)
            with open(f"./dataset/{args.dataset_cache}/Answers_dev_dataset.pkl", 'rb') as f:
                Answers_dev_dataset = pickle.load(f)
            with open(f"./dataset/{args.dataset_cache}/Answers_test_dataset.pkl", 'rb') as f:
                Answers_test_dataset = pickle.load(f)


        embeddingDict={nodeType:g.num_nodes(nodeType) for nodeType in g.ntypes}
        model=''
        if args.lp_model == 'rgcn':
            model = RGCNClassfier(embeddingDict,128, g.etypes,max_length=5,max_paths=10)
        else:
            model = ExplainableGraphTransformer(embeddingDict,128, g.etypes,max_length=5,num_head=8,device=args.device,max_paths=10)
        model.to(args.device)
        g=g.to(args.device)
        train_pos_loader = torch.utils.data.DataLoader(dataset = train_pos_dataset,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = 8)

        train_neg_loader = torch.utils.data.DataLoader(dataset = train_neg_dataset,
                                                    batch_size = args.batch_size*5,
                                                    shuffle = False,
                                                    num_workers = 8)
        Answers_dev_dataloader = torch.utils.data.DataLoader(dataset = Answers_dev_dataset,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = 8)
        Answers_test_dataloader = torch.utils.data.DataLoader(dataset = Answers_test_dataset,
                                            batch_size = args.batch_size,
                                            shuffle = False,
                                            num_workers = 8)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        #train
        best_mrr,best_auc,best_mrr_test,best_auc_test=train(model,g,train_pos_loader,train_neg_loader,Answers_dev_dataloader,Answers_test_dataloader,optimizer,args.epoch)

    else:
        pass
logging.critical(f'time:{datetime.datetime.now()}')
logging.critical('args:{}'.format(args))
logging.critical('best_val_result_mrr:{}'.format(best_mrr))
logging.critical('best_val_result_auc:{}'.format(best_auc))
logging.critical('best_test_result_mrr:{}'.format(best_mrr_test))
logging.critical('best_val_result_auc:{}'.format(best_auc_test))


