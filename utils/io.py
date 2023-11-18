import os
import os.path as osp
import pickle
from utils.data import  QADADataset
import torch
import os
import pickle
import random
import warnings
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



from utils.build_graph import build_graph_dgl
from utils.utils import generate_dataset
def readFile(dataset,filePath):
    with open(f"./dataset/{dataset}/{filePath}.pkl", 'rb') as file:
        return pickle.loads(file.read())
def load_data(args):
    Answers_train=readFile(dataset=args.dataset,filePath='Answers_train')
    Answers_dev=readFile(dataset=args.dataset,filePath='Answers_dev')
    Answers_test=readFile(dataset=args.dataset,filePath='Answers_test')
    Questions=readFile(dataset=args.dataset,filePath='Questions')
    Users=readFile(dataset=args.dataset,filePath='Users')
    g=build_graph_dgl(Users,Questions=Questions,Answers_dev=Answers_dev,Answers_test=Answers_test)

    def fillna_list(x):
        mask=x.isnull()
        for i in range(len(mask)):
            if mask[i]:
                x[i]=[]
        return x
    Users=Users.apply(fillna_list,axis=1)
    nodetype2edgetype={'user':['user_ask_question','user_comment_answer','user_follower_user','user_following_question','user_following_topic','user_following_user','user_write_answer'],
                   'answer':['answer_comment_user','answer_have_question','answer_write_user'],
                   'question':['question_ask_user','question_belongto_topic','question_following_user','question_have_answer'],
                   'topic':['topic_belongto_question','topic_following_user']}
    nodetype2id={'user':0,'answer':1,'topic':2,'question':3}
    id2nodetype={0:'user',1:'answer',2:'topic',3:'question'}
    def sample_paths(uid,aid,qid,max_path3=4,max_path4=10):
        Users.index=Users.uid
        Questions.index=Questions.questionId
        #path_1:user-answer
        paths=[[uid,aid,0,0,0]]
        masks=[[False,False,True,True,True]]
        node_types=[['user','answer','answer','answer','answer']]
        answer_position=[1]

        '''
        #path_2:user-question-answer
        user_followedQuestions=Users.loc[uid].followedQuestions
        if qid in user_followedQuestions:
            paths.append([uid,qid,aid,0,0])
            node_types.append(['user','question','answer','answer','answer'])
            masks.append([False,False,False,True,True])
            answer_position.append(2)
        '''
        
        #path_3:user-topic-question-answer
        user_followedTopics=set(Users.loc[uid].followedTopics)
        answer_topic=set(Questions.loc[qid].questionTopics)
        commen_topic=user_followedTopics&answer_topic
        for ct in commen_topic:
            if len(paths)<max_path3:
                paths.append([uid,ct,qid,aid,0])
                node_types.append(['user','topic','question','answer','answer'])
                masks.append([False,False,False,False,True])
                answer_position.append(3)
        #path4:user-question-topic-question-answer
        user_question=set(Users.loc[uid].followedQuestions)|set(Users.loc[uid].askedQuestions)
        for question_id in user_question:
            question_topic=set(Questions.loc[question_id].questionTopics)
            commen_topic=question_topic&answer_topic
            for ct in commen_topic:
                if len(paths)<max_path4:
                    paths.append([uid,question_id,ct,qid,aid])
                    node_types.append(['user','question','topic','question','answer']) 
                    masks.append([False,False,False,False,False])
                    answer_position.append(4)
        #padding to max_paths
        for i in range(len(paths),max_path4):
            paths.append([uid,aid,0,0,0])
            node_types.append(['user','answer','user','user','user']) 
            masks.append([False,False,True,True,True])
            answer_position.append(1)
        node_types=[[nodetype2id[nodetype] for nodetype in path] for path in node_types]
        return paths,node_types,masks,answer_position

    def sample_user_paths(x):
        node_paths,type_paths,masks,answer_position=sample_paths(x[0],x[1],x[2])
        x['node_paths']=node_paths
        x['type_paths']=type_paths
        x['masks']=masks
        x['answer_position']=answer_position
        return x

    Answers_train=Answers_train[['uid', 'answerId', 'questionId']].apply(sample_user_paths,axis=1)
    allUsersUid=Users.uid.values.tolist()
    def sampleNeg(x):
        temp=random.sample(allUsersUid,10)
        uids=[]
        i=0
        while len(uids)<5:
            if x!=temp[i]:
                uids.append(temp[i])
            i+=1
        return uids

    Answers_train['neg_uid']=Answers_train['uid'].apply(sampleNeg)
    train_pos_dataset=QADADataset(Answers_train.node_paths.values.tolist(),Answers_train.type_paths.values.tolist(),Answers_train.masks.values.tolist(),Answers_train.answer_position.values.tolist(),[1]*Answers_train.shape[0])
    Answers_train_neg=[]
    for uids,aid,qid in zip(Answers_train['neg_uid'],Answers_train['answerId'],Answers_train['questionId']):
        for uid in uids:
            Answers_train_neg.append([uid,aid,qid])
    Answers_train_neg=pd.DataFrame(Answers_train_neg,columns=['uid', 'answerId', 'questionId'])
    Answers_train_neg=Answers_train_neg[['uid', 'answerId', 'questionId']].apply(sample_user_paths,axis=1)

    train_neg_dataset=QADADataset(Answers_train_neg.node_paths.values.tolist(),Answers_train_neg.type_paths.values.tolist(),Answers_train_neg.masks.values.tolist(),Answers_train_neg.answer_position.values.tolist(),[0]*Answers_train_neg.shape[0])

    def build_test_dataset(Answers_dev):
        Answers_dev_pd=[]
        for uids,aid,qid in zip(Answers_dev['users'],Answers_dev['answerId'],Answers_dev['questionId']):
            for uid in uids:
                Answers_dev_pd.append([uid,aid,qid])
        Answers_dev_pd=pd.DataFrame(Answers_dev_pd,columns=['uid', 'answerId', 'questionId'])
        Answers_dev_pd['label']=torch.tensor(Answers_dev.labels.values.tolist()).view(-1).tolist()
        Answers_dev_pd=Answers_dev_pd[['uid', 'answerId', 'questionId']].apply(sample_user_paths,axis=1)
        Answers_dev_dataset=QADADataset(Answers_dev_pd.node_paths.values.tolist(),Answers_dev_pd.type_paths.values.tolist(),Answers_dev_pd.masks.values.tolist(),Answers_dev_pd.answer_position.values.tolist(),[0]*Answers_dev_pd.shape[0])
        return Answers_dev_dataset
    Answers_dev_dataset=build_test_dataset(Answers_dev)
    Answers_test_dataset=build_test_dataset(Answers_test)
    return g,train_pos_dataset,train_neg_dataset,Answers_dev_dataset,Answers_test_dataset

def load_dataset(dataset):
    path = osp.expanduser('dataset')
    path = osp.join(path, dataset)
    #判断是否向存在
    if not osp.exists(osp.join(path, 'data.pt')):
        generate_dataset(dataset)
    data = torch.load(osp.join(path, 'data.pt'))
    return data

#使用pickle将模型保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed组成
def save_model(model, args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method,args.attack_goal,  str(args.attack_rate), str(args.seed)]))
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

#使用pickle将clean模型从args.outputs文件夹下读取出来,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed组成
def load_clean_model(args):
    filename = osp.join(args.outputs, '_'.join([args.dataset, 'clean', args.lp_model, 'noattack','integrity',  '0.05', str(args.seed)]))
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model
#使用pickle将模型的best_val_result结果保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed和best_val_result组成
def save_best_val_result(best_val_result, args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method,args.attack_goal, str(args.attack_rate), str(args.seed), 'best_val_result']))
    with open(filename, 'wb') as f:
        pickle.dump(best_val_result, f)

#使用pickle将模型的best_test_result结果保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed和best_test_result组成
def save_best_test_result(best_test_result, args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method,args.attack_goal,str(args.attack_rate), str(args.seed), 'best_test_result']))
    with open(filename, 'wb') as f:
        pickle.dump(best_test_result, f)

#使用pickle将模型的best_scores结果保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed和best_scores组成
def save_best_scores(best_scores, args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method,args.attack_goal, str(args.attack_rate), str(args.seed), 'best_scores']))
    with open(filename, 'wb') as f:
        pickle.dump(best_scores, f)

#使用pickle将args保存到args.outputs文件夹下,文件名由args里面的dataset、exp_type、lp_model、attack_method、attack_rate、seed和args组成
def save_args(args):
    if not osp.exists(args.outputs):
        os.makedirs(args.outputs)
    filename = osp.join(args.outputs, '_'.join([args.dataset, args.exp_type, args.lp_model, args.attack_method, args.attack_goal,str(args.attack_rate), str(args.seed), 'args']))
    with open(filename, 'wb') as f:
        pickle.dump(args, f)

#保存model, best_val_result, best_test_result, best_scores, args, data
def save_results(model, best_val_result, best_test_result, best_scores, args, data):
    if args.lp_model=='gcn' or args.lp_model=='gat':
        save_model(model, args)
    save_best_val_result(best_val_result, args)
    save_best_test_result(best_test_result, args)
    #save_best_scores(best_scores, args)
    save_args(args)

#读取modifiedAdj.pkl
def load_modifiedAdj(args):
    if args.dataset=='zhihu' or args.dataset=='quora':
        filename=f'{args.outputs}/adj/{args.dataset}_{args.attack_method}_{args.attack_goal}_{args.attack_rate}_modifiedData.pkl'
    else:
        filename=f'{args.outputs}/adj/{args.dataset}_{args.attack_method}_{args.attack_goal}_{args.attack_rate}_modifiedAdj.pkl'
    with open(filename, 'rb') as f:
        modifiedAdj = pickle.load(f)
    return modifiedAdj