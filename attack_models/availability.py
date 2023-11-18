
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, Dataset
import random
import copy
def activation(x):
    return torch.sigmoid(x)
class Messaging(Module):
    def __init__(self,adj,node_num,feature_num,device):
        super(Messaging, self).__init__()
        self.device=device
        adj=torch.tensor(adj)
        self.adj=torch.tensor(adj/(adj.sum(axis=1)+1e-8),requires_grad=False,dtype=torch.float32).to(self.device)
        self.W2=torch.nn.Parameter(copy.deepcopy(self.adj.cpu())+torch.randn(( self.adj.shape[0], self.adj.shape[1])),requires_grad=True)
        self.node_num=node_num
        self.feature_num=feature_num
        self.eye=torch.tensor(torch.eye(adj.shape[0]),requires_grad=False,dtype=torch.float32).to(self.device)
    def forward(self,src,dst):
        node_feature=torch.randn((self.node_num, 1000),requires_grad=False).to(self.device)
        adj=activation(self.W2)
        node_feature_gnn=torch.mm(torch.mm(adj,adj)+0.5*adj+self.eye,node_feature)
        return node_feature_gnn[src],node_feature_gnn[dst]
class EdgeDataset(Dataset):
    def __init__(self, src_nodes, dst_nodes):
        super(EdgeDataset, self).__init__()
        assert len(src_nodes) == len(dst_nodes), "The lengths of src_nodes and dst_nodes must be the same."
        self.src_nodes = src_nodes
        self.dst_nodes = dst_nodes

    def __getitem__(self, index):
        src_node = self.src_nodes[index]
        dst_node = self.dst_nodes[index]
        return torch.tensor(src_node), torch.tensor(dst_node)

    def __len__(self):
        return len(self.src_nodes)

class AALP_Availability(object):
    #初始化
    def __init__(self, args,data):
        self.args=args
        self.data=data
        self.Lambda=1.0
        if args.dataset=='Cora':
            self.Lambda=0.1
    def attack(self):
        G = nx.Graph()
        # 添加所有节点
        for i in range(self.data.num_nodes):
            G.add_node(i)

        # 添加训练集所有边
        edge_index = np.array(self.data.edge_index)
        for j in range(edge_index.shape[1]):
            G.add_edge(edge_index[0][j], edge_index[1][j])


        adj_matrix = nx.adjacency_matrix(G).todense()
        model=Messaging(adj_matrix,self.data.num_nodes,1024,self.args.device)
        model.to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in tqdm.tqdm(range(300)):
            losses=0
            optimizer.zero_grad()
            val_src_pos,val_dst_pos=model(self.data.edge_index[0].to(self.args.device),self.data.edge_index[1].to(self.args.device))
            val_src_neg,val_dst_neg=model(self.data.edge_index[0].to(self.args.device),self.data.edge_index[1].to(self.args.device))

            similarity_val_pos = torch.sum(-F.cosine_similarity(val_src_pos, val_dst_pos, dim=0))
            similarity_val_neg = torch.sum(F.cosine_similarity(val_src_neg, val_dst_neg, dim=0))
            adj=activation(model.W2)
            diff=adj-model.adj
            reg_loss = torch.sum(torch.square(diff))
            loss=similarity_val_pos+similarity_val_neg+reg_loss*self.Lambda
            losses+=loss.item()
            loss.backward()
            optimizer.step()
            tqdm.tqdm.write(f"Loss {losses:.4f}")
        #"""
        adj=activation(model.W2)
        adj=np.array(adj.detach().cpu())
        adj_matrix=np.array(model.adj.detach().cpu())
        adj=np.abs(adj-adj_matrix)
        adj=torch.tensor(adj)
        up_adj=adj.mul(torch.ones_like(adj)-torch.tril(torch.ones_like(adj)))
        down_adj=adj.mul(torch.tril(torch.ones_like(adj))-torch.eye(adj.shape[0]))
        adj=up_adj+down_adj
        adj=np.array(adj)
        indices = np.argsort(adj, axis=None)[::-1]
        srcs,dsts = np.unravel_index(indices, adj.shape)
        idxs = [(src,dst) for src,dst in zip(srcs[:int(self.data.num_edges*self.args.attack_rate)],dsts[:int(self.data.num_edges*self.args.attack_rate)])]  # 要修改的下标位置
        
        #"""
        '''
        adj=activation(model.W2)
        adj=adj.detach().cpu()
        adj_matrix=model.adj.detach().cpu()
        adj=torch.abs(adj-adj_matrix)
        up_adj=adj.mul(torch.ones_like(adj)-torch.tril(torch.ones_like(adj)))
        #down_adj=adj.mul(torch.tril(torch.ones_like(adj))-torch.eye(adj.shape[0]))
        adj=up_adj#+down_adj.T
        #adj=torch.maximum(up_adj,down_adj.T)
        adj=np.array(adj.detach().cpu())
        indices = np.argsort(adj, axis=None)[::-1]
        srcs,dsts = np.unravel_index(indices, adj.shape)
        idxs = [(src,dst) for src,dst in zip(srcs[:int(self.data.num_edges*(self.args.attack_rate/2))],dsts[:int(self.data.num_edges*(self.args.attack_rate/2))])]
        idxs =idxs+ [(dst,src) for src,dst in zip(srcs[:int(self.data.num_edges*(self.args.attack_rate/2))],dsts[:int(self.data.num_edges*(self.args.attack_rate/2))])]
        '''
        
        
        adj_matrix[np.array(idxs)[:,0], np.array(idxs)[:,1]] = 1 - adj_matrix[np.array(idxs)[:,0], np.array(idxs)[:,1]]
        posioned_edge_index=torch.tensor(adj_matrix).nonzero().T

        
        G = nx.Graph()
        # 添加所有节点
        for i in range(self.data.num_nodes):
            G.add_node(i)
        # 添加训练集所有边
        edge_index = np.array(posioned_edge_index.cpu())
        for j in range(edge_index.shape[1]):
            G.add_edge(edge_index[0][j], edge_index[1][j])
        adj_matrix = nx.adjacency_matrix(G).todense()
        adj_matrix=torch.tensor(adj_matrix)
        return adj_matrix


