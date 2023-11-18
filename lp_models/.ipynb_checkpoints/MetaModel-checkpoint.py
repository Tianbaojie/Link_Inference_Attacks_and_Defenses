import copy
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import add_self_loops, structured_negative_sampling

from lp_models.BaseLP import BaseLp
from utils.evaluation import LPEvaluator
import networkx as nx

class MetaModel(nn.Module):
    def __init__(self,adj,device):
        super(MetaModel, self).__init__()
        self.device=device
        adj=torch.tensor(adj)
        self.adj=torch.tensor(adj/(adj.sum(axis=1)+1e-8),requires_grad=False,dtype=torch.float32).to(self.device)
        self.eye=torch.tensor(torch.eye(adj.shape[0]),requires_grad=False,dtype=torch.float32).to(self.device)
    def forward(self,X):
        node_feature_gnn=torch.mm(torch.mm(self.adj,self.adj)+0.5*self.adj+self.eye,X)
        return node_feature_gnn
#基于GCN的链接预测模型
class MetaModel_LP(BaseLp):
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
    def __init__(self,data,device):
        super(MetaModel_LP,self).__init__()
        self.device=device
        self.model=MetaModel(self.get_adj(data),device).to(device)
    def train(self,data,optimizer=None,epochs=1):
        '''
        This function is used to train the model.
        eval_result is the result of the model on the validation set.
        test_result is the result of the model on the test set.
        The best model is saved in best_model.
        evaluator is used to evaluate the result.Every 10 epochs, the result of the model on the validation set is evaluated.Then compare the result with the best result.If the result is better, the best result is updated and the best model is saved.
        '''
        evaluator = LPEvaluator()
        best_val_result = 0
        best_val_result_auc = 0
        best_test_result = 0
        best_model = copy.deepcopy(self.model)
        best_scores = None
        for epoch in tqdm.tqdm(range(epochs)):
            if (epoch + 1) % 1 == 0:
                output = self.model(data.x.to(self.device)).detach()
                output = F.normalize(output)
                scores = torch.mm(output, output.t())
                test_result = evaluator.eval({
                    'scores': scores,
                    'negative_edge_index': data.test_neg_edge_index.to(self.device),
                    'target_edge_index': data.test_pos_edge_index.to(self.device)
                })
                val_result = evaluator.eval({
                    'scores': scores,
                    'negative_edge_index': data.val_neg_edge_index.to(self.device),
                    'target_edge_index': data.val_pos_edge_index.to(self.device)
                })
                if val_result['auc'] > best_val_result_auc:
                    best_val_result = val_result
                    best_val_result_auc = val_result['auc']
                    best_test_result = test_result
                    best_model = copy.deepcopy(self.model)
                    best_scores = scores
        self.model=best_model
        return best_val_result,best_test_result ,best_scores
    def get_result(self,data,optimizer=None,epochs=None):
        best_val_result,best_test_result ,best_scores=self.train(data,optimizer,epochs)
        return best_val_result,best_test_result,best_scores
    def get_embedding(self,data,optimizer=None,epochs=None):
        best_val_result,best_test_result ,best_scores=self.train(data,optimizer,epochs)
        self.model.eval()
        embeddings=self.model(data.x.to(self.device)).detach()
        return F.normalize(embeddings)
    def get_evasion_result(self,data):
        evaluator = LPEvaluator()
        self.model.eval()
        output = self.model(data.x.to(self.device)).detach()
        output = F.normalize(output)
        scores = torch.mm(output, output.t())
        test_result = evaluator.eval({
            'scores': scores,
            'negative_edge_index': data.test_neg_edge_index.to(self.device),
            'target_edge_index': data.test_pos_edge_index.to(self.device)
        })
        val_result = evaluator.eval({
            'scores': scores,
            'negative_edge_index': data.val_neg_edge_index.to(self.device),
            'target_edge_index': data.val_pos_edge_index.to(self.device)
        })
        return val_result,test_result,scores
    def get_adj(self,data_o):
        G = nx.Graph()
        for i in range(data_o.num_nodes):
            G.add_node(i)
        edge_index = np.array(data_o.train_pos_edge_index.cpu())
        for j in range(edge_index.shape[1]):
            G.add_edge(edge_index[0][j], edge_index[1][j])
        adj_matrix = nx.adjacency_matrix(G).todense()
        adj_matrix=torch.tensor(adj_matrix)
        return adj_matrix
