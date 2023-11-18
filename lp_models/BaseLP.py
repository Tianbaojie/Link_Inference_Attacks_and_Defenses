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
from utils.evaluation import LPEvaluator


#base class for link prediction
class BaseLp(nn.Module):
    '''
    Base class for link prediction, which is used to train the model and get the result.

    The model should be implemented in the subclass.
    The train function is used to train the model.
    The get_result function is used to get the result of the model.
    '''
    def __init__(self):
        super(BaseLp, self).__init__()
    def train(self,data,optimizer,epochs):
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
            self.model.train()
            optimizer.zero_grad()
            output = self.model(data.x.to(self.device), data.train_pos_edge_index.to(self.device))
            output = F.normalize(output)
            scores = torch.mm(output, output.t())
            edge_index_with_self_loops = add_self_loops(data.train_pos_edge_index.to(self.device))[0]
            train_u, train_i, train_j = structured_negative_sampling(edge_index_with_self_loops, data.num_nodes)
            train_u = train_u[:data.train_pos_edge_index.shape[1]]
            train_i = train_i[:data.train_pos_edge_index.shape[1]]
            train_j = train_j[:data.train_pos_edge_index.shape[1]]
            loss = -torch.log(torch.sigmoid(scores[train_u, train_i] - scores[train_u, train_j])).sum()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                output = self.model(data.x.to(self.device), data.train_pos_edge_index.to(self.device)).detach()
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
        embeddings=self.model().detach()
        return F.normalize(embeddings)
    def get_evasion_result(self,data):
        evaluator = LPEvaluator()
        self.model.eval()
        output = self.model(data.x.to(self.device), data.train_pos_edge_index.to(self.device)).detach()
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