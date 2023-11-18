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
from sklearn.metrics import roc_auc_score
from deeprobust.graph.defense import Node2Vec as node2vec
from utils.evaluation import LPEvaluator
import sys
from torch_geometric.utils import contains_isolated_nodes, remove_isolated_nodes
from torch_geometric.loader import LinkNeighborLoader
def calculate_mrr(target, target_rank):
    mrr = 0.0
    for i in range(len(target)):
        if target[i] in target_rank[i]:
            rank = target_rank[i].index(target[i]) + 1
            mrr += 1.0 / rank
    mrr /= len(target)
    return mrr
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

        train_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
            edge_label_index=torch.concat([data.train_pos_edge_index,data.train_neg_edge_index],axis=1),
            edge_label=torch.concat([torch.ones(data.train_pos_edge_index.shape[1]),torch.zeros(data.train_neg_edge_index.shape[1])],axis=0),
            batch_size=512,
            shuffle=False,
            num_workers=16
        )

        val_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
            edge_label_index=torch.concat([data.val_pos_edge_index,data.val_neg_edge_index],axis=1),
            edge_label=torch.concat([torch.ones(data.val_pos_edge_index.shape[1]),torch.zeros(data.val_neg_edge_index.shape[1])],axis=0),
            batch_size=512,
            shuffle=False,
            num_workers=16
        )

        test_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
            edge_label_index=torch.concat([data.test_pos_edge_index,data.test_neg_edge_index],axis=1),
            edge_label=torch.concat([torch.ones(data.test_pos_edge_index.shape[1]),torch.zeros(data.test_neg_edge_index.shape[1])],axis=0),
            batch_size=512,
            shuffle=False,
            num_workers=16
        )
        best_val_result = 0
        best_val_result_auc = 0
        best_test_result = 0
        best_model = copy.deepcopy(self.model)
        best_scores = None
        for epoch in tqdm.tqdm(range(epochs)):
            self.model.train()
            for batch  in tqdm.tqdm(train_loader):
                optimizer.zero_grad()

                h = self.model(batch.x.to(self.device),batch.edge_index.to(self.device))
                h_src = h[batch.edge_label_index[0].to(self.device)]
                h_dst = h[batch.edge_label_index[1].to(self.device)]
                pred = (h_src * h_dst).sum(dim=-1)
                loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label.to(self.device))
                loss.backward()

                optimizer.step()
            if (epoch + 1) % 1 == 0:

                test_result = self.test(test_loader)
                val_result = self.test(val_loader)
                print('val_result',val_result)
                print('test_result',test_result)
                if val_result['auc'] > best_val_result_auc:
                    best_val_result = val_result
                    best_val_result_auc = val_result['auc']
                    best_test_result = test_result
                    best_model = copy.deepcopy(self.model)
                    best_scores = None
        self.model=best_model
        return best_val_result,best_test_result ,best_scores
    def test(self,data_loader):
        self.model.eval()
        preds,labels=[],[]
        with torch.no_grad():
            for batch  in tqdm.tqdm(data_loader):
                h = self.model(batch.x.to(self.device),batch.edge_index.to(self.device))
                h_src = h[batch.edge_label_index[0].to(self.device)]
                h_dst = h[batch.edge_label_index[1].to(self.device)]
                pred = (h_src * h_dst).sum(dim=-1)
                pred=F.sigmoid(pred)
                label=batch.edge_label
                preds.append(pred)
                labels.append(label)
        return self.eval(torch.concat(preds).detach().cpu(),torch.concat(labels).detach().cpu())
    def eval(self,ranking_scores,ranking_labels):
        auc = roc_auc_score(ranking_labels.numpy(), ranking_scores.numpy())
        acc = ((ranking_scores > 0.5) == ranking_labels).to(torch.float32).mean()
        recall = ((ranking_scores > 0.5) * ranking_labels).to(torch.float32).sum() / ranking_labels.sum()
        precision = ((ranking_scores > 0.5) * ranking_labels).to(torch.float32).sum() / (ranking_scores > 0.5).to(torch.float32).sum()
        f1 = 2 * precision * recall / (precision + recall)
        ranking_scores=torch.concat([ranking_scores[:2000].view(-1,1),ranking_scores[2000:].view(-1,4)],axis=1)
        mrr=calculate_mrr([0]*len(ranking_scores),ranking_scores.argsort(dim=1,descending=True).tolist())
        return {'auc': auc, 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1,'mrr':mrr}
    def get_result(self,data,optimizer=None,epochs=None):
        best_val_result,best_test_result ,best_scores=self.train(data,optimizer,epochs)
        return best_val_result,best_test_result,best_scores
    def get_embedding(self,data,optimizer=None,epochs=None):
        best_val_result,best_test_result ,best_scores=self.train(data,optimizer,epochs)
        self.model.eval()
        embeddings=self.model().detach()
        return F.normalize(embeddings)
    def get_evasion_result(self,data):
        val_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
            edge_label_index=torch.concat([data.val_pos_edge_index,data.val_neg_edge_index],axis=0),
            edge_label=torch.concat([torch.ones(data.val_pos_edge_index.shape[1]),torch.zeros(data.val_neg_edge_index.shape[1])],axis=0),
            batch_size=512,
            shuffle=False,
            num_workers=16
        )

        test_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
             edge_label_index=torch.concat([data.test_pos_edge_index,data.test_neg_edge_index],axis=0),
            edge_label=torch.concat([torch.ones(data.test_pos_edge_index.shape[1]),torch.zeros(data.test_neg_edge_index.shape[1])],axis=0),
            batch_size=516,
            shuffle=False,
            num_workers=16
        )
        test_result = self.test(test_loader)
        val_result = self.test(val_loader)
        return val_result,test_result,None
