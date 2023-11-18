import copy

import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch_geometric.nn import GATConv, GCNConv,GCN2Conv,GENConv
from torch_geometric.utils import add_self_loops, structured_negative_sampling
from torch_geometric.loader import LinkNeighborLoader
from lp_models.BaseLP import BaseLp
from utils.evaluation import LPEvaluator
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
class GEN(torch.nn.Module):
    def __init__(self, embedding_dim_in,embedding_dim_out, num_layers, dropout=0.0):
        super().__init__()

        self.fc = nn.Linear(embedding_dim_in, embedding_dim_out)
    

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GENConv(embedding_dim_out,embedding_dim_out))
        #dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, edge_index):
        x=self.fc(x)
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index)
            x = F.relu(x)
        x=self.dropout(x)
            


        return x

class GEN_LP(BaseLp):
    '''
    GEN_LP model used for link prediction based on GEN, and the output of the last layer is normalized.
    The model trains the GCN model and then uses the output of the last layer as the node representation.
    The loss function is the negative log likelihood loss.

    Parameters:
    embedding_dim: int
        The number of embedding dimension.
    device: torch.device
        The device to run the model.
    '''
    def __init__(self,embedding_dim_in,embedding_dim_out,device, num_layers, dropout=0.0):
        super(GEN_LP,self).__init__()
        self.device=device
        self.model=GEN(embedding_dim_in,embedding_dim_out, num_layers, dropout).to(device)
    def train(self,data,optimizer,epochs):
        train_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
            edge_label_index=torch.concat([data.train_pos_edge_index,data.train_neg_edge_index],axis=1),
            edge_label=torch.concat([torch.ones(data.train_pos_edge_index.shape[1]),torch.zeros(data.train_neg_edge_index.shape[1])],axis=0),
            batch_size=128,
            shuffle=False,
            num_workers=16
        )

        val_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
            edge_label_index=torch.concat([data.val_pos_edge_index,data.val_neg_edge_index],axis=1),
            edge_label=torch.concat([torch.ones(data.val_pos_edge_index.shape[1]),torch.zeros(data.val_neg_edge_index.shape[1])],axis=0),
            batch_size=128,
            shuffle=False,
            num_workers=16
        )

        test_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
            edge_label_index=torch.concat([data.test_pos_edge_index,data.test_neg_edge_index],axis=1),
            edge_label=torch.concat([torch.ones(data.test_pos_edge_index.shape[1]),torch.zeros(data.test_neg_edge_index.shape[1])],axis=0),
            batch_size=128,
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

                h = self.model(x=batch.x.to(self.device),edge_index=batch.edge_index.to(self.device))
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
    def get_evasion_result(self,data):
        val_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
            edge_label_index=torch.concat([data.val_pos_edge_index,data.val_neg_edge_index],axis=0),
            edge_label=torch.concat([torch.ones(data.val_pos_edge_index.shape[1]),torch.zeros(data.val_neg_edge_index.shape[1])],axis=0),
            batch_size=128,
            shuffle=False,
            num_workers=16
        )

        test_loader = LinkNeighborLoader(
            data=data,
            num_neighbors=[-1, -1],
             edge_label_index=torch.concat([data.test_pos_edge_index,data.test_neg_edge_index],axis=0),
            edge_label=torch.concat([torch.ones(data.test_pos_edge_index.shape[1]),torch.zeros(data.test_neg_edge_index.shape[1])],axis=0),
            batch_size=128,
            shuffle=False,
            num_workers=16
        )
        test_result = self.test(test_loader)
        val_result = self.test(val_loader)
        return val_result,test_result,None
    def eval(self,ranking_scores,ranking_labels):
        auc = roc_auc_score(ranking_labels.numpy(), ranking_scores.numpy())
        acc = ((ranking_scores > 0.5) == ranking_labels).to(torch.float32).mean()
        recall = ((ranking_scores > 0.5) * ranking_labels).to(torch.float32).sum() / ranking_labels.sum()
        precision = ((ranking_scores > 0.5) * ranking_labels).to(torch.float32).sum() / (ranking_scores > 0.5).to(torch.float32).sum()
        f1 = 2 * precision * recall / (precision + recall)
        ranking_scores=torch.concat([ranking_scores[:2000].view(-1,1),ranking_scores[2000:].view(-1,4)],axis=1)
        mrr=calculate_mrr([0]*len(ranking_scores),ranking_scores.argsort(dim=1,descending=True).tolist())
        return {'auc': auc, 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1,'mrr':mrr}

    
    
