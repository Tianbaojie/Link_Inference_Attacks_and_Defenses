import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch import nn
from torch_geometric.nn import APPNP, GAE

from lp_models_for_large_dataset.BaseLP import BaseLp
from torch_geometric.loader import LinkNeighborLoader
def calculate_mrr(target, target_rank):
    mrr = 0.0
    for i in range(len(target)):
        if target[i] in target_rank[i]:
            rank = target_rank[i].index(target[i]) + 1
            mrr += 1.0 / rank
    mrr /= len(target)
    return mrr
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index,scaling_factor=1.8):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)
        self.scaling_factor=scaling_factor

    def forward(self, x, edge_index, not_prop=0):
        x_ = self.linear1(x)
        x_ = self.propagate(x_, edge_index)

        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1) * self.scaling_factor
        x = self.propagate(x, edge_index)
        return x, x_
class VGNAE(GAE):
    def __init__(self, encoder, decoder = None):
        super().__init__(encoder, decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=10)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None,
                logstd= None) :
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    def test(self, z, pos_edge_index,neg_edge_index):
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        acc= ((pred > 0.5) == y).mean()
        recall = ((pred > 0.5) * y).sum() / y.sum()
        precision = ((pred > 0.5) * y).sum() / (pred > 0.5).sum()
        f1=2*recall*precision/(recall+precision)
        auc=roc_auc_score(y, pred)
        ranking_scores=torch.concat([ranking_scores[:2000].view(-1,1),ranking_scores[2000:].view(-1,4)],axis=1)
        mrr=calculate_mrr([0]*len(ranking_scores),ranking_scores.argsort(dim=1,descending=True).tolist())
        print(ranking_scores)
        return {'auc': auc, 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1,'mrr':mrr},[pos_pred, neg_pred]


class VGNAE_LP(BaseLp):
    #初始化
    def __init__(self, data, embedding_dim, device):
        super().__init__()
        self.model = VGNAE(Encoder(data.x.size()[1], embedding_dim, data.train_pos_edge_index)).to(device)
        self.data = data
        self.device = device
    def train(self,data,optimizer,epochs):
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
                z = self.model.encode(batch.x.to(self.device), batch.edge_index.to(self.device))
                loss = self.model.recon_loss(z, batch.edge_index.to(self.device))
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 1 == 0:
                self.model.eval()
                with torch.no_grad():
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
                z = self.model.encode(batch.x.to(self.device), batch.edge_index.to(self.device))
                pred=self.model.decoder(z, batch.edge_label_index.to(self.device), sigmoid=True)
                label=batch.edge_label
                preds.append(pred)
                labels.append(label)

        return self.eval(torch.concat(preds).detach().cpu(),torch.concat(labels).detach().cpu())
    
    def get_embedding(self,data,optimizer=None,epochs=None):
        best_val_result,best_test_result ,best_scores=self.train(data,optimizer,epochs)
        self.model.eval()
        embeddings=self.model.encode(self.data.x, self.data.train_pos_edge_index).detach()
        return F.normalize(embeddings)