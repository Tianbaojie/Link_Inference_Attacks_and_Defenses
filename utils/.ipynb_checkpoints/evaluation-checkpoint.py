from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch_geometric.utils import add_self_loops, structured_negative_sampling

from pGRACE.model import LogReg
class LPEvaluator:
    '''
    Link prediction evaluator
    '''
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(scores, negative_edge_index, target_edge_index):
        edge_index = torch.cat([negative_edge_index, target_edge_index], -1)
        ranking_scores = scores[edge_index[0], edge_index[1]]
        ranking_labels = torch.cat([torch.zeros(negative_edge_index.shape[1]), torch.ones(target_edge_index.shape[1])]).to(scores.device)
        auc = roc_auc_score(ranking_labels.detach().cpu().numpy(), ranking_scores.detach().cpu().numpy())
        acc = ((ranking_scores > 0.5) == ranking_labels).to(torch.float32).mean()
        recall = ((ranking_scores > 0.5) *ranking_labels).to(torch.float32).sum() / ranking_labels.sum()
        precision = ((ranking_scores > 0.5) * ranking_labels).to(torch.float32).sum() / (ranking_scores > 0.5).to(torch.float32).sum()
        f1 = 2 * precision * recall / (precision + recall)
        return {'auc': auc, 'acc': acc, 'recall': recall, 'precision': precision, 'f1': f1} 

    def eval(self, res):
        return self._eval(**res)