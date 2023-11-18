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

from lp_models_for_large_dataset.BaseLP import BaseLp
from utils.evaluation import LPEvaluator


def generate_walks(graph, num_walks, walk_length):
    '''
    generate random walks,evety node is the start node, and the length of each walk is walk_length.
    For isolated nodes, the walk is terminated,while for nodes with no neighbors, the walk is randomly chosen from the neighbors of the previous node.
    :param graph: networkx graph
    :param num_walks: number of walks per node
    :param walk_length: length of each walk
    :return: list of walks
    '''
    walks = []
    nodes = list(graph.nodes())
    for _ in tqdm.tqdm(range(num_walks)):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(walk[-1]))
                if not neighbors:
                    break
                walk.append(np.random.choice(neighbors))
            walks.append(walk)
    return walks
def deepwalk(graph, num_walks=10, walk_length=10, embedding_size=64, window_size=5, workers=4):
    '''
    deepwalk is a node embedding algorithm, which is based on random walk, and the embedding of each node is the average of the embedding of the nodes in the random walk.
    :param graph: networkx graph
    :param num_walks: number of walks per node
    :param walk_length: length of each walk
    :param embedding_size: dimension of embedding
    :param window_size: window size of skip-gram
    :param workers: number of workers
    :return: node embeddings
    '''
    walks = generate_walks(graph, num_walks, walk_length)
    walks = [[str(node) for node in walk] for walk in walks]  # 将 Tensor 转换成 str
    model = Word2Vec(walks, size=embedding_size, window=window_size, min_count=0, sg=1, workers=workers)
    node_embeddings = np.zeros((graph.number_of_nodes(), embedding_size))
    for i in range(graph.number_of_nodes()):
        node_embeddings[i] = model.wv[str(i)]  # 用 str(i) 作为 key 来获取对应节点的向量
    return node_embeddings

class DeepWalk(torch.nn.Module):
    def __init__(self,graph,embedding_dim,device) -> None:
        super(DeepWalk, self).__init__()
        self.embedding=torch.tensor(deepwalk(graph,embedding_size=embedding_dim),dtype=torch.float32,requires_grad=False).to(device)
        self.fc=nn.Linear(embedding_dim,embedding_dim)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self,x=None,edge_index=None):
        return self.fc(self.embedding)

#基于DeepWalk的链接预测模型
class DeepWalk_LP(BaseLp):
    def __init__(self,data,embedding_dim,device):
        super(DeepWalk_LP, self).__init__()
        self.device=device
        self.graph = self.bulid_graph(data)
        self.embedding_dim = embedding_dim
        self.model=DeepWalk(self.graph,self.embedding_dim,device).to(self.device)
    def bulid_graph(self,data):
        G = nx.Graph()
        # 添加所有节点
        for i in range(data.num_nodes):
            G.add_node(i)
        edge_index =  np.array(data.train_pos_edge_index.cpu())
        for j in range(edge_index.shape[1]):
            G.add_edge(edge_index[0][j], edge_index[1][j])
        return G
