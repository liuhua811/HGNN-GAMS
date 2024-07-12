import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv,GraphConv
import dgl
from ToPo import SemanticAttention

class Graph_Agg(nn.Module):
    def __init__(self,num_matr,hid_dim,dropout,num_heads=1):
        super(Graph_Agg, self).__init__()
        self.dropout = dropout
        self.attention_weights = nn.Parameter(torch.randn(num_matr))
        self.softmax = nn.Softmax(dim=0)
        self.gat = GATConv(hid_dim, hid_dim, num_heads,dropout, dropout, activation=F.tanh)
        self.linear = nn.Linear(hid_dim,hid_dim)
        self.non_linear = nn.Tanh()

    def forward(self, adj_list,feat):
        # Normalize attention weights using softmax
        feat = self.non_linear(self.linear(feat))
        attention_weights = self.softmax(self.attention_weights)

        # Merge adjacency matrices using attention weights
        merged_adj = sum([adj * weight for adj, weight in zip(adj_list, attention_weights)])
        src, dst = merged_adj.nonzero(as_tuple=True)
        paths = dgl.graph((src,dst))
        paths = dgl.add_self_loop(paths)
        feat = self.gat(paths,feat).flatten(1)

        return feat


class Graph_Agg2(nn.Module):
    def __init__(self,num_matr,hid_dim,dropout,num_heads=1):
        super(Graph_Agg2, self).__init__()
        self.num_matr = num_matr
        self.dropout = dropout
        self.attention_weights = nn.Parameter(torch.randn(num_matr))
        self.softmax = nn.Softmax(dim=0)
        self.gat = GATConv(hid_dim, hid_dim, num_heads,dropout, dropout, activation=F.tanh)
        self.linear = nn.Linear(hid_dim,hid_dim)
        self.non_linear = nn.Tanh()
        self.gat_meat = [GATConv(hid_dim, hid_dim, num_heads,dropout, dropout, activation=F.tanh) for _ in range(num_matr)]
        self.semantic = SemanticAttention(hid_dim)
        self.feat_trans = nn.Linear(hid_dim*2,hid_dim)
    def forward(self, adj_list,feat):
        # Normalize attention weights using softmax
        #feat = self.non_linear(self.linear(feat))
        attention_weights = self.softmax(self.attention_weights)

        # Merge adjacency matrices using attention weights
        merged_adj = sum([adj * weight for adj, weight in zip(adj_list, attention_weights)])
        src, dst = merged_adj.nonzero(as_tuple=True)
        paths = dgl.graph((src,dst))
        paths = dgl.add_self_loop(paths)
        feat_graph = self.gat(paths,feat).flatten(1)     #是融合元路径图聚合的特征

        meta_agg = []
        meta_agg.append(feat_graph)
        for i in range(self.num_matr):
            src, dst = adj_list[i].nonzero(as_tuple=True)
            paths = dgl.graph((src, dst))
            paths = dgl.add_self_loop(paths)
            meta_agg.append(self.gat_meat[i](paths,feat).flatten(1))
        semantic = self.semantic(torch.stack(meta_agg,dim=1))
        feat_all = torch.cat((feat_graph,semantic),dim=1)
        feat_all = self.non_linear(self.feat_trans(feat_all))


        return feat_all
        # return feat_graph