import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dgl.nn.pytorch import GATConv,GraphConv
import torch.nn.functional as F
import math
from torch.nn import Parameter


class GraphConvolution(nn.Module):  # 自己定义的GCN
    def __init__(self, in_features, out_features, bias=True):#7167,64
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  #这里的权重和偏置归一化
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj): #7167*7167   4019*7167
        support = torch.spmm(inputs, self.weight)  # HW in GCN
        output = torch.spmm(adj, support)  #AHW
        if self.bias is not None:
            return F.tanh(output + self.bias)  #这里激活函数按照定义为elu
        else:
            return F.tanh(output)



class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):  #64,128
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1, bias=False))# 64,128  -->  128,1
    def forward(self, z):
        #z.shape（4019,2,64）
        w = self.project(z).mean(0)     #w.shape(2,1)  经过project变成（4019,2,1） 在经过mean变成（2,1）
        beta = torch.softmax(w, dim=0)
        print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1)

class ToPo_Agg(nn.Module):
    def __init__(self,hid_dim,topo_num,num_heads=1,dropout=0.5):     #topo_num:[7167,60]
        super(ToPo_Agg, self).__init__()
        self.len_top_num = len(topo_num)
        self.gcn = [GraphConvolution(topo_num[i],hid_dim,bias=True) for i in range(self.len_top_num)]
        self.gat2 = GATConv(hid_dim, hid_dim, num_heads,dropout, dropout, activation=F.tanh)
        self.semantic_agg = SemanticAttention(hid_dim)
        self.linear1 = nn.Linear(self.len_top_num * hid_dim,hid_dim)
        self.linear2 = nn.Linear(hid_dim,hid_dim)
        self.non_linear = nn.Tanh()

    def forward(self,feature,adjM,similary,feature_attr):
        feat_topo = [self.gcn[i](feature_attr[i],adjM[i]) for i in range(self.len_top_num)]    #feature_attr 是两个one-hot矩阵7167*7167  adjM是对应的两个邻接矩阵4017*7167
        l = torch.cat(feat_topo,dim=1)
        l = self.non_linear(self.linear1(l))
        feat_feat = self.gat2(similary,feature).flatten(1)
        feat = self.semantic_agg(torch.stack([l,feat_feat],dim=1))
        return feat


