import torch
import torch.nn as nn
from ToPo import ToPo_Agg,SemanticAttention
from Graph import Graph_Agg2


class Model(nn.Module):
    def __init__(self,num_meta,infeat,hidfeat,numclass,drop,topo_num):  #topo_num:[7167,60]
        super(Model, self).__init__()
        self.feat_trans = nn.Linear(infeat,hidfeat)
        #self.feat_trans2 = nn.Linear(topo_num,hidfeat)
        self.topo = ToPo_Agg(hidfeat,topo_num)
        self.graph_agg = Graph_Agg2(num_meta,hidfeat,drop)
        self.linear = nn.Linear(hidfeat,numclass)
        self.no_linear = nn.Tanh()
        self.semantic =SemanticAttention(hidfeat)

    def forward(self,features, adjM,feat_similar_neighbors,ADJ,feature_attr): #feature_attr 是两个one-hot矩阵7167*7167  adjM是对应的两个邻接矩阵4017*7167
        #features_1 = self.no_linear(self.feat_trans2(features_1))

        # feat = self.no_linear(self.feat_trans(features))
        feat = self.feat_trans(features)
        feat_topo = self.topo(feat,adjM,feat_similar_neighbors,feature_attr)
        feat_meta = self.graph_agg(ADJ,feat)
        feat = self.semantic(torch.stack([feat_meta,feat_topo],dim=1))
        feat = feat.squeeze(1)
        return self.linear(feat),feat
        # return self.linear(feat_meta),feat_meta



