import argparse

import scipy.sparse
import torch
from tools import evaluate_results_nc, EarlyStopping
from load_data import load_acm, load_imdb, load_Yelp,load_dblp
import numpy as np
import random
from model import Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")
import dgl


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def process_similar(similar,select_num):
    similarity_matrix = cosine_similarity(similar)

    # 初始化邻接矩阵
    adjacency_matrix = np.zeros_like(similarity_matrix)

    # 对于每个节点，找出其最相似的top_t_neighbors个节点
    for i in range(len(similar)):
        # 获取节点i与其他所有节点的相似度
        node_similarities = similarity_matrix[i]

        # 找出除了自己之外最相似的top_t_neighbors个节点的索引
        nearest_neighbors = node_similarities.argsort()[::-1][1:select_num + 1]

        # 将这些邻居添加到邻接矩阵中
        adjacency_matrix[i, nearest_neighbors] = 1
        adjacency_matrix[nearest_neighbors, i] = 1  # 如果是无向图，则需要双向连接
    adjacency_matrix = scipy.sparse.coo_array(adjacency_matrix)
    adjacency_matrix = dgl.DGLGraph(adjacency_matrix)
    return adjacency_matrix

def topo_similar(adj,target_node_number,select_num):

    P = target_node_number
    adj_matrix = (adj.toarray())
    # 确保矩阵是对称的，表示无向图
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
    # 提取P节点的子矩阵
    P_matrix = adj_matrix[:P, :]

    # 计算余弦相似度
    similarity_matrix = cosine_similarity(P_matrix)

    # 找出每个P节点最相似的5个邻居节点
    most_similar_neighbors = []

    for i in range(P):
        # 获取节点i的相似度向量，并忽略自身
        similarities = similarity_matrix[i]
        similar_indices = np.argsort(similarities)[::-1]  # 从大到小排序
        similar_indices = similar_indices[similar_indices != i]  # 去掉自身
        most_similar_neighbors.append(similar_indices[:select_num])  # 取前5个

    # 将结果转换为NumPy数组
    most_similar_neighbors_array = np.array(most_similar_neighbors)
    most_similar_neighbors_array = scipy.sparse.coo_array(most_similar_neighbors_array)

    neighbors_tensor1 = dgl.DGLGraph(most_similar_neighbors_array)
    neighbors_tensor = dgl.add_self_loop(neighbors_tensor1)
    return neighbors_tensor


def main(args):
    ADJ,adjM, similarity_matrix, features,feature_attr, labels, num_classes, train_idx, val_idx, test_idx,topo_num = load_acm() #topo_num:[7167,60]
    target_node_number = features.shape[0]
    in_dims = features.shape[1]
    num_meta = len(ADJ)
    features = features.to(args['device'])
    labels = labels.to(args['device'])
    print("计算特征相似性：...")
    feat_similar_neighbors =process_similar(features,args['select_similar_num'])

    model = Model(num_meta,in_dims, args["hidden_units"], num_classes,args['drop'],topo_num)
    early_stopping = EarlyStopping(patience=args['patience'], verbose=True,
                                   save_path='checkpoint/checkpoint_{}.pt'.format('ACM'))
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits, h = model(features, adjM,feat_similar_neighbors,ADJ,feature_attr)
        loss = loss_fcn(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        logits, h = model(features, adjM,feat_similar_neighbors,ADJ,feature_attr)
        val_loss = loss_fcn(logits[val_idx], labels[val_idx])
        test_loss = loss_fcn(logits[test_idx], labels[test_idx])
        print('Epoch{:d}| Train Loss{:.4f}| Val Loss{:.4f}| Test Loss{:.4f}'.format(epoch + 1, loss.item(),
                                                                                    val_loss.item(), test_loss.item()))
        early_stopping(val_loss.data.item(), model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format('ACM')))
    model.eval()
    logits, h = model(features, adjM,feat_similar_neighbors,ADJ,feature_attr)
    evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(), int(labels.max()) + 1)
    # Y = labels[test_idx].numpy()
    # ml = TSNE(n_components=2)
    # node_pos = ml.fit_transform(h[test_idx].detach().cpu().numpy())
    # color_idx = {}
    # for i in range(len(h[test_idx].detach().cpu().numpy())):
    #     color_idx.setdefault(Y[i], [])
    #     color_idx[Y[i]].append(i)
    # for c, idx in color_idx.items():  # c是类型数，idx是索引
    #     if str(c) == '1':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1 )
    #     elif str(c) == '2':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1 )
    #     elif str(c) == '0':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1 )
    #     elif str(c) == '3':
    #         plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#006400', s=15, alpha=1)
    # plt.legend()
    # plt.savefig( str(args['dataset']) + "分类图.png", dpi=1000,
    #             bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=0.005, help='学习率')
    parser.add_argument('--dataset', default="imdb", help='学习率')
    parser.add_argument('--weight_decay', default=0.0006, help='权重衰减')
    parser.add_argument('--hidden_units', default=64, help='隐藏层数')
    parser.add_argument('--drop', default=0.5, help='特征丢弃率')
    parser.add_argument('--select_topo_num', default=10, help='拓扑中选择的邻居数量')
    parser.add_argument('--select_similar_num', default=20, help='特征中选择的邻居数量')
    parser.add_argument('--alpha', default=0.5, help='alpha')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--patience', type=int, default=10, help='耐心值')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--sample_rate', default=[5, 5], help='属性节点数量')
    parser.add_argument('--device', type=str, default='cpu', help='使用cuda:0或者cpu')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)
