import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp



def load_acm(prefix=r"G:\OtherCode\MyProject\0005大小路径鉴别-测试-终版\大小路径鉴别算法-GAT测试\ACM_processed"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_p.npz').toarray()
    features = torch.FloatTensor(features_0)

    similarity_matrix = np.load(prefix+"/similarity_matrix.npy")

    # adjM = sp.coo_matrix(adjM[:4019,4019:])
    # adjM = dgl.DGLGraph(adjM)
    # adjM = dgl.add_self_loop(adjM)
    adjM = sp.load_npz(prefix + "/adjM.npz").toarray()
    adj_PA = torch.from_numpy(adjM[:4019,4019:4019+7167]).type(torch.FloatTensor)
    adj_PS = torch.from_numpy(adjM[:4019,4019+7167:4019+7167+60]).type(torch.FloatTensor)
    # adjM =  F.normalize(torch.from_numpy(adj_PA).type(torch.FloatTensor), dim=1, p=2)
    # adjM =  F.normalize(torch.from_numpy(adj_PS).type(torch.FloatTensor), dim=1, p=2)
    features_1 = torch.FloatTensor(np.eye(7167 ))
    features_2 = torch.FloatTensor(np.eye(60))
    feature_attr = [features_1,features_2]
    adj = [adj_PA,adj_PS]

    # 标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    PAP = scipy.sparse.load_npz(prefix + '/PAP.npz').toarray()
    norm_PAP = F.normalize(torch.from_numpy(PAP).type(torch.FloatTensor), dim=1, p=2)
    PSP = scipy.sparse.load_npz(prefix + '/PSP.npz').toarray()
    norm_PSP = F.normalize(torch.from_numpy(PSP).type(torch.FloatTensor), dim=1, p=2)
    # PSPSP = scipy.sparse.load_npz(prefix + '/pspsp.npz').toarray()
    # norm_PSPSP = F.normalize(torch.from_numpy(PSPSP).type(torch.FloatTensor), dim=1, p=2)
    # PAPAP = scipy.sparse.load_npz(prefix + '/papap.npz').toarray()
    # norm_PAPAP = F.normalize(torch.from_numpy(PAPAP).type(torch.FloatTensor), dim=1, p=2)
    # PSPAP = scipy.sparse.load_npz(prefix + '/pspap.npz').toarray()
    # norm_PSPAP = F.normalize(torch.from_numpy(PSPAP).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_PAP,norm_PSP]


    return ADJ,adj, similarity_matrix, features,feature_attr, labels, num_classes, train_idx, val_idx, test_idx,[7167 , 60]



def load_imdb(prefix=r"G:\OtherCode\MyProject\0005大小路径鉴别-测试-终版\大小路径鉴别算法-GAT测试\IMDB_processed"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_M.npz').toarray()
    features = torch.FloatTensor(features_0)
    similarity_matrix = np.load(prefix + "/similarity_matrix.npy")
    similarity_matrix = torch.from_numpy(similarity_matrix).type(torch.FloatTensor)
    # 标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    adjM = sp.load_npz(prefix + "/adjM.npz")

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    MAM = sp.load_npz(prefix + '/MAM.npz').toarray()
    norm_MAM = F.normalize(torch.from_numpy(MAM).type(torch.FloatTensor), dim=1, p=2)
    MDM = scipy.sparse.load_npz(prefix + '/MDM.npz').toarray()
    norm_MDM = F.normalize(torch.from_numpy(MDM).type(torch.FloatTensor), dim=1, p=2)
    MAMAM = sp.load_npz(prefix + '/MAMAM.npz').toarray()
    norm_MAMAM = F.normalize(torch.from_numpy(MAMAM).type(torch.FloatTensor), dim=1, p=2)
    # MAMDM = sp.load_npz(prefix + '/MAMDM.npz').toarray()
    # norm_MAMDM = F.normalize(torch.from_numpy(MAMDM).type(torch.FloatTensor), dim=1, p=2)
    MDMDM = sp.load_npz(prefix + '/MDMDM.npz').toarray()
    norm_MDMDM = F.normalize(torch.from_numpy(MDMDM).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_MAM,norm_MDM]

    adjM = sp.load_npz(prefix + "/adjM.npz").toarray()
    adjM = adjM[:4278, 4278:]
    adjM = F.normalize(torch.from_numpy(adjM).type(torch.FloatTensor), dim=1, p=2)
    features_1 = torch.FloatTensor(np.eye(2081 + 5257))
    return ADJ,adjM, similarity_matrix, features,features_1, labels, num_classes, train_idx, val_idx, test_idx,2081 + 5257


def load_Yelp(prefix=r"G:\OtherCode\MyProject\0005大小路径鉴别-测试-终版\大小路径鉴别算法-GAT测试\4_Yelp"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_b.npz').toarray()  # B:2614
    # features_1 = scipy.sparse.load_npz(prefix + '/features_1_u.npz').toarray()  # U:1286
    # features_2 = scipy.sparse.load_npz(prefix + '/features_2_s.npz').toarray()  # S:4
    # features_3 = scipy.sparse.load_npz(prefix + '/features_3_l.npz').toarray()  # L:9
    features = torch.FloatTensor(features_0)
    adjM = sp.load_npz(prefix + "/adjM.npz")


    similarity_matrix = np.load(prefix + "/similarity_matrix.npy")
    similarity_matrix = torch.from_numpy(similarity_matrix).type(torch.FloatTensor)

    # 标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npy', allow_pickle=True)
    train_idx = train_val_test_idx.item()['train_idx'].astype(int)
    val_idx = train_val_test_idx.item()['val_idx']
    test_idx = train_val_test_idx.item()['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3


    BUB = sp.load_npz(prefix + '/adj_bub_one.npz').toarray()
    norm_BUB = F.normalize(torch.from_numpy(BUB).type(torch.FloatTensor), dim=1, p=2)
    BSB = scipy.sparse.load_npz(prefix + '/adj_bsb_one.npz').toarray()
    norm_BSB = F.normalize(torch.from_numpy(BSB).type(torch.FloatTensor), dim=1, p=2)
    BLB = scipy.sparse.load_npz(prefix + '/adj_blb_one.npz').toarray()
    norm_BLB = F.normalize(torch.from_numpy(BLB).type(torch.FloatTensor), dim=1, p=2)
    BLBLB = scipy.sparse.load_npz(prefix + '/blblb.npz').toarray()
    norm_BLBLB = F.normalize(torch.from_numpy(BLBLB).type(torch.FloatTensor), dim=1, p=2)
    # BSBLB = scipy.sparse.load_npz(prefix + '/bsblb.npz').toarray()
    # norm_BSBLB = F.normalize(torch.from_numpy(BSBLB).type(torch.FloatTensor), dim=1, p=2)
    BSBSB = scipy.sparse.load_npz(prefix + '/bsbsb.npz').toarray()
    norm_BSBSB = F.normalize(torch.from_numpy(BSBSB).type(torch.FloatTensor), dim=1, p=2)
    # BSBUB = scipy.sparse.load_npz(prefix + '/bsbub.npz').toarray()
    # norm_BSBUB = F.normalize(torch.from_numpy(BSBUB).type(torch.FloatTensor), dim=1, p=2)
    BUBUB = scipy.sparse.load_npz(prefix + '/bubub.npz').toarray()
    norm_BUBUB = F.normalize(torch.from_numpy(BUBUB).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_BUB,norm_BUBUB,norm_BSBSB,norm_BLBLB]

    adjM = sp.load_npz(prefix + "/adjM.npz").toarray()
    adjM = adjM[:2614, 2614:]
    adjM = F.normalize(torch.from_numpy(adjM).type(torch.FloatTensor), dim=1, p=2)
    features_1 = torch.FloatTensor(np.eye(1286+4+9))
    return ADJ, adjM, similarity_matrix, features, features_1, labels, num_classes, train_idx, val_idx, test_idx,1286+4+9


def load_dblp(prefix=r"G:\OtherCode\MyProject\0005大小路径鉴别-测试-终版\大小路径鉴别算法-GAT测试\DBLP_processed"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_A.npz').toarray()
    features = torch.FloatTensor(features_0)
    similarity_matrix = np.load(prefix + "/similarity_matrix.npy")
    similarity_matrix = torch.from_numpy(similarity_matrix).type(torch.FloatTensor)
    # 标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    adjM = sp.load_npz(prefix + "/adjM.npz")

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 4

    APA = sp.load_npz(prefix + '/apa.npz').toarray()
    norm_APA = F.normalize(torch.from_numpy(APA).type(torch.FloatTensor), dim=1, p=2)
    apcpa = scipy.sparse.load_npz(prefix + '/apcpa.npz').toarray()
    norm_apcpa = F.normalize(torch.from_numpy(apcpa).type(torch.FloatTensor), dim=1, p=2)
    aptpa = sp.load_npz(prefix + '/aptpa.npz').toarray()
    norm_aptpa = F.normalize(torch.from_numpy(aptpa).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_APA,norm_apcpa,norm_aptpa]

    adjM = sp.load_npz(prefix + "/adjM.npz").toarray()
    adjM = adjM[:4057, 4057:]
    adjM = F.normalize(torch.from_numpy(adjM).type(torch.FloatTensor), dim=1, p=2)
    features_1 = torch.FloatTensor(np.eye(14328+7723+20))
    return ADJ, adjM, similarity_matrix, features, features_1, labels, num_classes, train_idx, val_idx, test_idx,14328+7723+20


if __name__ == "__main__":
    load_acm()
