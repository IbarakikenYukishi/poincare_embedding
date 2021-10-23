import os
import sys
import torch
import random
import numpy as np
from torch import nn
from torch import optim
from tqdm import trange, tqdm
from collections import Counter
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from datasets import hyperbolic_geometric_graph
from copy import deepcopy

np.random.seed(0)

def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))


# Dataset
# DataLoader
# Optimizer(RSGD)   

def create_dataset(
    adj_mat,
    n_max_positives=2,
    n_max_negatives=10,
    val_size=0.1
):
    _adj_mat=deepcopy(adj_mat)
    n_nodes=_adj_mat.shape[0]
    for i in range(n_nodes):
        _adj_mat[i,i]=-1
    # -1はサンプリング済みの箇所か対角要素

    data=[]

    for i in range(n_nodes):
        # positiveをサンプリング
        idx_positives=np.where(_adj_mat[i,:]==1)[0]
        idx_negatives=np.where(_adj_mat[i,:]==0)[0]
        n_positives=min(len(idx_positives), n_max_positives)
        idx_positives=np.random.permutation(idx_positives)
        idx_negatives=np.random.permutation(idx_negatives)        

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_positives[0:n_positives]:
            data.append((i, j, 1)) # positive sample
            _adj_mat[i, j]=-1
            _adj_mat[j, i]=-1

            # 負例が不足した場合に備える。
            n_negatives=min(len(idx_negatives), n_max_negatives)
            for k in range(n_negatives):
                data.append((i, idx_negatives[k], 0))
                _adj_mat[i, k]=-1
                _adj_mat[k, i]=-1

            # サンプリングしたものを取り除く
            idx_negatives=idx_negatives[n_negatives:]

    data=np.random.permutation(data)

    train=data[0:int(len(data)*(1-val_size))]
    val=data[int(len(data)*(1-val_size)):]
    return train, val

def get_unobserved(
    adj_mat, 
    data
):
    # 観測された箇所が-1となる行列を返す。
    _adj_mat=deepcopy(adj_mat)
    n_nodes=_adj_mat.shape[0]

    for i in range(n_nodes):
        _adj_mat[i,i]=-1

    for datum in data:
        _adj_mat[datum[0], datum[1]]=-1
        _adj_mat[datum[1], datum[0]]=-1

    return _adj_mat

class Graph(Dataset):
    def __init__(
        self, 
        data
    ):
        self.data=torch.Tensor(data).long()
        self.n_items=len(data)

    def __len__(self):
        # データの長さを返す関数
        return self.n_items

    def __getitem__(self, i):
        # ノードとラベルを返す。
        return self.data[i, 0:2], self.data[i, 2]

class SamplingGraph(Dataset):
    def __init__(
        self, 
        adj_mat,
        n_max_data=1100,
        positive_size=1/11,
    ):  
        self.adj_mat=deepcopy(adj_mat)
        self.n_nodes=self.adj_mat.shape[0]
        for i in range(self.n_nodes):
            for j in range(i+1):
                self.adj_mat[i, j]=-1
        # 今まで観測されていないデータからのみノードの組をサンプルをする。
        data_unobserved=np.array(np.where(self.adj_mat!=-1)).T
        data_unobserved=np.random.permutation(data_unobserved)
        n_data=min(n_max_data, len(data_unobserved))
        data_sampled=data_unobserved[:n_data]
        # ラベルを人工的に作成する。
        labels=np.zeros(n_data)
        labels[0:int(n_data*positive_size)]=1
        # 連結してデータとする。
        self.data=np.concatenate([data_sampled, labels.reshape((-1, 1))], axis=1)
        self.data=torch.Tensor(self.data).long()
        self.n_items=len(self.data)

    def __len__(self):
        # データの長さを返す関数
        return self.n_items

    def __getitem__(self, i):
        # ノードとラベルを返す。
        return self.data[i, 0:2], self.data[i, 2]

# class RSGD(optim.Optimizer):

# class Lorentz(nn.Module):


if __name__=='__main__':
    # データセット作成
    adj_mat=hyperbolic_geometric_graph(n_nodes=15, n_dim=10, R=10, sigma=1, T=10)
    train, val=create_dataset(
        adj_mat=adj_mat,
        n_max_positives=1,
        n_max_negatives=5,
        val_size=0.1
    )
    # print(adj_mat)
    # print(train)

    u_adj_mat=get_unobserved(adj_mat, train)
    # print(u_adj_mat)

    # print(SamplingGraph(u_adj_mat, 22, 1/11)[0:22])

    # パラメータ
    burn_epochs=1000
    learning_rate=0.1
    burn_batch_size=32
    snml_epochs=100
    snml_learning_rate=0.01
    loader_workers=4
    shuffle=True

    # burn-inでの処理
    dataloader = DataLoader(
        Graph(train),
        shuffle=shuffle,
        batch_size=burn_batch_size,
        num_workers=loader_workers,
    )

    # for epoch in range(burn_epochs):
    #     rsgd.learning_rate = learning_rate
    #     for I, Ks in dataloader:
    #         print(I)
    #         rsgd.zero_grad()
    #         loss = net(I, Ks).mean()
    #         loss.backward()
    #         rsgd.step()

    # snmlの計算処理