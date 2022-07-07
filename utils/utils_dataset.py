import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
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
# from datasets import hyperbolic_geometric_graph, connection_prob, create_dataset, create_dataset_for_basescore
from copy import deepcopy
import torch.multiprocessing as multi
from functools import partial
import pandas as pd
import gc
import time
from torch import Tensor
from scipy import integrate
from sklearn import metrics
import math
from scipy import stats

np.random.seed(0)

import matplotlib

matplotlib.use("Agg")  # this needs to come before other matplotlib imports
import matplotlib.pyplot as plt

plt.style.use("ggplot")


class Graph(Dataset):

    def __init__(
        self,
        data
    ):
        self.data = torch.Tensor(data).long()
        self.n_items = len(data)

    def __len__(self):
        # データの長さを返す関数
        return self.n_items

    def __getitem__(self, i):
        # ノードとラベルを返す。
        return self.data[i, 0:2], self.data[i, 2]


class NegGraph(Dataset):

    def __init__(
        self,
        adj_mat,
        n_max_positives=5,
        n_max_negatives=50,
    ):
        # データセットを作成し、trainとvalidationに分ける
        self.n_max_positives = n_max_positives
        self.n_max_negatives = n_max_negatives
        self._adj_mat = deepcopy(adj_mat)
        self.n_nodes = self._adj_mat.shape[0]
        for i in range(self.n_nodes):
            self._adj_mat[i, i] = -1

    def __len__(self):
        # データの長さを返す関数
        return self.n_nodes

    def __getitem__(self, i):

        data = []

        # positiveをサンプリング
        idx_positives = np.where(self._adj_mat[i, :] == 1)[0]
        idx_negatives = np.where(self._adj_mat[i, :] == 0)[0]
        idx_positives = np.random.permutation(idx_positives)
        idx_negatives = np.random.permutation(idx_negatives)
        n_positives = min(len(idx_positives), self.n_max_positives)
        n_negatives = min(len(idx_negatives), self.n_max_negatives)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_positives[0:n_positives]:
            data.append((i, j, 1))  # positive sample

        for j in idx_negatives[0:n_negatives]:
            data.append((i, j, 0))  # negative sample

        if n_positives + n_negatives < self.n_max_positives + self.n_max_negatives:
            rest = self.n_max_positives + self.n_max_negatives - \
                (n_positives + n_negatives)
            rest_idx = np.append(
                idx_positives[n_positives:], idx_negatives[n_negatives:])
            rest_label = np.append(np.ones(len(idx_positives) - n_positives), np.zeros(
                len(idx_negatives) - n_negatives))

            rest_data = np.append(rest_idx.reshape(
                (-1, 1)), rest_label.reshape((-1, 1)), axis=1).astype(np.int)

            rest_data = np.random.permutation(rest_data)

            for datum in rest_data[:rest]:
                data.append((i, datum[0], datum[1]))

        data = np.random.permutation(data)

        torch.Tensor(data).long()

        # ノードとラベルを返す。
        return data[:, 0:2], data[:, 2]


def get_unobserved(
    adj_mat,
    data
):
    # 観測された箇所が-1となる行列を返す。
    _adj_mat = deepcopy(adj_mat)
    n_nodes = _adj_mat.shape[0]

    for i in range(n_nodes):
        _adj_mat[i, i] = -1

    for datum in data:
        _adj_mat[datum[0], datum[1]] = -1
        _adj_mat[datum[1], datum[0]] = -1

    return _adj_mat


def create_dataset_for_basescore(
    adj_mat,
    n_max_samples,
):
    # データセットを作成し、trainとvalidationに分ける
    _adj_mat = deepcopy(adj_mat)
    n_nodes = _adj_mat.shape[0]
    for i in range(n_nodes):
        _adj_mat[i, i] = -1
    # -1はサンプリング済みの箇所か対角要素

    data = []
    # print(np.sum(_adj_mat))

    for i in range(n_nodes):
        idx_samples = np.where(_adj_mat[i, :] != -1)[0]
        idx_samples = np.random.permutation(idx_samples)
        n_samples = min(len(idx_samples), n_max_samples)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_samples[0:n_samples]:
            data.append((i, j, _adj_mat[i, j]))

        # 隣接行列から既にサンプリングしたものを取り除く
        # _adj_mat[i, idx_samples[0:n_samples]] = -1
        # _adj_mat[idx_samples[0:n_samples], i] = -1

    # data = np.random.permutation(data)

    # train = data[0:int(len(data) * (1 - val_size))]
    # val = data[int(len(data) * (1 - val_size)):]
    print(np.sum(_adj_mat))

    return data


def create_test_for_link_prediction(
    adj_mat,
    params_dataset
):
    # testデータとtrain_graphを作成する
    n_total_positives = np.sum(adj_mat) / 2
    n_samples_test = int(n_total_positives * 0.1)
    n_neg_samples_per_positive = 1  # positive1つに対してnegativeをいくつサンプリングするか

    # positive sampleのサンプリング
    train_graph = np.copy(adj_mat)
    # 対角要素からはサンプリングしない
    for i in range(params_dataset["n_nodes"]):
        train_graph[i, i] = -1

    positive_samples = np.array(np.where(train_graph == 1)).T
    # 実質的に重複している要素を削除
    positive_samples_ = []
    for p in positive_samples:
        if p[0] > p[1]:
            positive_samples_.append([p[0], p[1]])
    positive_samples = np.array(positive_samples_)

    positive_samples = np.random.permutation(positive_samples)[:n_samples_test]

    # サンプリングしたデータをtrain_graphから削除
    for t in positive_samples:
        train_graph[t[0], t[1]] = -1
        train_graph[t[1], t[0]] = -1

    # negative sampleのサンプリング
    # permutationが遅くなるので直接サンプリングする
    negative_samples = []
    while len(negative_samples) < n_samples_test * n_neg_samples_per_positive:
        u = np.random.randint(0, params_dataset["n_nodes"])
        v = np.random.randint(0, params_dataset["n_nodes"])
        if train_graph[u, v] != 0:
            continue
        else:
            negative_samples.append([u, v])
            train_graph[u, v] = -1
            train_graph[v, u] = -1

    negative_samples = np.array(negative_samples)

    # これは重複を許す
    lik_data = create_dataset_for_basescore(
        adj_mat=train_graph,
        n_max_samples=int((params_dataset["n_nodes"] - 1) * 0.1)
    )

    return positive_samples, negative_samples, train_graph, lik_data


def create_dataset(
    adj_mat,
    n_max_positives=2,
    n_max_negatives=10,
    val_size=0.1
):
    # データセットを作成し、trainとvalidationに分ける
    _adj_mat = deepcopy(adj_mat)
    n_nodes = _adj_mat.shape[0]
    for i in range(n_nodes):
        _adj_mat[i, i] = -1
    # -1はサンプリング済みの箇所か対角要素

    data = []

    for i in range(n_nodes):
        # positiveをサンプリング
        idx_positives = np.where(_adj_mat[i, :] == 1)[0]
        idx_negatives = np.where(_adj_mat[i, :] == 0)[0]
        idx_positives = np.random.permutation(idx_positives)
        idx_negatives = np.random.permutation(idx_negatives)
        n_positives = min(len(idx_positives), n_max_positives)
        n_negatives = min(len(idx_negatives), n_max_negatives)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_positives[0:n_positives]:
            data.append((i, j, 1))  # positive sample

        # 隣接行列から既にサンプリングしたものを取り除く
        _adj_mat[i, idx_positives[0:n_positives]] = -1
        _adj_mat[idx_positives[0:n_positives], i] = -1

        for j in idx_negatives[0:n_negatives]:
            data.append((i, j, 0))  # positive sample

        # 隣接行列から既にサンプリングしたものを取り除く
        _adj_mat[i, idx_negatives[0:n_negatives]] = -1
        _adj_mat[idx_negatives[0:n_negatives], i] = -1

    data = np.random.permutation(data)

    train = data[0:int(len(data) * (1 - val_size))]
    val = data[int(len(data) * (1 - val_size)):]
    return train, val

if __name__ == "__main__":
    print("test")
