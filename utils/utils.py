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


def arcosh(
    x,
    use_torch=True
):
    if use_torch:
        return torch.log(x + torch.sqrt(x - 1) * torch.sqrt(x + 1))
    else:
        return np.log(x + np.sqrt(x - 1) * np.sqrt(x + 1))


def lorentz_scalar_product(
    x,
    y,
    use_torch=True
):
    # 内積
    # 2次元の入力を仮定している。
    # BD, BD -> B
    m = x * y
    if use_torch:
        return m[:, 1:].sum(dim=1) - m[:, 0]
    else:
        return np.sum(m[:, 1:], axis=1) - m[:, 0]


def h_dist(
    u_e,
    v_e,
    use_torch=True
):
    dists = -lorentz_scalar_product(u_e, v_e, use_torch)
    if use_torch:
        dists = torch.where(dists <= 1, torch.ones_like(dists) + 1e-6, dists)
    else:
        dists = np.where(dists <= 1, 1e-6, dists)
    dists = arcosh(dists, use_torch)

    return dists


def tangent_norm(
    x,
    use_torch=True
):
    if use_torch:
        return torch.sqrt(lorentz_scalar_product(x, x, use_torch))
    else:
        return np.sqrt(lorentz_scalar_product(x, x, use_torch))


def exp_map(
    x,
    v
):
    # Exponential Map
    tn = tangent_norm(v).unsqueeze(dim=1)
    tn_expand = tn.repeat(1, x.size()[-1])
    result = torch.cosh(tn) * x + torch.sinh(tn) * (v / tn)
    result = torch.where(tn_expand > 0, result, x)
    return result


def set_dim0(x, R):
    x[:, 1:] = torch.renorm(x[:, 1:], p=2, dim=0,
                            maxnorm=np.sinh(R))  # 半径Rの範囲に収めたい
    # 発散しないように気を使う。
    x_max = torch.max(torch.abs(x[:, 1:]), dim=1, keepdim=True)[0].double()
    x_max = torch.where(x_max < 1.0, 1.0, x_max)

    dim0 = x_max * torch.sqrt((1 / x_max)**2 +
                              ((x[:, 1:] / x_max) ** 2).sum(dim=1, keepdim=True))
    x[:, 0] = dim0[:, 0]
    return x


def calc_log_C_D(n_dim, R, sigma):
    def integral_sinh_(n, n_dim, exp_C):  # (exp(exp_C)/2)^(D-1)で割った結果
        if n == 0:
            # print("n=0:", R * (2 * np.exp(-exp_C))**(n_dim - 1))
            return R * (2 * np.exp(-exp_C))**(n_dim - 1)
        elif n == 1:
            # print("n=1:", (1 / sigma) * (np.exp(sigma * R - exp_C) + np.exp(- sigma * R - exp_C) - 2 * np.exp(-exp_C)) * (2 * np.exp(-exp_C))**(n_dim - 2))
            return (1 / sigma) * (np.exp(sigma * R - exp_C) + np.exp(- sigma * R - exp_C) - 2 * np.exp(-exp_C)) * (2 * np.exp(-exp_C))**(n_dim - 2)
        else:
            ret = 1 / (sigma * n)
            # print("ret_1:", ret)
            ret = ret * (np.exp(sigma * R - exp_C) - np.exp(- sigma * R - exp_C)
                         )**(n - 1) * (np.exp(sigma * R - exp_C) + np.exp(- sigma * R - exp_C))
            # print("ret_2:", ret)
            ret = ret * (2 * np.exp(-exp_C)
                         )**(n_dim - 1 - n)
            # print("ret_3:", ret)
            return ret - (n - 1) / n * integral_sinh_(n=n - 2, n_dim=n_dim, exp_C=exp_C)

    sigma * R
    exp_C = max(1, sigma * R)  # sigma*Rが大きい時のみやる。

    log_C_D = (n_dim - 1) * exp_C - (n_dim - 1) * np.log(2)  # 支配項
    C = integral_sinh_(n=n_dim - 1, n_dim=n_dim, exp_C=exp_C)
    # sigma*Rが十分小さく、次元が十分大きい時にCが負になることがあり、nanを生成するのでその対策
    # そもそもsigma*Rがそんなに小さくならないように　sigmaの範囲を制限した方がいいかもしれない
    C = max(C, 0.00000001)
    # print("C:", C)
    log_C_D = log_C_D + np.log(C)

    return log_C_D


def plot_figure(adj_mat, table, path):
    # skip padding. plot x y

    print(table.shape)

    plt.figure(figsize=(7, 7))

    _adj_mat = deepcopy(adj_mat)
    for i in range(len(_adj_mat)):
        _adj_mat[i, 0:i + 1] = -1

    edges = np.array(np.where(_adj_mat == 1)).T

    for edge in edges:
        plt.plot(
            table[edge, 0],
            table[edge, 1],
            color="black",
            # marker="o",
            alpha=0.5,
        )
    plt.scatter(table[:, 0], table[:, 1])
    plt.gca().set_xlim(-1, 1)
    plt.gca().set_ylim(-1, 1)
    plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, edgecolor="black"))
    plt.savefig(path)
    plt.close()


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

if __name__ == "__main__":
    # All of the functions assumes the curvature of hyperbolic space is 1.
    # arcosh
    print("arcosh")
    x = torch.cosh(torch.Tensor([0., 2., 4., 8.]))
    print(arcosh(x))
    print(arcosh(x.numpy(), use_torch=False))

    # lorentz scalar product
    x = torch.Tensor([
        [1, 3, 0, 5],
        [-1, 2, 3, 0]
    ]
    )
    y = torch.Tensor([
        [5, 2, -1, 2],
        [8, 0, 1, 0]
    ]
    )

    print(lorentz_scalar_product(x, y))
    print(lorentz_scalar_product(x.numpy(), y.numpy(), use_torch=False))

    # h_dist
    print("h_dist")
    x = torch.Tensor([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]
    )
    y = torch.Tensor([
        [np.cosh(10), np.sinh(10), 0, 0],
        [np.cosh(10), 0, np.sinh(10), 0],
        [np.cosh(10), 0, 0, np.sinh(10)]
    ]
    )
    print(h_dist(x, y))
    print(h_dist(x.numpy(), y.numpy(), use_torch=False))

    # tangent norm
    print("tangent norm")
    x = torch.Tensor([
        [1, 2, -1, 0],
        [-1, 1, 2, 1],
        [1, -3, 1, 2]
    ]
    )
    print(tangent_norm(x))
    print(tangent_norm(x.numpy(), use_torch=False))

    # exp map
    print("exp map")
    x = torch.Tensor([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]
    )
    v = torch.Tensor([
        [0, np.sinh(1), 0, 0],
        [0, 0, np.sinh(1), 0],
        [0, 0, 0, np.sinh(1)]
    ]
    )

    print(exp_map(x, v))

    # set_dim0
    print("set_dim0")
    x = torch.Tensor([
        [1, 2, -1, 0],
        [-1, 1, 2, 1],
        [1, -3, 1, 2]
    ]
    )
    print(set_dim0(x, R=3))

    # calc_log_C_D
    print("calc_log_C_D")
    # R*sigma>=1
    print(calc_log_C_D(n_dim=64, R=10, sigma=1))
    print(calc_log_C_D(n_dim=32, R=7, sigma=0.5))
    print(calc_log_C_D(n_dim=16, R=9, sigma=1.5))
    print(calc_log_C_D(n_dim=2, R=10, sigma=2))
    # R*sigma<1
    # wolfram alphaの出力などと比べると正しくない。
    print(calc_log_C_D(n_dim=64, R=10, sigma=0.00001))
    print(calc_log_C_D(n_dim=32, R=7, sigma=0.00001))
    print(calc_log_C_D(n_dim=16, R=9, sigma=0.00001))
    print(calc_log_C_D(n_dim=2, R=10, sigma=0.00001))

    print(calc_log_C_D(n_dim=64, R=10, sigma=0.1))
    print(calc_log_C_D(n_dim=32, R=7, sigma=0.1))
    print(calc_log_C_D(n_dim=16, R=9, sigma=0.1))
    print(calc_log_C_D(n_dim=2, R=10, sigma=0.1))
