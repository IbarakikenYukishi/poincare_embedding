import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import os
import sys
import torch
import random
import numpy as np
from torch import nn, optim, Tensor
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
from scipy import integrate
from sklearn import metrics
import math
from scipy import stats
from sklearn.linear_model import LogisticRegression

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


def e_dist(
    u_e,
    v_e,
    use_torch=True
):
    if use_torch:
        return torch.sum((u_e - v_e)**2, dim=1)
    else:
        return np.sum((u_e - v_e)**2, axis=1)


def h_dist_p(
    u_e,
    v_e,
    use_torch=True
):
    ret = 1.0
    ret += (2.0 * e_dist(u_e, v_e, use_torch)) / \
        ((1.0 - e_dist(0.0, u_e, use_torch)) * (1.0 - e_dist(0.0, v_e, use_torch)))
    return arcosh(ret, use_torch)


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


def log_map(
    z,
    mu
):
    alpha = -lorentz_scalar_product(z, mu).reshape((-1, 1)).float()
    alpha = torch.where(alpha > 1.0, alpha, torch.tensor(
        [1.0], dtype=alpha.dtype).to(alpha.device))
    u = (arcosh(alpha) / torch.sqrt(alpha**2 - 1 + 0.000001)) * \
        (z - alpha * mu)  # ゼロ除算を防ぐ
    return u


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


def integral_sinh(n, n_dim, sigma, R, exp_C):  # (exp(exp_C)/2)^(D-1)で割った結果
    if n == 0:
        return R * (2 * np.exp(-exp_C))**(n_dim - 1)
    elif n == 1:
        return (1 / sigma) * (np.exp(sigma * R - exp_C) + np.exp(- sigma * R - exp_C) - 2 * np.exp(-exp_C)) * (2 * np.exp(-exp_C))**(n_dim - 2)
    else:
        ret = 1 / (sigma * n)
        ret = ret * (np.exp(sigma * R - exp_C) - np.exp(- sigma * R - exp_C)
                     )**(n - 1) * (np.exp(sigma * R - exp_C) + np.exp(- sigma * R - exp_C))
        ret = ret * (2 * np.exp(-exp_C)
                     )**(n_dim - 1 - n)
        return ret - (n - 1) / n * integral_sinh(n=n - 2, n_dim=n_dim, sigma=sigma, R=R, exp_C=exp_C)


def calc_log_C_D(n_dim, sigma, R):

    exp_C = max(1, sigma * R)  # sigma*Rが大きい時のみやる。

    log_C_D = (n_dim - 1) * exp_C - (n_dim - 1) * np.log(2)  # 支配項
    C = integral_sinh(n=n_dim - 1, n_dim=n_dim, sigma=sigma, R=R, exp_C=exp_C)
    # sigma*Rが十分小さく、次元が十分大きい時にCが負になることがあり、nanを生成するのでその対策
    # そもそもsigma*Rがそんなに小さくならないように　sigmaの範囲を制限した方がいいかもしれない
    C = max(C, 0.00000001)
    # print("C:", C)
    log_C_D = log_C_D + np.log(C)

    return log_C_D


def calc_likelihood_list(r, n_dim, R, sigma_min, sigma_max, DIV=1000):
    sigma_list = np.arange(sigma_min, sigma_max, (sigma_max - sigma_min) / DIV)

    ret = []

    for sigma_ in sigma_list:
        # rの尤度
        lik = -(n_dim - 1) * (np.log(1 - np.exp(-2 * sigma_ *
                                                r) + 0.00001) + sigma_ * r - np.log(2))

        log_C_D = calc_log_C_D(n_dim=n_dim, sigma=sigma_, R=R)

        # # rの正規化項
        # log_C_D = (n_dim - 1) * sigma_ * R - (n_dim - 1) * np.log(2)  # 支配項
        # # のこり。計算はwikipediaの再帰計算で代用してもいいかも
        # # https://en.wikipedia.org/wiki/List_of_integrals_of_hyperbolic_functions
        # C = integral_sinh(n=n_dim - 1, n_dim=n_dim, R=R, sigma=sigma_)
        # log_C_D = log_C_D + np.log(C)

        lik = lik + log_C_D
        ret.append(np.sum(lik))
        sigma_hat = float(sigma_list[np.argmin(ret)])

    return sigma_list, np.array(ret), sigma_hat


def calc_beta_hat(z, train_graph, n_samples, R, beta_min, beta_max):
    n_nodes = z.shape[0]
    idx = np.array(range(n_nodes))
    idx = np.random.permutation(idx)[:n_samples]
    # print("n_samples:", n_samples)

    z_ = z[idx, :]

    first_term = - z_[:, :1] * z_[:, :1].T

    # remaining = x_e_[:, 1:].dot(x_e_[:, 1:].T)
    remaining = torch.mm(z_[:, 1:], z_[:, 1:].T)
    adj_mat_hat = - (first_term + remaining)
    for i in range(n_samples):
        adj_mat_hat[i, i] = 1

    # 数値誤差対策
    adj_mat_hat = adj_mat_hat.double()
    adj_mat_hat = torch.where(adj_mat_hat <= 1 + 1e-6, 1 + 1e-6, adj_mat_hat)

    # distance matrix
    adj_mat_hat = arcosh(adj_mat_hat)
    for i in range(n_samples):
        adj_mat_hat[i, i] = -1

    train_graph_ = train_graph[idx][:, idx]

    for i in range(n_samples):
        train_graph_[i, :i + 1] = -1

    adj_mat_hat = adj_mat_hat.cpu().numpy()

    y = train_graph_.flatten()
    x = adj_mat_hat.flatten()

    non_empty_idx = np.where(y != -1)[0]
    y = y[non_empty_idx].reshape((-1, 1))
    x = -(x[non_empty_idx] - R).reshape((-1, 1))

    # print(y)
    # print(x)

    lr = LogisticRegression()  # ロジスティック回帰モデルのインスタンスを作成
    lr.fit(x, y)  # ロジスティック回帰モデルの重みを学習

    return max(min(lr.coef_[0, 0], beta_max), beta_min)


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

    # log map
    print("log map")
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

    print(log_map(exp_map(x, v), x))

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
    # wolfram alphaの出力などと比べると正しくない
    print(calc_log_C_D(n_dim=64, R=10, sigma=0.00001))
    print(calc_log_C_D(n_dim=32, R=7, sigma=0.00001))
    print(calc_log_C_D(n_dim=16, R=9, sigma=0.00001))
    print(calc_log_C_D(n_dim=2, R=10, sigma=0.00001))

    print(calc_log_C_D(n_dim=64, R=10, sigma=0.1))
    print(calc_log_C_D(n_dim=32, R=7, sigma=0.1))
    print(calc_log_C_D(n_dim=16, R=9, sigma=0.1))
    print(calc_log_C_D(n_dim=2, R=10, sigma=0.1))
