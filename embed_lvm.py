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
from datasets import hyperbolic_geometric_graph, connection_prob, create_dataset
from copy import deepcopy
import torch.multiprocessing as multi
from functools import partial
import pandas as pd
import gc
import time
from torch import Tensor
from scipy import integrate

np.random.seed(0)

import matplotlib

matplotlib.use("Agg")  # this needs to come before other matplotlib imports
import matplotlib.pyplot as plt

plt.style.use("ggplot")


def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))


def create_prob_matrix(x_e, n_nodes, R, beta):

    prob_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            distance = h_dist(x_e[i].reshape((1, -1)),
                              x_e[j].reshape((1, -1)))[0]
            prob = connection_prob(distance, R, beta)
            prob_mat[i, j] = prob
            prob_mat[j, i] = prob

    return prob_mat


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
        n_positives = min(len(idx_positives), n_max_positives)
        n_negatives = min(len(idx_negatives), n_max_negatives)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_positives[0:n_positives]:
            data.append((i, j, 1))  # positive sample

        for j in idx_negatives[0:n_negatives]:
            data.append((i, j, 0))  # negative sample

        if n_positives + n_negatives < n_max_positives + n_max_negatives:
            rest = n_max_positives + n_max_negatives - \
                (n_positives + n_negatives)
            rest_idx = np.append(
                idx_positives[n_positives:], idx_negatives[n_negatives:])
            rest_label = np.append(np.ones(len(idx_positives) - n_positives), np.zeros(
                len(idx_negatives) - n_negatives))

            rest_data = np.append(rest_idx.reshape(
                (-1, 1)), rest_label.reshape((-1, 1)), axis=1).astype(np.int)

            rest_data = np.random.permutation(rest_data)

            # print(rest_data)

            for datum in rest_data[:rest]:
                # print(datum)
                data.append((i, datum[0], datum[1]))

            # print(data)

        data = np.random.permutation(data)

        torch.Tensor(data).long()

        # ノードとラベルを返す。
        return data[:, 0:2], data[:, 2]


# class SamplingGraph(Dataset):

#     def __init__(
#         self,
#         adj_mat,
#         n_max_data=1100,
#         positive_size=1 / 2,  # 一様にしないという手もある。
#     ):
#         self.adj_mat = deepcopy(adj_mat)
#         self.n_nodes = self.adj_mat.shape[0]
#         for i in range(self.n_nodes):
#             self.adj_mat[i, 0:i + 1] = -1
#         # print(self.adj_mat)
#         # 今まで観測されていないデータからのみノードの組をサンプルをする。
#         data_unobserved = np.array(np.where(self.adj_mat != -1)).T
#         n_data = min(n_max_data, len(data_unobserved))
#         self.n_possibles = 2 * len(data_unobserved)
#         data_sampled = np.random.choice(
#             np.arange(len(data_unobserved)), size=n_data, replace=False)
#         data_sampled = data_unobserved[data_sampled, :]
#         # ラベルを人工的に作成する。
#         labels = np.zeros(n_data)
#         labels[0:int(n_data * positive_size)] = 1
#         # 連結してデータとする。
#         self.data = np.concatenate(
#             [data_sampled, labels.reshape((-1, 1))], axis=1)
#         self.data = torch.Tensor(self.data).long()
#         self.n_items = len(self.data)

#     def __len__(self):
#         # データの長さを返す関数
#         return self.n_items

#     def __getitem__(self, i):
#         # ノードとラベルを返す。
#         return self.data[i, 0:2], self.data[i, 2]

#     def get_all_data(self):
#         return self.data[:, 0:2], self.data[:, 2], self.n_possibles


class RSGD(optim.Optimizer):
    """
    Riemaniann Stochastic Gradient Descentを行う関数。
    """

    def __init__(
        self,
        params,
        learning_rate,
        R,
        sigma_max,
        sigma_min,
        beta_max,
        beta_min
    ):
        R_e = np.sqrt((np.cosh(R) - 1) / (np.cosh(R) + 1))  # 直交座標系での最適化の範囲
        defaults = {
            "learning_rate": learning_rate,
            'R_e': R_e,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "beta_max": beta_max,
            "beta_min": beta_min
        }
        super().__init__(params, defaults=defaults)

    def step(self):
        # print(self.param_groups)
        for group in self.param_groups:
            # print(group["params"].shape)

            # betaとsigmaの更新
            beta = group["params"][0]
            sigma = group["params"][1]

            beta_update = beta.data - group["learning_rate"] * beta.grad.data
            beta_update = max(beta_update, group["beta_min"])
            beta_update = min(beta_update, group["beta_max"])
            beta.data.copy_(torch.tensor(beta_update))

            sigma_update = sigma.data - \
                group["learning_rate"] * sigma.grad.data
            sigma_update = max(sigma_update, group["sigma_min"])
            sigma_update = min(sigma_update, group["sigma_max"])
            sigma.data.copy_(torch.tensor(sigma_update))

            # print("beta:", beta)
            # print("sigma:", sigma)
            # print("sigma.grad:", sigma.grad)

            # うめこみの更新
            for p in group["params"][2:]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                # if d_p.is_sparse:
                #     p_sqnorm = torch.sum(
                #         p[d_p._indices()[0].squeeze()] ** 2, dim=1,
                #         keepdim=True
                #     ).expand_as(d_p._values())
                #     n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
                #     d_p = torch.sparse.DoubleTensor(
                #         d_p._indices(), n_vals, d_p.size())
                #     update = p.data - group["learning_rate"] * d_p

                # else:
                #     # 勾配を元に更新。
                #     # torch.normがdim=1方向にまとめていることに注意。
                #     update = torch.clone(p.data)
                #     update -= group["learning_rate"] * \
                #         d_p * \
                #         ((1 - (torch.norm(p, dim=1)**2).reshape((-1, 1)))**2 / 4)

                p_norm = torch.norm(p, dim=1, keepdim=True).reshape((-1, 1))
                d = group["learning_rate"] * d_p * ((1 - p_norm**2)**2 / 4)
                delta = group["learning_rate"] * d_p
                Delta = torch.sum(delta * d, dim=1, keepdim=True)
                F = torch.sum(delta * p, dim=1, keepdim=True)
                mu = torch.cosh(Delta) - 1

                h_2 = 1 - p_norm**2
                k = h_2 / (2 * torch.sqrt(torch.cosh(Delta) + 1)) * \
                    torch.sinc(
                        Delta / (1.0j * torch.Tensor([np.pi])))  # sincを書く
                k = k.real
                t_2 = 2 + mu * (1 + p_norm**2) - 2 * (F**2) * (k**2)
                xi = (-F * k) * (t_2 - 2 * mu * (p_norm**2) + 2 * (F**2) * (k**2)) - \
                    t_2 * torch.sqrt(t_2 - mu * p_norm **
                                     2 + 2 * (F**2) * (k**2))
                xi = xi / (4 * (p_norm**2) * mu * (F**2) *
                           (k**2) - 4 * (F**4) * (k**4) + t_2**2)

                update = torch.clone(p.data)
                update += (h_2 * k * xi - (2 * h_2 * F * (k**2) * (xi**2)) / (1 + torch.sqrt(
                    1 - 4 * (p_norm**2) * mu * (xi**2) + 4 * (F**2) * (k**2) * (xi**2)))) * delta
                update += (2 * h_2 * mu * (xi**2) / (1 + torch.sqrt(1 - 4 * (p_norm**2)
                                                                    * mu * (xi**2) + 4 * (F**2) * (k**2) * (xi**2)))) * torch.clone(p.data)
                # print(update)
                # print(p_update)
                # print(p_norm.shape)
                # print(d.shape)
                # print(delta.shape)
                # print(Delta.shape)
                # print(F.shape)
                # print(mu.shape)
                # print(h_2.shape)
                # print(k.shape)
                # print(t_2.shape)
                # print(xi.shape)
                # raise ValueError

                # 発散したところなどを補正
                is_nan_inf = torch.isnan(update) | torch.isinf(update)
                update = torch.where(is_nan_inf, p, update)
                # 半径R_eの球に入るように縮小
                # projection(update, group["R_e"])
                update.renorm_(p=2, dim=0, maxnorm=group["R_e"])
                # pのアップデート
                p.data.copy_(update)


def e_dist_2(u_e, v_e):
    return torch.sum((u_e - v_e)**2, dim=1)


def h_dist(u_e, v_e):
    ret = 1.0
    ret += (2.0 * e_dist_2(u_e, v_e)) / \
        ((1.0 - e_dist_2(0.0, u_e)) * (1.0 - e_dist_2(0.0, v_e)))
    return arcosh(ret)


class Poincare(nn.Module):

    def __init__(
        self,
        n_nodes,
        n_dim,
        R,
        beta,
        sigma,
        init_range=0.01,
        sparse=True,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.beta = nn.Parameter(torch.tensor(beta))
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.R = R
        self.table = nn.Embedding(n_nodes, n_dim, sparse=sparse)

        self.I_D = torch.zeros(self.n_dim - 1)  # 0番目は空
        for j in range(1, self.n_dim - 1):
            numerator = lambda theta: np.sin(theta)**(self.n_dim - 1 - j)
            self.I_D[j] = integrate.quad(numerator, 0, np.pi)[0]

        nn.init.uniform_(self.table.weight, -init_range, init_range)

    def integral_sinh(self, n):  # (exp(sigma*R)/2)^(D-1)で割った結果
        if n == 0:
            return self.R * (2 * torch.exp(-self.sigma * self.R))**(self.n_dim - 1)
        elif n == 1:
            return 1 / self.sigma * (1 + torch.exp(-2 * self.sigma * self.R) - 2 * torch.exp(-self.sigma * self.R)) * (2 * torch.exp(-self.sigma * self.R))**(self.n_dim - 2)
        else:
            ret = 1 / (self.sigma * n)
            ret = ret * (1 - torch.exp(-2 * self.sigma * self.R)
                         )**(n - 1) * (1 + torch.exp(-2 * self.sigma * self.R))
            ret = ret * (2 * torch.exp(-self.sigma * self.R)
                         )**(self.n_dim - 1 - n)
            return ret - (n - 1) / n * self.integral_sinh(n - 2)

    def forward(
        self,
        pairs,
        labels
    ):
        # 座標を取得
        us = self.table(pairs[:, 0])
        vs = self.table(pairs[:, 1])

        # ロス計算
        dist = h_dist(us, vs)
        loss = torch.clone(labels).float()
        # 数値計算の問題をlogaddexpで回避
        # zを固定した下でのyのロス
        loss = torch.where(
            loss == 1,
            torch.logaddexp(torch.tensor([0]), self.beta * (dist - self.R)),
            torch.logaddexp(torch.tensor([0]), -self.beta * (dist - self.R))
        )

        # z自体のロス
        # rのロス
        def latent_lik(x):
            # 座標変換
            # 半径方向
            r = h_dist(x, torch.Tensor([[0.0]]))
            x_ = x**2
            x_ = torch.cumsum(
                x_[:, torch.arange(self.n_dim - 1, -1, -1)], dim=1)  # j番目がDからD-jの和
            x_ = x_[:, torch.arange(self.n_dim - 1, -1, -1)]  # j番目がDからj+1の和
            x_ = torch.max(torch.Tensor([[0.000001]]), x_)
            # 角度方向
            sin_theta = torch.zeros((x.shape[0], self.n_dim - 1))
            for j in range(1, self.n_dim - 1):
                sin_theta[:, j] = (x_[:, j] / x_[:, j - 1])**0.5

            # rの尤度
            lik = -(self.n_dim - 1) * (torch.log(1 - torch.exp(-2 * self.sigma *
                                                               r) + 0.00001) + self.sigma * r - torch.log(torch.Tensor([2])))

            # rの正規化項
            log_C_D = torch.Tensor([(self.n_dim - 1) * self.sigma *
                                    self.R - (self.n_dim - 1) * np.log(2)])  # 支配項
            # のこり。計算はwikipediaの再帰計算で代用してもいいかも
            # https://en.wikipedia.org/wiki/List_of_integrals_of_hyperbolic_functions

            C = torch.Tensor([self.integral_sinh(self.n_dim - 1)])
            log_C_D = log_C_D + torch.Tensor(torch.log(C))

            # print(log_C_D)

            lik = lik + log_C_D

            # print(lik)

            # 角度方向の尤度
            for j in range(1, self.n_dim - 1):
                # lik = lik-(self.n_dim - 1 - j) * torch.log(x_[:, j])
                lik = lik - (self.n_dim - 1 - j) * torch.log(sin_theta[:, j])
                # 正規化項を足す
                lik = lik + torch.log(self.I_D[j])

            lik = lik + torch.log(2 * torch.Tensor([np.pi]))

            return lik

        lik_us = latent_lik(us)
        lik_vs = latent_lik(vs)

        loss = loss + (lik_us + lik_vs) / (self.n_nodes - 1)

        return loss

    def y_given_z(
        self,
        pairs,
        labels
    ):
        # 座標を取得
        us = self.table(pairs[:, 0])
        vs = self.table(pairs[:, 1])

        # ロス計算
        dist = h_dist(us, vs)
        loss = torch.clone(labels).float()
        # 数値計算の問題をlogaddexpで回避
        # zを固定した下でのyのロス
        loss = torch.where(
            loss == 1,
            torch.logaddexp(torch.tensor([0]), self.beta * (dist - self.R)),
            torch.logaddexp(torch.tensor([0]), -self.beta * (dist - self.R))
        )

        return loss

    def get_poincare_table(self):
        return self.table.weight.data.cpu().numpy()

    def get_PC(
        self,
        sigma_max,
        sigma_min,
        beta_max,
        beta_min
    ):
        # DNMLのPCの計算
        x_e = self.get_poincare_table()
        norm_x_e_2 = np.sum(x_e**2, axis=1).reshape((-1, 1))
        denominator_mat = (1 - norm_x_e_2) * (1 - norm_x_e_2.T)
        numerator_mat = norm_x_e_2 + norm_x_e_2.T
        numerator_mat -= 2 * x_e.dot(x_e.T)
        # arccoshのエラー対策
        for i in range(self.n_nodes):
            numerator_mat[i, i] = 0
        dist_mat = np.arccosh(1 + 2 * numerator_mat / denominator_mat)
        X = self.R - dist_mat
        for i in range(self.n_nodes):
            X[i, i] = 0

        # I_n
        def sqrt_I_n(beta):
            return np.sqrt(np.sum(X**2 / ((np.cosh(beta * X / 2.0) * 2)**2)) / (self.n_nodes * (self.n_nodes - 1)))

        # I
        def sqrt_I(sigma):
            denominator = self.integral_sinh(self.n_dim - 1)
            numerator_1 = lambda r: (r**2) * ((torch.exp(self.sigma * (r - self.R)) + torch.exp(-self.sigma * (r + self.R)))**2) * (
                (torch.exp(self.sigma * (r - self.R)) - torch.exp(-self.sigma * (r + self.R)))**(self.n_dim - 3))
            first_term = ((self.n_dim - 1)**2) * \
                integrate.quad(numerator_1, 0, self.R)[0] / denominator

            numerator_2 = lambda r: r * (torch.exp(self.sigma * (r - self.R)) + torch.exp(-self.sigma * (r + self.R))) * (
                (torch.exp(self.sigma * (r - self.R)) - torch.exp(-self.sigma * (r + self.R)))**(self.n_dim - 2))
            second_term = (
                (self.n_dim - 1) * integrate.quad(numerator_2, 0, self.R)[0] / denominator)**2

            return torch.sqrt(first_term - second_term)

        return 0.5 * (np.log(self.n_nodes) + np.log(self.n_nodes - 1) - np.log(4 * np.pi)) + np.log(integrate.quad(sqrt_I_n, beta_min, beta_max)[0]), 0.5 * (np.log(self.n_nodes) - np.log(2 * np.pi)) + np.log(integrate.quad(sqrt_I, sigma_min, sigma_max)[0])


def plot_figure(adj_mat, table, path):
    # table = net.get_poincare_table()
    # skip padding. plot x y

    print(table.shape)

    plt.figure(figsize=(7, 7))

    _adj_mat = deepcopy(adj_mat)
    for i in range(len(_adj_mat)):
        _adj_mat[i, 0:i + 1] = -1

    edges = np.array(np.where(_adj_mat == 1)).T

    # print(edges)
    # print(table)

    for edge in edges:
        # print(edge)
        # print(table[edge[0]])
        # print(table[edge[1]])
        plt.plot(
            table[edge, 0],
            table[edge, 1],
            color="black",
            # marker="o",
            alpha=0.5,
        )
    plt.scatter(table[:, 0], table[:, 1])

    # plt.title(path)
    plt.gca().set_xlim(-1, 1)
    plt.gca().set_ylim(-1, 1)
    plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, edgecolor="black"))
    plt.savefig(path)
    plt.close()


def CV_HGG(
    adj_mat,
    params_dataset,
    model_n_dim,
    burn_epochs,
    burn_batch_size,
    n_max_positives,
    n_max_negatives,
    learning_rate,
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    k_folds=5,
    loader_workers=16,
    shuffle=True,
    sparse=False
):
    data, _ = create_dataset(
        adj_mat=adj_mat,
        n_max_positives=5,
        n_max_negatives=50,
        val_size=0.0
    )

    # print(data.shape)

    CV_score = 0

    for fold in range(k_folds):
        train_index = np.array([])
        val_index = np.array([])
        for j in range(k_folds):
            if j == fold:
                val_index = np.append(val_index, np.array(
                    range(int(len(data) * j / k_folds), int(len(data) * (j + 1) / k_folds))))
            else:
                train_index = np.append(train_index, np.array(
                    range(int(len(data) * j / k_folds), int(len(data) * (j + 1) / k_folds))))

        train_index = np.array(train_index).reshape(-1).astype(np.int)
        val_index = np.array(val_index).reshape(-1).astype(np.int)

        # print(train_index)
        # print(val_index)

        train = data[train_index, :]
        val = data[val_index, :]

        dataloader = DataLoader(
            Graph(train),
            shuffle=shuffle,
            batch_size=256,
            num_workers=loader_workers,
            pin_memory=True
        )

        # Rは決め打ちするとして、Tは後々平均次数とRから推定する必要がある。
        # 平均次数とかから逆算できる気がする。
        model = Poincare(
            n_nodes=params_dataset['n_nodes'],
            n_dim=model_n_dim,  # モデルの次元
            R=params_dataset['R'],
            sigma=params_dataset['sigma'],
            beta=params_dataset['beta'],
            init_range=0.001,
            sparse=sparse
        )
        # 最適化関数。
        rsgd = RSGD(
            model.parameters(),
            learning_rate=learning_rate,
            R=params_dataset['R'],
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            beta_max=beta_max,
            beta_min=beta_min
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device="cpu"
        # device = "cuda:0"
        model.to(device)
        # model=nn.DataParallel(model, device_ids=[0, 1, 2, 3])

        loss_history = []

        start = time.time()

        for epoch in range(burn_epochs):
            if epoch != 0 and epoch % 50 == 0:  # 10 epochごとに学習率を減少
                rsgd.param_groups[0]["learning_rate"] /= 5
            losses = []
            for pairs, labels in dataloader:

                pairs = pairs.to(device)
                labels = labels.to(device)

                rsgd.zero_grad()
                loss = model(pairs, labels).mean()
                loss.backward()
                rsgd.step()
                losses.append(loss)

            loss_history.append(torch.Tensor(losses).mean().item())
            print("epoch:", epoch, ", loss:",
                  torch.Tensor(losses).mean().item())

        elapsed_time = time.time() - start
        print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

        dataloader_all = DataLoader(
            Graph(val),
            shuffle=shuffle,
            batch_size=burn_batch_size,
            num_workers=loader_workers,
            pin_memory=True
        )

        # -2*log(p)の計算
        basescore_y_and_z = 0
        basescore_y_given_z = 0
        for pairs, labels in dataloader_all:
            pairs = pairs.to(device)
            labels = labels.to(device)

            CV_score += model(pairs, labels).sum().item()
            # basescore_y_given_z += model.y_given_z(pairs, labels).sum().item()

    print("CV_score:", CV_score)


def DNML_HGG(
    adj_mat,
    params_dataset,
    model_n_dim,
    burn_epochs,
    burn_batch_size,
    n_max_positives,
    n_max_negatives,
    learning_rate,
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    loader_workers=16,
    shuffle=True,
    sparse=False
):

    print("model_n_dim:", model_n_dim)
    # burn-inでの処理
    dataloader = DataLoader(
        NegGraph(adj_mat, n_max_positives, n_max_negatives),
        shuffle=shuffle,
        batch_size=burn_batch_size,
        num_workers=loader_workers,
        pin_memory=True
    )

    # Rは決め打ちするとして、Tは後々平均次数とRから推定する必要がある。
    # 平均次数とかから逆算できる気がする。
    model = Poincare(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=params_dataset['sigma'],
        beta=params_dataset['beta'],
        init_range=0.001,
        sparse=sparse
    )
    # 最適化関数。
    rsgd = RSGD(
        model.parameters(),
        learning_rate=learning_rate,
        R=params_dataset['R'],
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        beta_max=beta_max,
        beta_min=beta_min
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    # device = "cuda:0"
    model.to(device)
    # model=nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    loss_history = []

    start = time.time()

    for epoch in range(burn_epochs):
        if epoch != 0 and epoch % 50 == 0:  # 10 epochごとに学習率を減少
            rsgd.param_groups[0]["learning_rate"] /= 5
        losses = []
        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            rsgd.zero_grad()
            # print(model(pairs, labels))
            loss = model(pairs, labels).mean()
            loss.backward()
            rsgd.step()
            losses.append(loss)

        loss_history.append(torch.Tensor(losses).mean().item())
        print("epoch:", epoch, ", loss:",
              torch.Tensor(losses).mean().item())

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # 尤度計算

    data, _ = create_dataset(
        adj_mat=adj_mat,
        n_max_positives=9999999999,
        n_max_negatives=9999999999,
        val_size=0.00
    )

    print(data.shape)

    dataloader_all = DataLoader(
        Graph(data),
        shuffle=shuffle,
        batch_size=burn_batch_size*(n_max_negatives+n_max_positives),
        num_workers=loader_workers,
        pin_memory=True
    )

    # -2*log(p)の計算
    basescore_y_and_z = 0
    basescore_y_given_z = 0
    for pairs, labels in dataloader_all:
        pairs = pairs.to(device)
        labels = labels.to(device)

        basescore_y_and_z += model(pairs, labels).sum().item()
        basescore_y_given_z += model.y_given_z(pairs, labels).sum().item()

    AIC_naive = basescore_y_given_z + \
        params_dataset['n_nodes'] * model_n_dim
    BIC_naive = basescore_y_given_z + (params_dataset['n_nodes'] * model_n_dim / 2) * (
        np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] - 1) - np.log(2))

    pc_first, pc_second = model.get_PC(
        sigma_max, sigma_min, beta_max, beta_min)
    # print(pc_first, pc_second)
    DNML_codelength = basescore_y_and_z + pc_first + pc_second

    print("p(y, z; theta):", basescore_y_and_z)
    print("p(y|z; theta):", basescore_y_given_z)
    print("DNML:", DNML_codelength)
    print("AIC naive:", AIC_naive)
    print("BIC_naive:", BIC_naive)

    return basescore_y_and_z, basescore_y_given_z, DNML_codelength, AIC_naive, BIC_naive


if __name__ == '__main__':
    # データセット作成
    params_dataset = {
        'n_nodes': 1000,
        'n_dim': 5,
        'R': 10,
        'sigma': 0.1,
        'beta': 0.3
    }

    # パラメータ
    burn_epochs = 500
    burn_batch_size = 12
    n_max_positives = int(params_dataset["n_nodes"]*0.02)
    n_max_negatives = n_max_positives*10
    learning_rate = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32  # batchサイズに対応して学習率変更
    sigma_max = 1.0
    sigma_min = 0.001
    beta_min = 0.1
    beta_max = 10.0
    # それ以外
    loader_workers = 16
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = False

    # 隣接行列
    adj_mat, x_e = hyperbolic_geometric_graph(
        n_nodes=params_dataset['n_nodes'],
        n_dim=params_dataset['n_dim'],
        R=params_dataset['R'],
        sigma=params_dataset['sigma'],
        beta=params_dataset['beta']
    )

    # 平均次数が少なくなるように手で調整する用
    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    result = pd.DataFrame()
    basescore_y_and_z_list = []
    basescore_y_given_z_list = []
    DNML_codelength_list = []
    AIC_naive_list = []
    BIC_naive_list = []

    model_n_dims = [2, 3, 4, 5, 6, 7, 8]
    true_dims=[5, ]
    n_graphs=1

    for model_n_dim in model_n_dims:
        basescore_y_and_z, basescore_y_given_z, DNML_codelength, AIC_naive, BIC_naive = DNML_HGG(
            adj_mat=adj_mat,
            params_dataset=params_dataset,
            model_n_dim=model_n_dim,
            burn_epochs=burn_epochs,
            burn_batch_size=burn_batch_size,
            n_max_positives=n_max_positives,
            n_max_negatives=n_max_negatives,
            learning_rate=learning_rate,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            loader_workers=16,
            shuffle=True,
            sparse=False
        )
        basescore_y_and_z_list.append(basescore_y_and_z)
        basescore_y_given_z_list.append(basescore_y_given_z)
        DNML_codelength_list.append(DNML_codelength)
        AIC_naive_list.append(AIC_naive)
        BIC_naive_list.append(BIC_naive)

    result["model_n_dims"] = model_n_dims
    result["DNML_codelength"] = DNML_codelength_list
    result["AIC_naive"] = AIC_naive_list
    result["BIC_naive"] = BIC_naive_list
    result["basescore_y_and_z"] = basescore_y_and_z_list
    result["basescore_y_given_z"] = basescore_y_given_z_list

    result.to_csv("result.csv", index=False)

    # CV_HGG(
    #     adj_mat=adj_mat,
    #     params_dataset=params_dataset,
    #     model_n_dim=model_n_dim,
    #     burn_epochs=burn_epochs,
    #     burn_batch_size=burn_batch_size,
    #     n_max_positives=n_max_positives,
    #     n_max_negatives=n_max_negatives,
    #     learning_rate=learning_rate,
    #     sigma_min=sigma_min,
    #     sigma_max=sigma_max,
    #     beta_min=beta_min,
    #     beta_max=beta_max,
    #     k_folds=5,
    #     loader_workers=16,
    #     shuffle=True,
    #     sparse=False
    # )

    # plot_figure(adj_mat, model.get_poincare_table(), "embedding.png")
    # plot_figure(adj_mat, x_e, "original.png")
