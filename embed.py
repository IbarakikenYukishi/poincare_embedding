import warnings
warnings.simplefilter('ignore', UserWarning)
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
from datasets import hyperbolic_geometric_graph, connection_prob
from copy import deepcopy
import multiprocessing as multi
from functools import partial
from multiprocessing import Pool
import pandas as pd

np.random.seed(0)


def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))


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
        n_positives = min(len(idx_positives), n_max_positives)
        idx_positives = np.random.permutation(idx_positives)
        idx_negatives = np.random.permutation(idx_negatives)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_positives[0:n_positives]:
            data.append((i, j, 1))  # positive sample
            _adj_mat[i, j] = -1
            _adj_mat[j, i] = -1

            # 負例が不足した場合に備える。
            n_negatives = min(len(idx_negatives), n_max_negatives)
            for k in range(n_negatives):
                data.append((i, idx_negatives[k], 0))
                _adj_mat[i, k] = -1
                _adj_mat[k, i] = -1

            # サンプリングしたものを取り除く
            idx_negatives = idx_negatives[n_negatives:]

    data = np.random.permutation(data)

    train = data[0:int(len(data) * (1 - val_size))]
    val = data[int(len(data) * (1 - val_size)):]
    return train, val


def create_prob_matrix(x_e, n_nodes, R, T):

    prob_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            distance = h_dist(x_e[i].reshape((1, -1)),
                              x_e[j].reshape((1, -1)))[0]
            prob = connection_prob(distance, R, T)
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


class SamplingGraph(Dataset):

    def __init__(
        self,
        adj_mat,
        n_max_data=1100,
        positive_size=1 / 2,  # 一様にしないという手もある。
    ):
        self.adj_mat = deepcopy(adj_mat)
        self.n_nodes = self.adj_mat.shape[0]
        for i in range(self.n_nodes):
            for j in range(i + 1):
                self.adj_mat[i, j] = -1
        # 今まで観測されていないデータからのみノードの組をサンプルをする。
        data_unobserved = np.array(np.where(self.adj_mat != -1)).T
        data_unobserved = np.random.permutation(data_unobserved)
        n_data = min(n_max_data, len(data_unobserved))
        self.n_possibles = 2 * len(data_unobserved)
        data_sampled = data_unobserved[:n_data]
        # ラベルを人工的に作成する。
        labels = np.zeros(n_data)
        labels[0:int(n_data * positive_size)] = 1
        # 連結してデータとする。
        self.data = np.concatenate(
            [data_sampled, labels.reshape((-1, 1))], axis=1)
        self.data = torch.Tensor(self.data).long()
        self.n_items = len(self.data)

    def __len__(self):
        # データの長さを返す関数
        return self.n_items

    def __getitem__(self, i):
        # ノードとラベルを返す。
        return self.data[i, 0:2], self.data[i, 2]

    def get_all_data(self):
        return self.data[:, 0:2], self.data[:, 2], self.n_possibles


def projection(theta, R_e):
    # 多様体に入るようR_eの半径の円に押し込める。
    theta_norm = torch.norm(theta, dim=1)
    for i, norm in enumerate(theta_norm):
        if norm > R_e:
            theta[i] = (theta[i] * R_e) / norm


class RSGD(optim.Optimizer):
    """
    Riemaniann Stochastic Gradient Descentを行う関数。
    """

    def __init__(self, params, learning_rate, R):
        R_e = np.sqrt((np.cosh(R) - 1) / (np.cosh(R) + 1))  # 直交座標系での最適化の範囲
        defaults = {"learning_rate": learning_rate, 'R_e': R_e}
        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 勾配を元に更新。
                # torch.normがdim=1方向にまとめていることに注意。
                update = torch.clone(p.data)
                update -= group["learning_rate"] * \
                    p.grad.data * \
                    ((1 - (torch.norm(p, dim=1)**2).reshape((-1, 1)))**2 / 4)
                # 発散したところなどを補正
                is_nan_inf = torch.isnan(update) | torch.isinf(update)
                update = torch.where(is_nan_inf, p, update)
                # 半径R_eの球に入るように縮小
                projection(update, group["R_e"])
                # pのアップデート
                p.data.copy_(update)


def e_dist_2(u_e, v_e):
    return torch.sum((u_e - v_e)**2, axis=1)


def h_dist(u_e, v_e):
    ret = 1
    ret += (2 * e_dist_2(u_e, v_e)) / \
        ((1 - e_dist_2(0, u_e)) * (1 - e_dist_2(0, v_e)))
    return arcosh(ret)


class Poincare(nn.Module):

    def __init__(
        self,
        n_nodes,
        n_dim,
        R,
        T,
        init_range=0.001
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.T = T
        self.R = R
        self.table = nn.Embedding(n_nodes, n_dim)
        nn.init.uniform_(self.table.weight, -init_range, init_range)

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
        loss = torch.where(loss == 1, torch.log(torch.exp(
            (dist - self.R) / self.T) + 1), torch.log(1 + 1 / torch.exp((dist - self.R) / self.T)))
        return loss

    def get_poincare_table(self):
        return self.table.weight.data.numpy()


def _mle(idx, model, pairs, labels, n_iter, learning_rate, R):
    # あるデータの尤度を計算する補助関数。
    _model = deepcopy(model)
    rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
                R=R)
    pair = pairs[idx].reshape((1, -1))
    label = labels[idx].reshape((1, -1))

    for _ in range(n_iter):
        rsgd.zero_grad()
        loss = model(pair, label).mean()
        loss.backward()
        rsgd.step()

    return loss.item()


def calc_pc(model, pairs, labels, n_possibles, n_iter, learning_rate, R):
    # parametric complexityをサンプリングで計算する補助関数。
    n_samples = len(labels)

    mle = partial(_mle, model=model, pairs=pairs, labels=labels,
                  n_iter=n_iter, learning_rate=learning_rate, R=R)

    ctx = multi.get_context('spawn')  # pytorchを使うときはこれを使わないとダメらしい。
    p = ctx.Pool(multi.cpu_count() - 1)
    args = list(range(n_samples))
    res = p.map(mle, args)
    p.close()
    # resは-log p
    res = -np.array(res)  # log p
    res = np.exp(res)  # p
    res = np.mean(res)  # pの平均

    return np.log(n_possibles) + np.log(res)

if __name__ == '__main__':

    # データセット作成
    params_dataset = {
        'n_nodes': 500,
        'n_dim': 8,
        'R': 10,
        'sigma': 1,
        'T': 2
    }

    # パラメータ
    burn_epochs = 50
    learning_rate = 10
    burn_batch_size = 16
    # SNML用
    snml_n_iter = 10
    snml_learning_rate = 0.01
    snml_n_max_data = 500
    # それ以外
    loader_workers = 16
    shuffle = True

    model_n_dims = [2, 4, 8, 16, 32]

    result = pd.DataFrame()

    for n_graph in range(5): # データ5本で性能比較
        print("n_graph:", n_graph)
        # 隣接行列
        adj_mat = hyperbolic_geometric_graph(
            n_nodes=params_dataset['n_nodes'],
            n_dim=params_dataset['n_dim'],
            R=params_dataset['R'],
            sigma=params_dataset['sigma'],
            T=params_dataset['T']
        )
        train, val = create_dataset(
            adj_mat=adj_mat,
            n_max_positives=2,
            n_max_negatives=10,
            val_size=0.02
        )

        print(len(val))

        # 平均次数が少なくなるように手で調整する用
        print('average degree:', np.sum(adj_mat) / len(adj_mat))

        u_adj_mat = get_unobserved(adj_mat, train)


        for model_n_dim in model_n_dims:
            print("model_n_dim:", model_n_dim)
            # burn-inでの処理
            dataloader = DataLoader(
                Graph(train),
                shuffle=shuffle,
                batch_size=burn_batch_size,
                num_workers=loader_workers,
            )

            # Rは決め打ちするとして、Tは後々平均次数とRから推定する必要がある。
            # 平均次数とかから逆算できる気がする。
            model = Poincare(
                n_nodes=params_dataset['n_nodes'],
                n_dim=model_n_dim,  # モデルの次元
                R=params_dataset['R'],
                T=params_dataset['T'],
                init_range=0.001
            )
            # 最適化関数。
            rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
                        R=params_dataset['R'])

            loss_history = []

            for epoch in range(burn_epochs):
                if epoch != 0 and epoch % 10 == 0:  # 10 epochごとに学習率を減少
                    rsgd.param_groups[0]["learning_rate"] /= 5
                losses = []
                for pairs, labels in dataloader:
                    rsgd.zero_grad()
                    # print(model(pairs, labels))
                    loss = model(pairs, labels).mean()
                    loss.backward()
                    rsgd.step()
                    losses.append(loss)

                loss_history.append(torch.Tensor(losses).mean().item())
                print("epoch:", epoch, ", loss:",
                      torch.Tensor(losses).mean().item())

            # 以下ではmodelのみを流用する。
            # snmlの計算処理
            # こっちは1サンプルずつやる
            dataloader_snml = DataLoader(
                Graph(val),
                shuffle=shuffle,
                batch_size=1,
                num_workers=0,
            )

            snml_codelength = 0
            snml_codelength_history = []

            for pair, label in dataloader_snml:

                # parametric complexityのサンプリングによる計算
                sampling_data = SamplingGraph(
                    adj_mat=u_adj_mat,
                    n_max_data=snml_n_max_data,
                    positive_size=1 / 2  # 一様サンプリング以外は実装の修正が必要。
                )
                pairs, labels, n_possibles = sampling_data.get_all_data()

                snml_pc = calc_pc(model, pairs, labels, n_possibles,
                                  snml_n_iter, snml_learning_rate, params_dataset['R'])

                # valのデータでの尤度
                rsgd = RSGD(model.parameters(), learning_rate=snml_learning_rate,
                            R=params_dataset['R'])
                for _ in range(snml_n_iter):
                    rsgd.zero_grad()
                    loss = model(pair, label).mean()
                    loss.backward()
                    rsgd.step()

                snml_codelength += loss.item() + snml_pc
                print('snml_codelength:', snml_codelength)
                # print('-log p:', loss.item())
                # print('snml_pc:', snml_pc)

                # valで使用したデータの削除
                u_adj_mat[pair[0, 0], pair[0, 1]] = -1
                u_adj_mat[pair[0, 1], pair[0, 0]] = -1

                snml_codelength_history.append(snml_codelength)

            df_row = pd.DataFrame(
                {"model_n_dim": [model_n_dim], "snml_codelength": snml_codelength_history[-1]})
            result = pd.concat([result, df_row], axis=0)

        result.to_csv("result_"+str(n_graph)+".csv", index=False)
