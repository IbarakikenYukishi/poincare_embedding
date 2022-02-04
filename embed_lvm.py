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
            self.adj_mat[i, 0:i + 1] = -1
        # print(self.adj_mat)
        # 今まで観測されていないデータからのみノードの組をサンプルをする。
        data_unobserved = np.array(np.where(self.adj_mat != -1)).T
        n_data = min(n_max_data, len(data_unobserved))
        self.n_possibles = 2 * len(data_unobserved)
        data_sampled = np.random.choice(
            np.arange(len(data_unobserved)), size=n_data, replace=False)
        data_sampled = data_unobserved[data_sampled, :]
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
        beta_max
    ):
        R_e = np.sqrt((np.cosh(R) - 1) / (np.cosh(R) + 1))  # 直交座標系での最適化の範囲
        defaults = {
            "learning_rate": learning_rate,
            'R_e': R_e,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "beta_max": beta_max
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
            beta_update = max(beta_update, 0.1)
            beta_update = min(beta_update, group["beta_max"])
            beta.data.copy_(torch.tensor(beta_update))

            # sigma_update = sigma.data - \
            #     group["learning_rate"] * sigma.grad.data
            # sigma_update = max(sigma_update, group["sigma_min"])
            # sigma_update = min(sigma_update, group["sigma_max"])
            # sigma.data.copy_(torch.tensor(sigma_update))

            print("beta:", beta)
            print("sigma:", sigma)
            print("sigma.grad:", sigma.grad)

            # うめこみの更新
            for p in group["params"][2:]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if d_p.is_sparse:
                    p_sqnorm = torch.sum(
                        p[d_p._indices()[0].squeeze()] ** 2, dim=1,
                        keepdim=True
                    ).expand_as(d_p._values())
                    n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
                    d_p = torch.sparse.DoubleTensor(
                        d_p._indices(), n_vals, d_p.size())
                    update = p.data - group["learning_rate"] * d_p

                else:
                    # 勾配を元に更新。
                    # torch.normがdim=1方向にまとめていることに注意。
                    update = torch.clone(p.data)
                    update -= group["learning_rate"] * \
                        d_p * \
                        ((1 - (torch.norm(p, dim=1)**2).reshape((-1, 1)))**2 / 4)
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
            # D = x.shape[1]
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
            # sinの導出
            for j in range(1, self.n_dim - 1):
                sin_theta[:, j] = (x_[:, j] / x_[:, j - 1])**0.5

            # rの尤度
            # print(r)
            # lik=torch.where(
            #     r>0.001/self.sigma,
            #     -(self.n_dim - 1) * (torch.log(1 -torch.exp(-2*self.sigma * r)+0.00001) + self.sigma * r - torch.log(torch.Tensor([2]))),
            #     # -(self.n_dim - 1) * (torch.log(torch.sinh(self.sigma*r)))
            #     -(self.n_dim - 1) * (-torch.exp(-2*self.sigma*r)-0.5*torch.exp(-4*self.sigma*r) + self.sigma * r - torch.log(torch.Tensor([2]))) # マクローリン展開
            # )
            lik = -(self.n_dim - 1) * (torch.log(1 -torch.exp(-2*self.sigma * r)+0.00001) + self.sigma * r - torch.log(torch.Tensor([2])))

            # lik = -(self.n_dim - 1) * (torch.log(1 -torch.exp(-2*self.sigma * r)) + self.sigma * r - torch.log(torch.Tensor([2])))
            # print("途中の項:",torch.sinh(self.sigma*r))
            # print(torch.log(torch.sinh(self.sigma*r)))
            # print(self.sigma)
            # print("lik:", lik)
            # rの正規化項

            log_C_D = torch.Tensor([(self.n_dim - 1) * self.sigma *
                                    self.R - (self.n_dim - 1) * np.log(2)])  # 支配項
            # のこり。計算はwikipediaの再帰計算で代用してもいいかも
            # https://en.wikipedia.org/wiki/List_of_integrals_of_hyperbolic_functions

            def integral_sinh(n):
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
                    return ret - (n - 1) / n * integral_sinh(n - 2)

            C = torch.Tensor([integral_sinh(self.n_dim - 1)])

            # numerator = lambda r: ((torch.exp(
            #     self.sigma * (r - self.R)) - torch.exp(-self.sigma * (r + self.R))))**(self.n_dim - 1)
            # C = torch.Tensor([integrate.quad(numerator, 0, self.R)[0]])
            # print(C)
            log_C_D = log_C_D + torch.Tensor(torch.log(C))
            # print("log_C_D:", log_C_D)

            lik = lik + log_C_D

            # print(lik)

            # 角度方向の尤度
            for j in range(1, self.n_dim - 1):
                # lik = lik-(self.n_dim - 1 - j) * torch.log(x_[:, j])
                lik = lik - (self.n_dim - 1 - j) * torch.log(sin_theta[:, j])
                # 正規化項を足す
                lik = lik + torch.log(self.I_D[j])

            lik = lik + torch.log(2 * torch.Tensor([np.pi]))

            # print(lik)

            # raise ValueError

            return lik

        lik_us = latent_lik(us)
        lik_vs = latent_lik(vs)
        # print(self.I_D)
        # print((lik_vs + lik_us) / (self.n_nodes - 1))

        loss = loss + (lik_us + lik_vs) / (self.n_nodes - 1)
        # print("loss")
        # print(loss)

        return loss

    def get_poincare_table(self):
        return self.table.weight.data.cpu().numpy()

'''
def sampling_related_nodes(pair, label, dataset, n_samples=5):
    # uとvに関わるデータを5個ずつサンプリングする。
    # n_samplesが大きすぎるとエラーが出る可能性がある。
    u = pair[0, 0].item()
    v = pair[0, 1].item()
    u_indice = np.union1d(np.where(dataset[:, 0] == u)[
                          0], np.where(dataset[:, 1] == u)[0])
    u_indice = np.random.choice(u_indice, size=n_samples, replace=False)
    v_indice = np.union1d(np.where(dataset[:, 0] == v)[
                          0], np.where(dataset[:, 1] == v)[0])
    v_indice = np.random.choice(v_indice, size=n_samples, replace=False)

    pair_ = torch.cat((pair, torch.Tensor(
        dataset[u_indice, 0:2]), torch.Tensor(dataset[v_indice, 0:2])), dim=0).long()
    label_ = torch.cat((label.reshape((-1, 1)), torch.Tensor(dataset[
                       u_indice, 2].reshape((-1, 1))), torch.Tensor(dataset[v_indice, 2].reshape((-1, 1))))).long()

    del u_indice, v_indice

    return pair_, label_


def _mle(idx, model, pairs, labels, n_iter, learning_rate, R, dataset):
    print(idx)
    # あるデータの尤度を計算する補助関数。
    _model = deepcopy(model)
    rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
                R=R)
    pair = pairs[idx].reshape((1, -1))
    label = labels[idx]

    # uとvに関わるデータを5個ずつサンプリングする。
    pair_, label_ = sampling_related_nodes(pair, label, dataset)

    for _ in range(n_iter):
        rsgd.zero_grad()
        loss = model(pair_, label_).mean()
        loss.backward()
        rsgd.step()

    del pair_
    del label_

    return model(pair, label).item()


def calc_lik_pc_cpu(model, val_pair, val_label, pairs, labels, n_possibles, n_iter, learning_rate, R, dataset):
    # parametric complexityをサンプリングで計算する補助関数。
    n_samples = len(labels)

    # 0番目のデータにvalidationをおいておく。
    pairs_ = torch.cat((val_pair, pairs), dim=0)
    labels_ = torch.cat((val_label.reshape((-1, 1)),
                         labels.reshape((-1, 1))), dim=0)

    mle = partial(_mle, model=model, pairs=pairs_, labels=labels_,
                  n_iter=n_iter, learning_rate=learning_rate, R=R, dataset=dataset)

    # pytorchを使うときはこれを使わないとダメらしい。
    with multi.get_context('spawn').Pool(multi.cpu_count() - 1) as p:
        args = list(range(n_samples + 1))
        res = p.map(mle, args)
    # resは-log p
    lik = res[0]
    res = -np.array(res[1:])  # log p
    res = np.exp(res[1:])  # p
    res = np.mean(res[1:])  # pの平均

    del pairs_
    del labels_

    return lik, np.log(n_possibles) + np.log(res)


def calc_lik_pc_gpu(model, val_pair, val_label, pairs, labels, n_possibles, n_iter, learning_rate, R, dataset):
    # multiprocessingが動かないのでおまじない。
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    # parametric complexityをサンプリングで計算する補助関数。
    n_samples = len(labels)

    # 0番目のデータにvalidationをおいておく。
    pairs_ = torch.cat((val_pair, pairs), dim=0)
    labels_ = torch.cat((val_label.reshape((-1, 1)),
                         labels.reshape((-1, 1))), dim=0)

    res = []

    create_sub_dataset = partial(_create_sub_dataset, pairs=pairs_,
                                 labels=labels_, dataset=dataset)

    # pytorchを使用するときはspawnを指定
    with multi.get_context('spawn').Pool(multi.cpu_count() - 1) as p:
        args = list(range(n_samples))
        sub_dataset = p.map(create_sub_dataset, args)

    sub_dataset = torch.stack(sub_dataset, dim=0)

    # print(sub_dataset.shape)

    mle_gpu = partial(_mle_gpu, n_div=4, model=model, sub_dataset=sub_dataset,
                      n_iter=n_iter, learning_rate=learning_rate, R=R)

    multi.set_sharing_strategy('file_system')

    with multi.get_context('spawn').Pool(4) as p:
        args = list(range(4))
        res = p.map(mle_gpu, args)

    res = np.array(res).reshape(-1)

    # resは-log p
    lik = res[0]
    res = -np.array(res[1:])  # log p
    res = np.exp(res[1:])  # p
    res = np.mean(res[1:])  # pの平均

    del pairs_
    del labels_

    return lik, np.log(n_possibles) + np.log(res)


def _mle_gpu(idx, model, n_div, sub_dataset, n_iter, learning_rate, R):
    device = "cuda:" + str(idx)

    n_samples = len(sub_dataset)
    # gpuの個数で割ったデータ分だけ切り出し、GPUに送る。
    _sub_dataset = sub_dataset[
        int(n_samples * idx / n_div): int(n_samples * (idx + 1) / n_div)]
    _sub_dataset = _sub_dataset.to(device)

    #　返り値
    res = []

    for datum in _sub_dataset:
        pair_ = datum[:, 0:2]
        label_ = datum[:, 2]
        # モデルとoptimizer
        model_ = deepcopy(model)
        model_.to(device)
        rsgd = RSGD(model_.parameters(), learning_rate=learning_rate,
                    R=R)
        # データの組を準備
        for _ in range(n_iter):
            rsgd.zero_grad()
            loss = model_(pair_, label_).mean()
            loss.backward()
            rsgd.step()

        res.append(model_(pair_[0].reshape((1, -1)), label_[0]).item())

        del pair_
        del label_
        del model_

    return res


def _create_sub_dataset(idx, pairs, labels, dataset):
    pair = pairs[idx].reshape((1, -1))
    label = labels[idx]
    # uとvに関わるデータを5個ずつサンプリングする。
    pair_, label_ = sampling_related_nodes(pair, label, dataset)

    # tripletにして返す。
    triplets = torch.cat([pair_, label_], dim=1)

    return triplets
'''


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

if __name__ == '__main__':
    # データセット作成
    params_dataset = {
        'n_nodes': 128,
        'n_dim': 2,
        'R': 10,
        'sigma': 1.0,
        'beta': 0.5  # Tが小さすぎると最適化のときにinfが増えてバグる。
    }

    # パラメータ
    burn_epochs = 200
    burn_batch_size = 256
    learning_rate = 10.0 * burn_batch_size / 32  # batchサイズに対応して学習率変更
    # それ以外
    loader_workers = 16
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = True

    model_n_dim = 2

    result = pd.DataFrame()
    # 隣接行列
    adj_mat, x_e = hyperbolic_geometric_graph(
        n_nodes=params_dataset['n_nodes'],
        n_dim=params_dataset['n_dim'],
        R=params_dataset['R'],
        sigma=params_dataset['sigma'],
        beta=params_dataset['beta']
    )
    train, val = create_dataset(
        adj_mat=adj_mat,
        n_max_positives=5,
        n_max_negatives=20,
        val_size=0.02
    )

    print(len(val))

    # 平均次数が少なくなるように手で調整する用
    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    u_adj_mat = get_unobserved(adj_mat, train)

    print("model_n_dim:", model_n_dim)
    # burn-inでの処理
    dataloader = DataLoader(
        Graph(train),
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
    rsgd = RSGD(model.parameters(),
                learning_rate=learning_rate,
                R=params_dataset['R'],
                sigma_max=10.0,
                sigma_min=0.001,
                beta_max=20.0
                )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    # device = "cuda:0"
    model.to(device)
    # model=nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    loss_history = []

    start = time.time()

    for epoch in range(burn_epochs):
        if epoch != 0 and epoch % 25 == 0:  # 10 epochごとに学習率を減少
            rsgd.param_groups[0]["learning_rate"] /= 5
        losses = []
        for pairs, labels in dataloader:

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

    # -2*log(p)の計算
    basescore = 0
    for pairs, labels in dataloader:
        # print(model(pairs, labels))
        pairs = pairs.to(device)
        labels = labels.to(device)

        loss = model(pairs, labels).sum().item()
        basescore += 2 * loss

    print(basescore)

    plot_figure(adj_mat, model.get_poincare_table(), "embedding.png")
    plot_figure(adj_mat, x_e, "original.png")
