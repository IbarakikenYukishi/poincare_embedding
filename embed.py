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
from datasets import hyperbolic_geometric_graph, connection_prob
from copy import deepcopy
import multiprocessing as multi
from functools import partial
from multiprocessing import Pool
import pandas as pd
import gc
import time
from torch import Tensor

np.random.seed(0)

# narrow: 次元を指定し、指定した長さのデータを切り取る
# clamp: 最大値と最小値で抑える関数


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
        idx_positives = np.random.permutation(idx_positives)
        idx_negatives = np.random.permutation(idx_negatives)
        n_positives = min(len(idx_positives), n_max_positives)
        n_negatives = min(len(idx_negatives), n_max_negatives)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_positives[0:n_positives]:
            data.append((i, j, 1))  # positive sample
            _adj_mat[i, j] = -1
            _adj_mat[j, i] = -1

        for j in idx_negatives[0:n_negatives]:
            data.append((i, j, 0))  # positive sample
            _adj_mat[i, j] = -1
            _adj_mat[j, i] = -1

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
        # self.data = deepcopy(data)
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
        data_sampled = np.random.choice(np.arange(len(data_unobserved)), size=n_data, replace=False)
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

    def __init__(self, params, learning_rate, R):
        R_e = np.sqrt((np.cosh(R) - 1) / (np.cosh(R) + 1))  # 直交座標系での最適化の範囲
        defaults = {"learning_rate": learning_rate, 'R_e': R_e}
        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
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
        T,
        init_range=0.001,
        sparse=True
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.T = T
        self.R = R
        self.table = nn.Embedding(n_nodes, n_dim, sparse=sparse)
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


# def _mle(idx, model, n_iter, learning_rate, R, sub_dataset):
#     # あるデータの尤度を計算する補助関数。
#     _model = deepcopy(model)
#     rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
#                 R=R)
#     # pair = pairs[idx].reshape((1, -1))
#     # label = labels[idx]

#     # uとvに関わるデータを5個ずつサンプリングする。
#     pair_ = sub_dataset[idx][0]
#     label_ = sub_dataset[idx][1]

#     for _ in range(n_iter):
#         rsgd.zero_grad()
#         loss = model(pair_, label_).mean()
#         loss.backward()
#         rsgd.step()

#     del pair_
#     del label_

#     return model(pair, label).item()

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
    # parametric complexityをサンプリングで計算する補助関数。
    n_samples = len(labels)

    # 0番目のデータにvalidationをおいておく。
    pairs_ = torch.cat((val_pair, pairs), dim=0)
    labels_ = torch.cat((val_label.reshape((-1, 1)),
                         labels.reshape((-1, 1))), dim=0)

    res = []

    create_sub_dataset = partial(_create_sub_dataset, pairs=pairs_,
                                 labels=labels_, dataset=dataset)

    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    # pytorchを使うときはこれを使わないとダメらしい。
    with multi.get_context('spawn').Pool(multi.cpu_count() - 1) as p:
        args = list(range(n_samples))
        sub_dataset = p.map(create_sub_dataset, args)

    for sub_datum in sub_dataset:
        # モデルとoptimizer
        model_ = deepcopy(model)
        model_.to("cuda:0")
        rsgd = RSGD(model_.parameters(), learning_rate=learning_rate,
                    R=R)
        # データの組を準備
        pair_ = sub_datum[0]
        label_ = sub_datum[1]
        pair_ = pair_.to("cuda:0")
        label_ = label_.to("cuda:0")
        for _ in range(n_iter):
            rsgd.zero_grad()
            loss = model_(pair_, label_).mean()
            loss.backward()
            rsgd.step()

        res.append(model_(pair_[0].reshape((1, -1)), label_[0]).item())

        del pair_
        del label_
        del model_

    # resは-log p
    lik = res[0]
    res = -np.array(res[1:])  # log p
    res = np.exp(res[1:])  # p
    res = np.mean(res[1:])  # pの平均

    del pairs_
    del labels_

    return lik, np.log(n_possibles) + np.log(res)


def _create_sub_dataset(idx, pairs, labels, dataset):
    pair = pairs[idx].reshape((1, -1))
    label = labels[idx]
    # uとvに関わるデータを5個ずつサンプリングする。
    pair_, label_ = sampling_related_nodes(pair, label, dataset)

    # print(idx)
    # print(pair_, label_)

    return pair_, label_

# def calc_lik_pc_cpu(model, val_pair, val_label, pairs, labels, n_possibles, n_iter, learning_rate, R, dataset):
#     # parametric complexityをサンプリングで計算する補助関数。
#     n_samples = len(labels)

#     print("subdataset creation")
#     print(n_samples)

#     # データ作成
#     create_sub_dataset = partial(_create_sub_dataset, pairs=pairs,
#                              labels=labels, dataset=dataset)

#     # torch.multiprocessing.set_sharing_strategy('file_system')
#     # import resource
#     # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#     # resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


#     ctx = multi.get_context('spawn')  # pytorchを使うときはこれを使わないとダメらしい。
#     with ctx.Pool(4) as p:
#         args = list(range(n_samples))
#         sub_dataset = p.map(create_sub_dataset, args)

#     print(len(sub_dataset))
#     gc.collect()

#     mle = partial(_mle, model=model, n_iter=n_iter, learning_rate=learning_rate, R=R, sub_dataset=sub_dataset)

#     # mle = partial(_mle, model=model, pairs=pairs, labels=labels,
#     #               n_iter=n_iter, learning_rate=learning_rate, R=R, dataset=dataset)

#     ctx = multi.get_context('spawn')  # pytorchを使うときはこれを使わないとダメらしい。
#     with ctx.Pool(4) as p:
#         args = list(range(n_samples))
#         res = p.map(mle, args)
#     # resは-log p
#     res = -np.array(res)  # log p
#     res = np.exp(res)  # p
#     res = np.mean(res)  # pの平均

#     # validationデータの尤度計算
#     val_pair_, val_label_ = sampling_related_nodes(val_pair, val_label, dataset)

#     rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
#                 R=R)
#     for _ in range(n_iter):
#         rsgd.zero_grad()
#         loss = model(val_pair_, val_label_).mean()
#         loss.backward()
#         rsgd.step()

#     with torch.no_grad():
#         lik = model(val_pair, val_label).item()

#     return lik, np.log(n_possibles) + np.log(res)


def calc_pc_gpu(model, pairs, labels, n_possibles, n_iter, learning_rate, R, dataset):
    # parametric complexityをサンプリングで計算する補助関数。
    n_samples = len(labels)

    create_subdata = partial(_create_subdata, pairs=pairs,
                             labels=labels, dataset=dataset)

    ctx = multi.get_context('spawn')  # pytorchを使うときはこれを使わないとダメらしい。
    p = ctx.Pool(multi.cpu_count() - 1)
    args = list(range(n_samples))
    res = p.map(create_subdata, args)
    p.close()

    ret_loss = []

    print(res)
    print(res[0])
    print(res[1])

    # resは-log p
    ret_loss = -np.array(ret_loss)  # log p
    ret_loss = np.exp(ret_loss)  # p
    ret_loss = np.mean(ret_loss)  # pの平均

    return np.log(n_possibles) + np.log(ret_loss)


if __name__ == '__main__':
    # データセット作成
    params_dataset = {
        'n_nodes': 1000,
        'n_dim': 30,
        'R': 10,
        'sigma': 1,
        'T': 2
    }

    # パラメータ
    burn_epochs = 10
    burn_batch_size = 256
    learning_rate = 10.0 * burn_batch_size / 32  # batchサイズに対応して学習率変更
    # SNML用
    snml_n_iter = 10
    snml_learning_rate = 0.1
    snml_n_max_data = 500
    # それ以外
    loader_workers = 16
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = True

    model_n_dim = 30

    result = pd.DataFrame()
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
        T=params_dataset['T'],
        init_range=0.001,
        sparse=sparse
    )
    # 最適化関数。
    rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
                R=params_dataset['R'])

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    device = "cuda:0"
    model.to(device)
    # model=nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    loss_history = []

    start = time.time()

    for epoch in range(burn_epochs):
        if epoch != 0 and epoch % 10 == 0:  # 10 epochごとに学習率を減少
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
