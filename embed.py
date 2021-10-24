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

np.random.seed(0)


def arcosh(x):
    return torch.log(x + torch.sqrt(x ** 2 - 1))


def create_dataset(
    adj_mat,
    n_max_positives=2,
    n_max_negatives=10,
    val_size=0.1
):
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
        positive_size=1 / 11,
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


def retraction(theta, R_e):
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

                update = torch.clone(p.data)
                update -= group["learning_rate"] * \
                    p.grad.data * ((1 - (torch.norm(p, dim=1)**2).reshape((-1,1)))**2 / 4)
                print('p:', p)
                print('p.grad:',p.grad.data)
                print('dif:', -group["learning_rate"]* \
                    p.grad.data* \
                    (((1-torch.norm(p)**2))**2/4))
                print('update:', update)
                print('一つ目の項:',group["learning_rate"])
                print('二つ目の項:',((1 - (torch.norm(p, dim=1)**2).reshape((-1,1)))**2 / 4))
                # 発散したところなどを補正
                is_nan_inf = torch.isnan(update) | torch.isinf(update)
                update = torch.where(is_nan_inf, p, update)
                # 半径R_eの球に入るように縮小
                retraction(update, group["R_e"])
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
        # 座標
        us = self.table(pairs[:, 0])
        vs = self.table(pairs[:, 1])

        # ロス計算
        dist = h_dist(us, vs)
        # prob_pos=1/(torch.exp((dist-self.R)/self.T)+1)
        # loss=-labels*torch.log(prob_pos)-(1-labels)*torch.log(1-prob_pos)
        # print(loss)

        loss = torch.clone(labels).float()
        loss = torch.where(loss == 1, torch.log(torch.exp(
            (dist - self.R) / self.T) + 1), torch.log(1 + 1 / torch.exp((dist - self.R) / self.T)))
        return loss

    def get_poincare_table(self):
        return self.table.weight.data.numpy()


if __name__ == '__main__':
    # データセット作成
    params_dataset = {
        'n_nodes': 10,
        'n_dim': 2,
        'R': 10,
        'sigma': 1,
        'T': 2
    }
    adj_mat = hyperbolic_geometric_graph(
        n_nodes=params_dataset['n_nodes'],
        n_dim=params_dataset['n_dim'],
        R=params_dataset['R'],
        sigma=params_dataset['sigma'],
        T=params_dataset['T']
    )
    # adj_mat=np.array([
    #     [0,0,1],
    #     [0,0,1],
    #     [1,1,0]
    # ])
    train, val = create_dataset(
        adj_mat=adj_mat,
        n_max_positives=2,
        n_max_negatives=10,
        val_size=0.1
    )

    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    u_adj_mat = get_unobserved(adj_mat, train)

    # パラメータ
    burn_epochs = 100
    learning_rate = 10
    burn_batch_size = 16
    # snml_epochs = 100
    # snml_learning_rate = 0.01
    loader_workers = 16
    shuffle = True

    # burn-inでの処理
    dataloader = DataLoader(
        Graph(train),
        shuffle=shuffle,
        batch_size=burn_batch_size,
        num_workers=loader_workers,
    )

    print(train)

    # Rは決め打ちするとして、Tは後々平均次数とRから推定する必要がある。
    model = Poincare(
        n_nodes=params_dataset['n_nodes'],
        n_dim=params_dataset['n_dim'],
        R=params_dataset['R'],
        T=params_dataset['T'],
        init_range=0.1
    )

    rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
                R=params_dataset['R'])
    rsgd.learning_rate=learning_rate
    # pairs, labels=iter(dataloader).next()
    # # print(pairs)
    # # print(labels)
    # pairs=torch.Tensor([0,1]).reshape(-1,2).long()
    # labels=torch.Tensor([0]).long()

    # print(model.get_poincare_table())

    # for i in range(10000):
    #     rsgd.zero_grad()
    #     loss=model(pairs, labels).mean()
    #     loss.backward()
    #     rsgd.step()

    # print(model.get_poincare_table())

    loss_history = []

    for epoch in range(burn_epochs):
        print(epoch)
        losses = []
        if epoch%10==9:
            rsgd.learning_rate/=10
        print(rsgd.learning_rate)
        for pairs, labels in dataloader:
            rsgd.zero_grad()
            loss = model(pairs, labels).mean()
            loss.backward()
            rsgd.step()
            losses.append(loss)

        loss_history.append(torch.Tensor(losses).mean())
        # print(torch.Tensor(losses).mean())

    print(loss_history)
    print(adj_mat)
    print(model.get_poincare_table())
    x_e = torch.Tensor(model.get_poincare_table())

    prob_mat = np.zeros((params_dataset['n_nodes'], params_dataset['n_nodes']))
    for i in range(params_dataset['n_nodes']):
        for j in range(i + 1, params_dataset['n_nodes']):
            distance = h_dist(x_e[i].reshape((1, -1)),
                              x_e[j].reshape((1, -1)))[0]
            prob = connection_prob(distance, params_dataset[
                                   'R'], params_dataset['T'])
            prob_mat[i, j] = prob
            prob_mat[j, i] = prob

    print(prob_mat)

    # snmlの計算処理
