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

from utils.utils import (
    arcosh,
    h_dist,
    lorentz_scalar_product,
    tangent_norm,
    exp_map,
    set_dim0,
    calc_likelihood_list,
    calc_log_C_D,
    integral_sinh,
    calc_beta_hat
)
from utils.utils_dataset import (
    get_unobserved,
    Graph,
    NegGraph,
    create_test_for_link_prediction,
    create_dataset_for_basescore,
    create_dataset
)


class RSGD(optim.Optimizer):
    """
    Riemaniann Stochastic Gradient Descentを行う関数。
    """

    def __init__(
        self,
        params,
        lr_embeddings,
        lr_beta,
        lr_sigma,
        R,
        # sigma_max,
        # sigma_min,
        beta_max,
        beta_min,
        device
    ):
        defaults = {
            "lr_embeddings": lr_embeddings,
            "lr_beta": lr_beta,
            "lr_sigma": lr_sigma,
            'R': R,
            # "sigma_max": sigma_max,
            # "sigma_min": sigma_min,
            "beta_max": beta_max,
            "beta_min": beta_min,
            "device": device
        }
        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:

            # betaとsigmaの更新
            beta = group["params"][0]
            # sigma = group["params"][1]

            beta_update = beta.data - \
                group["lr_beta"] * beta.grad.data
            beta_update = max(beta_update, group["beta_min"])
            beta_update = min(beta_update, group["beta_max"])
            if not math.isnan(beta_update):
                beta.data.copy_(torch.tensor(beta_update))

            # sigma_update = sigma.data - \
            #     group["lr_sigma"] * sigma.grad.data
            # sigma_update = max(sigma_update, group["sigma_min"])
            # sigma_update = min(sigma_update, group["sigma_max"])
            # if not math.isnan(sigma_update):
            #     sigma.data.copy_(torch.tensor(sigma_update))
            # print(group["params"])

            # うめこみの更新
            for p in group["params"][1:]:
                # print("p.grad:", p.grad)
                if p.grad is None:
                    continue
                B, D = p.size()
                gl = torch.eye(D, device=p.device, dtype=p.dtype)
                gl[0, 0] = -1
                grad_norm = torch.norm(p.grad.data)
                grad_norm = torch.where(
                    grad_norm > 1, grad_norm, torch.tensor(1.0, device=p.device))
                # only normalize if global grad_norm is more than 1
                h = (p.grad.data / grad_norm) @ gl
                proj = (
                    h
                    + (
                        lorentz_scalar_product(p, h)
                    ).unsqueeze(1)
                    * p
                )
                update = exp_map(p, -group["lr_embeddings"] * proj)
                is_nan_inf = torch.isnan(update) | torch.isinf(update)
                update = torch.where(is_nan_inf, p, update)
                # We've found that the performance sometimes deteriorates when we optimize z over the sphere radius R
                # Thus, we optimize embeddings in radius R*0.9
                # update = set_dim0(update, group["R"] * 0.90)
                update = set_dim0(update, group["R"] * 1.00)

                p.data.copy_(update)


class Lorentz(nn.Module):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        R,
        beta,
        sigma,
        sigma_min,
        sigma_max,
        # beta_min,
        # beta_max,
        init_range=0.01,
        sparse=True,
        device="cpu"
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.beta = nn.Parameter(torch.tensor(beta))
        # self.beta = beta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        # self.beta_min = beta_min
        # self.beta_max = beta_max
        # self.sigma = nn.Parameter(torch.tensor(sigma))
        self.R = R
        self.table = nn.Embedding(n_nodes, n_dim + 1, sparse=sparse)
        self.device = device

        self.I_D = torch.zeros(self.n_dim - 1)  # 0番目は空
        for j in range(1, self.n_dim - 1):
            numerator = lambda theta: np.sin(theta)**(self.n_dim - 1 - j)
            self.I_D[j] = integrate.quad(numerator, 0, np.pi)[0]

        print("分母:", self.I_D)

        self.avg_codelength = torch.zeros(self.n_dim - 1)

        for j in range(1, self.n_dim - 1):
            numerator = lambda theta: -(np.sin(theta)**(self.n_dim - 1 - j) / self.I_D[j].numpy()) * (
                (self.n_dim - 1 - j) * np.log(np.sin(theta)) - np.log(self.I_D[j].numpy()))
            self.avg_codelength[j] = integrate.quad(numerator, 0, np.pi)[0]

        print("平均符号長:", self.avg_codelength)

        # nn.init.uniform_(self.table.weight, -init_range, init_range)
        nn.init.normal(self.table.weight, 0, init_range)

        # 0次元目をセット
        with torch.no_grad():
            set_dim0(self.table.weight, self.R)

    def sigma_hat(
        self
    ):
        x = self.table.weight.data.cpu()
        r = arcosh(x[:, 0].reshape((-1, 1))).numpy()
        r = np.where(r <= 1e-6, 1e-6, r)[:, 0]

        sigma_list, ret, sigma_hat = calc_likelihood_list(
            r, n_dim=self.n_dim, R=self.R, sigma_min=self.sigma_min, sigma_max=self.sigma_max, DIV=1000)

        print(r)

        self.sigma = sigma_hat
        print("sigma:", self.sigma)
        print("beta:", self.beta)

    # def beta_hat(
    #     self,
    #     train_graph,
    #     n_samples
    # ):
    #     x = self.table.weight.data
    #     # r = arcosh(x[:, 0].reshape((-1, 1))).numpy()
    #     # r = np.where(r <= 1e-6, 1e-6, r)[:, 0]

    #     # n_samples = int(n_nodes * 0.1)

    #     lr = calc_beta_hat(z=x, train_graph=train_graph, n_samples=n_samples,
    # R=self.R, beta_min=self.beta_min, beta_max=self.beta_max)

    #     self.beta = lr

    #     print("beta:", self.beta)

    def latent_lik(
        self,
        x,
        polar=False
    ):
        # 半径方向
        r = arcosh(torch.sqrt(
            1 + (x[:, 1:]**2).sum(dim=1, keepdim=True))).double()
        r = torch.where(r <= 1e-6, 1e-6, r)[:, 0]

        # rの尤度
        lik = -(self.n_dim - 1) * (torch.log(1 - torch.exp(-2 * self.sigma *
                                                           r) + 0.00001) + self.sigma * r - torch.log(torch.Tensor([2]).to(self.device)))

        # rの正規化項
        log_C_D = calc_log_C_D(n_dim=self.n_dim, sigma=self.sigma, R=self.R)

        lik = lik + log_C_D

        if polar:
            x_ = x[:, 1:]
            x_ = x_**2
            x_ = torch.cumsum(
                x_[:, torch.arange(self.n_dim - 1, -1, -1)], dim=1)  # j番目がDからD-jの和
            x_ = x_[:, torch.arange(self.n_dim - 1, -1, -1)]  # j番目がDからj+1の和
            x_ = torch.max(torch.Tensor([[0.000001]]).to(self.device), x_)
            # 角度方向
            sin_theta = torch.zeros(
                (x.shape[0], self.n_dim - 1)).to(self.device)
            for j in range(1, self.n_dim - 1):
                sin_theta[:, j] = (x_[:, j] / x_[:, j - 1])**0.5

            # 角度方向の尤度
            for j in range(1, self.n_dim - 1):
                lik = lik - (self.n_dim - 1 - j) * torch.log(sin_theta[:, j])
                # 正規化項を足す
                lik = lik + torch.log(self.I_D[j])

            lik = lik + torch.log(2 * torch.Tensor([np.pi])).to(self.device)

        else:
            # 角度方向の尤度
            for j in range(1, self.n_dim - 1):
                # 正規化項を足す
                lik = lik + torch.log(self.I_D[j])

            lik = lik + torch.log(2 * torch.Tensor([np.pi])).to(self.device)

            # ヤコビアン由来の項
            lik = lik + (self.n_dim - 1) * (torch.log(1 - torch.exp(-2 * r) +
                                                      0.00001) + r - torch.log(torch.Tensor([2]).to(self.device)))

            lik = lik + torch.log(1 + torch.exp(-2 * r) + 0.00001) + \
                r - torch.log(torch.Tensor([2]).to(self.device))

        return lik

    def forward(
        self,
        pairs,
        labels
    ):
        # zを与えた下でのyの尤度
        loss = self.lik_y_given_z(
            pairs,
            labels
        )

        # z自体のロス
        # 座標を取得
        us = self.table(pairs[:, 0])
        vs = self.table(pairs[:, 1])

        lik_us = self.latent_lik(us)
        lik_vs = self.latent_lik(vs)

        loss = loss + (lik_us + lik_vs) / (self.n_nodes - 1)

        return loss

    def lik_y_given_z(
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
            torch.logaddexp(torch.tensor([0.0]).to(
                self.device), self.beta * (dist - self.R)),
            torch.logaddexp(torch.tensor([0.0]).to(
                self.device), -self.beta * (dist - self.R))
        )

        return loss

    def z(
        self
    ):
        z = self.table.weight.data
        lik_z = self.latent_lik(z).sum().item()

        return lik_z

    def get_lorentz_table(self):
        return self.table.weight.data.cpu().numpy()

    def get_poincare_table(self):
        table = self.table.weight.data.cpu().numpy()
        return table[:, 1:] / (
            table[:, :1] + 1
        )  # diffeomorphism transform to poincare ball

    def calc_probability(
        self,
        samples,
    ):
        samples_ = torch.Tensor(samples).to(self.device).long()

        # 座標を取得
        us = self.table(samples_[:, 0])
        vs = self.table(samples_[:, 1])

        dist = h_dist(us, vs)
        p = torch.exp(-torch.logaddexp(torch.tensor([0.0]).to(
            self.device), self.beta * (dist - self.R)))
        print(p)

        return p.detach().cpu().numpy()

    def calc_dist(
        self,
        samples,
    ):
        samples_ = torch.Tensor(samples).to(self.device).long()

        # 座標を取得
        us = self.table(samples_[:, 0])
        vs = self.table(samples_[:, 1])

        dist = h_dist(us, vs)

        return dist.detach().cpu().numpy()

    def get_PC(
        self,
        sigma_max,
        sigma_min,
        beta_max,
        beta_min,
        sampling=True
    ):
        if sampling == False:
            # DNMLのPCの計算
            x_e = self.get_poincare_table()
        else:
            idx = np.array(range(self.n_nodes))
            idx = np.random.permutation(idx)[:int(self.n_nodes * 0.1)]
            x_e = self.get_poincare_table()[idx, :]

        n_nodes_sample = len(x_e)
        print(n_nodes_sample)

        norm_x_e_2 = np.sum(x_e**2, axis=1).reshape((-1, 1))
        denominator_mat = (1 - norm_x_e_2) * (1 - norm_x_e_2.T)
        numerator_mat = norm_x_e_2 + norm_x_e_2.T
        numerator_mat -= 2 * x_e.dot(x_e.T)
        # arccoshのエラー対策
        for i in range(n_nodes_sample):
            numerator_mat[i, i] = 0
        dist_mat = np.arccosh(1 + 2 * numerator_mat / denominator_mat)

        is_nan_inf = np.isnan(dist_mat) | np.isinf(dist_mat)
        dist_mat = np.where(is_nan_inf, 2 * self.R, dist_mat)
        # dist_mat
        X = self.R - dist_mat
        for i in range(n_nodes_sample):
            X[i, i] = 0

        # I_n
        def sqrt_I_n(
            beta
        ):
            return np.sqrt(np.sum(X**2 / ((np.cosh(beta * X / 2.0) * 2)**2)) / (n_nodes_sample * (n_nodes_sample - 1)))

        # I
        def sqrt_I(
            sigma
        ):
            # denominator = self.integral_sinh(self.n_dim - 1)
            denominator = integral_sinh(
                n=self.n_dim - 1, n_dim=self.n_dim, sigma=self.sigma, R=self.R, exp_C=self.sigma * self.R)

            numerator_1 = lambda r: (r**2) * ((np.exp(self.sigma * (r - self.R)) + np.exp(-self.sigma * (r + self.R)))**2) * (
                (np.exp(self.sigma * (r - self.R)) - np.exp(-self.sigma * (r + self.R)))**(self.n_dim - 3))
            first_term = ((self.n_dim - 1)**2) * \
                integrate.quad(numerator_1, 0, self.R)[0] / denominator

            numerator_2 = lambda r: r * (np.exp(self.sigma * (r - self.R)) + np.exp(-self.sigma * (r + self.R))) * (
                (np.exp(self.sigma * (r - self.R)) - np.exp(-self.sigma * (r + self.R)))**(self.n_dim - 2))
            second_term = (
                (self.n_dim - 1) * integrate.quad(numerator_2, 0, self.R)[0] / denominator)**2

            return np.sqrt(np.abs(first_term - second_term))

        return 0.5 * (np.log(self.n_nodes) + np.log(self.n_nodes - 1) - np.log(4 * np.pi)) + np.log(integrate.quad(sqrt_I_n, beta_min, beta_max)[0]), 0.5 * (np.log(self.n_nodes) - np.log(2 * np.pi)) + np.log(integrate.quad(sqrt_I, sigma_min, sigma_max)[0])


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
    device,
    k_folds=5,
    loader_workers=16,
    shuffle=True,
    sparse=False
):
    data, _ = create_dataset(
        adj_mat=adj_mat,
        n_max_positives=n_max_positives,
        n_max_negatives=n_max_negatives,
        val_size=0.0
    )

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

        train = data[train_index, :]
        val = data[val_index, :]

        dataloader = DataLoader(
            Graph(train),
            shuffle=shuffle,
            batch_size=burn_batch_size * (n_max_positives + n_max_negatives),
            num_workers=loader_workers,
            pin_memory=True
        )

        # Rは決め打ちするとして、Tは後々平均次数とRから推定する必要がある。
        # 平均次数とかから逆算できる気がする。
        model = Poincare(
            n_nodes=params_dataset['n_nodes'],
            n_dim=model_n_dim,  # モデルの次元
            R=params_dataset['R'],
            sigma=1.0,
            beta=1.0,
            init_range=0.001,
            sparse=sparse,
            device=device
        )
        # 最適化関数。
        rsgd = RSGD(
            model.parameters(),
            learning_rate=learning_rate,
            R=params_dataset['R'],
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            beta_max=beta_max,
            beta_min=beta_min,
            device=device
        )

        model.to(device)

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
        for pairs, labels in dataloader_all:
            pairs = pairs.to(device)
            labels = labels.to(device)

            CV_score += model(pairs, labels).sum().item()

    print("CV_score:", CV_score)
    return CV_score


def LinkPrediction(
    adj_mat,
    train_graph,
    positive_samples,
    negative_samples,
    lik_data,
    x_lorentz,
    params_dataset,
    model_n_dim,
    burn_epochs,
    burn_batch_size,
    n_max_positives,
    n_max_negatives,
    lr_embeddings,
    lr_epoch_10,
    lr_beta,
    lr_sigma,
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    device,
    loader_workers=16,
    shuffle=True,
    sparse=False,
    calc_groundtruth=False
):

    print("model_n_dim:", model_n_dim)

    print("pos data", len(positive_samples))
    print("neg data", len(negative_samples))
    print("len data", len(lik_data))

    # burn-inでの処理
    dataloader = DataLoader(
        NegGraph(train_graph, n_max_positives, n_max_negatives),
        shuffle=shuffle,
        batch_size=burn_batch_size,
        num_workers=loader_workers,
        pin_memory=True
    )

    # Rは決め打ちするとして、Tは後々平均次数とRから推定する必要がある。
    # 平均次数とかから逆算できる気がする。
    model = Lorentz(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        # beta_min=beta_min,
        # beta_max=beta_max,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta=1.0,
        init_range=0.001,
        sparse=sparse,
        device=device
    )
    # 最適化関数。
    rsgd = RSGD(
        model.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_sigma=lr_sigma,
        R=params_dataset['R'],
        # sigma_max=sigma_max,
        # sigma_min=sigma_min,
        beta_max=beta_max,
        beta_min=beta_min,
        device=device
    )

    model.to(device)

    loss_history = []

    start = time.time()

    for epoch in range(burn_epochs):
        # if epoch != 0 and epoch % 30 == 0:  # 10 epochごとに学習率を減少
        #     rsgd.param_groups[0]["lr_embeddings"] /= 5
        if epoch == 10:
            # batchサイズに対応して学習率変更
            rsgd.param_groups[0]["lr_embeddings"] = lr_epoch_10

        losses = []
        model.sigma_hat()
        # model.beta_hat(
        #     train_graph=train_graph,
        #     n_samples=int(len(train_graph)*0.5)
        # )
        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

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

    # 真のデータ数
    n_data = params_dataset['n_nodes'] * (params_dataset['n_nodes'] - 1)

    # サンプリングしたデータのみで尤度を計算する。
    dataloader_all = DataLoader(
        Graph(lik_data),
        shuffle=shuffle,
        batch_size=burn_batch_size * (n_max_negatives + n_max_positives) * 10,
        num_workers=loader_workers,
        pin_memory=True
    )

    # -2*log(p)の計算
    # basescore_y_and_z = 0
    basescore_y_given_z = 0
    for pairs, labels in dataloader_all:
        pairs = pairs.to(device)
        labels = labels.to(device)

        # basescore_y_and_z += model(pairs, labels).sum().item()
        basescore_y_given_z += model.lik_y_given_z(pairs, labels).sum().item()

    basescore_z = model.z()

    basescore_y_given_z = basescore_y_given_z * (n_data / len(lik_data)) / 2
    basescore_y_and_z = basescore_y_given_z + basescore_z

    AIC_naive = basescore_y_given_z + \
        (params_dataset['n_nodes'] * model_n_dim + 1)
    BIC_naive = basescore_y_given_z + ((params_dataset['n_nodes'] * model_n_dim + 1) / 2) * (
        np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] - 1) - np.log(2))

    pc_first, pc_second = model.get_PC(
        sigma_max, sigma_min, beta_max, beta_min, sampling=True)
    DNML_codelength = basescore_y_and_z + pc_first + pc_second

    # リンク予測の指標計算
    # positive_prob = model.calc_probability(positive_samples)
    # negative_prob = model.calc_probability(negative_samples)

    positive_prob = -model.calc_dist(positive_samples)
    negative_prob = -model.calc_dist(negative_samples)

    pred = np.append(positive_prob, negative_prob)
    ground_truth = np.append(np.ones(len(positive_prob)),
                             np.zeros(len(negative_prob)))

    AUC = metrics.roc_auc_score(ground_truth, pred)

    if calc_groundtruth:

        # 真の座標でのAUC
        # 座標を取得
        us = torch.Tensor(x_lorentz[positive_samples[:, 0], :])
        vs = torch.Tensor(x_lorentz[positive_samples[:, 1], :])

        dist = h_dist(us, vs)
        # p_positive = torch.exp(-torch.logaddexp(torch.tensor([0.0]), params_dataset["beta"] * (dist - params_dataset["R"]))).detach().cpu().numpy()
        p_positive = -dist.detach().cpu().numpy()

        # 座標を取得
        us = torch.Tensor(x_lorentz[negative_samples[:, 0], :])
        vs = torch.Tensor(x_lorentz[negative_samples[:, 1], :])

        dist = h_dist(us, vs)
        # p_negative = torch.exp(-torch.logaddexp(torch.tensor([0.0]), params_dataset["beta"] * (dist - params_dataset["R"]))).detach().cpu().numpy()
        p_negative = -dist.detach().cpu().numpy()

        pred_g = np.append(p_positive, p_negative)
        GT_AUC = metrics.roc_auc_score(ground_truth, pred_g)
        print("GT_AUC:", GT_AUC)
        gt_r = torch.Tensor(x_lorentz[:, 0])
        gt_r = torch.max(gt_r, torch.Tensor([1.0 + 0.00001]))

        es_r = torch.Tensor(model.get_lorentz_table()[:, 0])
        # es_r = torch.max(es_r, torch.Tensor([1.0 + 0.00001]))
        # es_r = torch.where(es_r <= 1.0+0.00001, torch.Tensor([1.0+0.0001]), es_r)[:, 0]

        print(gt_r)
        print(es_r)

        gt_r = arcosh(gt_r)
        es_r = arcosh(es_r)

        correlation, _ = stats.spearmanr(gt_r, es_r)
        print("Cor:", correlation)

    else:
        GT_AUC = None
        correlation = None

    print("p(y, z; theta):", basescore_y_and_z)
    print("p(y|z; theta):", basescore_y_given_z)
    print("p(z; theta):", basescore_z)
    print("DNML:", DNML_codelength)
    print("AIC naive:", AIC_naive)
    print("BIC_naive:", BIC_naive)
    print("AUC:", AUC)

    return basescore_y_and_z, basescore_y_given_z, basescore_z, DNML_codelength, pc_first, pc_second, AIC_naive, BIC_naive, AUC, GT_AUC, correlation, model


def DNML_HGG(
    adj_mat,
    params_dataset,
    model_n_dim,
    burn_epochs,
    burn_batch_size,
    n_max_positives,
    n_max_negatives,
    lr_embeddings,
    lr_epoch_10,
    lr_beta,
    lr_sigma,
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    device,
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
    model = Lorentz(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        beta=1.0,
        init_range=0.001,
        sparse=sparse,
        device=device
    )
    # 最適化関数。
    rsgd = RSGD(
        model.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_sigma=lr_sigma,
        R=params_dataset['R'],
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        beta_max=beta_max,
        beta_min=beta_min,
        device=device
    )

    model.to(device)

    loss_history = []

    start = time.time()

    for epoch in range(burn_epochs):
        # if epoch != 0 and epoch % 30 == 0:  # 10 epochごとに学習率を減少
            # rsgd.param_groups[0]["lr_embeddings"] /= 5
        if epoch == 10:
            rsgd.param_groups[0]["lr_embeddings"] = lr_epoch_10

        losses = []
        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

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

    # 尤度計算

    data, _ = create_dataset(
        adj_mat=adj_mat,
        n_max_positives=9999999999,
        n_max_negatives=9999999999,
        val_size=0.00
    )

    dataloader_all = DataLoader(
        Graph(data),
        shuffle=shuffle,
        batch_size=burn_batch_size * (n_max_negatives + n_max_positives),
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
        basescore_y_given_z += model.lik_y_given_z(pairs, labels).sum().item()

    basescore_z = model.z()

    AIC_naive = basescore_y_given_z + \
        (params_dataset['n_nodes'] * model_n_dim + 1)
    BIC_naive = basescore_y_given_z + ((params_dataset['n_nodes'] * model_n_dim + 1) / 2) * (
        np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] - 1) - np.log(2))

    pc_first, pc_second = model.get_PC(
        sigma_max, sigma_min, beta_max, beta_min)
    DNML_codelength = basescore_y_and_z + pc_first + pc_second

    print("p(y, z; theta):", basescore_y_and_z)
    print("p(y|z; theta):", basescore_y_given_z)
    print("p(z; theta):", basescore_z)
    print("DNML:", DNML_codelength)
    print("AIC naive:", AIC_naive)
    print("BIC_naive:", BIC_naive)

    return basescore_y_and_z, basescore_y_given_z, basescore_z, DNML_codelength, pc_first, pc_second, AIC_naive, BIC_naive, model.get_lorentz_table()


if __name__ == '__main__':
    # データセット作成
    # params_dataset = {
    #     'n_nodes': 150,
    #     'n_dim': 16,
    #     'R': 10,
    #     'sigma': 0.1,
    #     'beta': 0.3
    # }

    n_nodes = 6400

    # print("R:", np.log(n_nodes) - 0.5)

    print("R:", np.log(n_nodes))

    params_dataset = {
        'n_nodes': n_nodes,
        'n_dim': 4,
        'R': np.log(n_nodes),
        'sigma': 1,
        'beta': 0.4
    }

    # パラメータ
    burn_epochs = 800
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    lr_beta = 0.001
    lr_sigma = 0.001
    sigma_max = 10.0
    sigma_min = 0.1
    beta_min = 0.1
    beta_max = 10.0
    # それ以外
    loader_workers = 16
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 隣接行列
    adj_mat, x_lorentz = hyperbolic_geometric_graph(
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
    basescore_z_list = []
    DNML_codelength_list = []
    pc_first_list = []
    pc_second_list = []
    AIC_naive_list = []
    BIC_naive_list = []
    CV_score_list = []
    AUC_list = []
    GT_AUC_list = []
    Cor_list = []

    model_n_dims = [4, 8, 16, 32, 64]
    # model_n_dims = [16, 32, 64]

    # model_n_dims = [64]

    # model_n_dims = [64]

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat=adj_mat,
        params_dataset=params_dataset
    )

    # negative samplingの比率を平均次数から決定
    pos_train_graph = len(np.where(train_graph == 1)[0])
    neg_train_graph = len(np.where(train_graph == 0)[0])
    # print(pos_train_graph)
    ratio = neg_train_graph / pos_train_graph
    print("ratio:", ratio)

    # ratio=10

    n_max_negatives = int(n_max_positives * ratio)
    print("n_max_negatives:", n_max_negatives)
    lr_embeddings = 0.1
    lr_epoch_10 = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32 / 100  # batchサイズに対応して学習率変更

    for model_n_dim in model_n_dims:
        # basescore_y_and_z, basescore_y_given_z, basescore_z, DNML_codelength, pc_first, pc_second, AIC_naive, BIC_naive = DNML_HGG(
        #     adj_mat=adj_mat,
        #     params_dataset=params_dataset,
        #     model_n_dim=model_n_dim,
        #     burn_epochs=burn_epochs,
        #     burn_batch_size=burn_batch_size,
        #     n_max_positives=n_max_positives,
        #     n_max_negatives=n_max_negatives,
        #     lr_embeddings=lr_embeddings,
        #     lr_beta=lr_beta,
        #     lr_sigma=lr_sigma,
        #     sigma_min=sigma_min,
        #     sigma_max=sigma_max,
        #     beta_min=beta_min,
        #     beta_max=beta_max,
        #     device=device,
        #     loader_workers=16,
        #     shuffle=True,
        #     sparse=False
        # )
        # basescore_y_and_z_list.append(basescore_y_and_z)
        # basescore_y_given_z_list.append(basescore_y_given_z)
        # basescore_z_list.append(basescore_z)
        # DNML_codelength_list.append(DNML_codelength)
        # pc_first_list.append(pc_first)
        # pc_second_list.append(pc_second)
        # AIC_naive_list.append(AIC_naive)
        # BIC_naive_list.append(BIC_naive)

        basescore_y_and_z, basescore_y_given_z, basescore_z, DNML_codelength, pc_first, pc_second, AIC_naive, BIC_naive, AUC, GT_AUC, correlation, model = LinkPrediction(
            adj_mat=adj_mat,
            train_graph=train_graph,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            lik_data=lik_data,
            x_lorentz=x_lorentz,
            params_dataset=params_dataset,
            model_n_dim=model_n_dim,
            burn_epochs=burn_epochs,
            burn_batch_size=burn_batch_size,
            n_max_positives=n_max_positives,
            n_max_negatives=n_max_negatives,
            lr_embeddings=lr_embeddings,
            lr_epoch_10=lr_epoch_10,
            lr_beta=lr_beta,
            lr_sigma=lr_sigma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            device=device,
            loader_workers=16,
            shuffle=True,
            sparse=False,
            calc_groundtruth=True
        )
        basescore_y_and_z_list.append(basescore_y_and_z)
        basescore_y_given_z_list.append(basescore_y_given_z)
        basescore_z_list.append(basescore_z)
        DNML_codelength_list.append(DNML_codelength)
        pc_first_list.append(pc_first)
        pc_second_list.append(pc_second)
        AIC_naive_list.append(AIC_naive)
        BIC_naive_list.append(BIC_naive)
        AUC_list.append(AUC)
        GT_AUC_list.append(GT_AUC)
        Cor_list.append(correlation)

        torch.save(model.state_dict(), "result_" + str(model_n_dim) + ".pth")

        # CV_score = CV_HGG(
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
        #     device=device,
        #     k_folds=5,
        #     loader_workers=16,
        #     shuffle=True,
        #     sparse=False
        # )
        # CV_score_list.append(CV_score)

    result["model_n_dims"] = model_n_dims
    result["n_nodes"] = params_dataset["n_nodes"]
    result["n_dim"] = params_dataset["n_dim"]
    result["R"] = params_dataset["R"]
    result["sigma"] = params_dataset["sigma"]
    result["beta"] = params_dataset["beta"]
    result["DNML_codelength"] = DNML_codelength_list
    result["AIC_naive"] = AIC_naive_list
    result["BIC_naive"] = BIC_naive_list
    result["AUC"] = AUC_list
    result["GT_AUC"] = GT_AUC_list
    result["Cor"] = Cor_list
    result["basescore_y_and_z"] = basescore_y_and_z_list
    result["basescore_y_given_z"] = basescore_y_given_z_list
    result["basescore_z"] = basescore_z_list
    result["pc_first"] = pc_first_list
    result["pc_second"] = pc_second_list
    result["burn_epochs"] = burn_epochs
    result["burn_batch_size"] = burn_batch_size
    result["n_max_positives"] = n_max_positives
    result["n_max_negatives"] = n_max_negatives
    result["lr_embeddings"] = lr_embeddings
    result["lr_epoch_10"] = lr_epoch_10
    result["lr_beta"] = lr_beta
    result["lr_sigma"] = lr_sigma
    result["sigma_max"] = sigma_max
    result["sigma_min"] = sigma_min
    result["beta_max"] = beta_max
    result["beta_min"] = beta_min

    filepath = "result_lorentz.csv"

    if os.path.exists(filepath):
        result_previous = pd.read_csv(filepath)
        result = pd.concat([result_previous, result])
        result.to_csv(filepath, index=False)
    else:
        result.to_csv(filepath, index=False)

    # result.to_csv("result_lorentz.csv", index=False)
