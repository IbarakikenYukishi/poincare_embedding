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
import functools as fts
from scipy import stats, special

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
    log_map,
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


@fts.lru_cache(maxsize=None)
def multigamma_ln(a, d):
    return special.multigammaln(a, d)


class RSGD(optim.Optimizer):
    """
    Riemaniann Stochastic Gradient Descentを行う関数。
    """

    def __init__(
        self,
        params,
        lr_embeddings,
        lr_beta,
        # lr_sigma,
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
            # "lr_sigma": lr_sigma,
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
        init_range=0.01,
        sparse=True,
        device="cpu",
        calc_latent=True
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.beta = nn.Parameter(torch.tensor(beta))
        self.R = R
        self.table = nn.Embedding(n_nodes, n_dim + 1, sparse=sparse)
        self.device = device
        self.calc_latent = calc_latent

        nn.init.normal(self.table.weight, 0, init_range)

        # 0次元目をセット
        with torch.no_grad():
            set_dim0(self.table.weight, self.R)

    def latent_lik(
        self,
        x
    ):
        pass

    def params_mle(
        self,
    ):
        pass

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

        if self.calc_latent:  # calc_latentがTrueの時のみ計算する
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
        pass


class PseudoUniform(Lorentz):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        R,
        beta,
        sigma,
        sigma_min,
        sigma_max,
        init_range=0.01,
        sparse=True,
        device="cpu",
        calc_latent=True
    ):
        super().__init__(
            n_nodes=n_nodes,
            n_dim=n_dim,  # 次元より1つ多くデータを取る必要があることに注意
            R=R,
            beta=beta,
            init_range=init_range,
            sparse=sparse,
            device=device,
            calc_latent=calc_latent
        )
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

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

    def params_mle(
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


class WrappedNormal(Lorentz):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        R,
        Sigma,
        beta,
        eps_1,
        init_range=0.01,
        sparse=True,
        device="cpu",
        calc_latent=True
    ):
        super().__init__(
            n_nodes=n_nodes,
            n_dim=n_dim,  # 次元より1つ多くデータを取る必要があることに注意
            R=R,
            beta=beta,
            init_range=init_range,
            sparse=sparse,
            device=device,
            calc_latent=calc_latent
        )
        self.Sigma = Sigma
        self.Sigma = self.Sigma.to(self.device)
        self.eps_1 = eps_1

        # nn.init.normal(self.table.weight, 0, init_range)

        # # 0次元目をセット
        # with torch.no_grad():
        #     set_dim0(self.table.weight, self.R)

    def params_mle(
        self
    ):
        z = self.table.weight.data
        mu = torch.zeros((self.n_nodes, self.n_dim + 1)).to(self.device)
        mu[:, 0] = 1

        v_ = log_map(z, mu)
        v = v_[:, 1:]

        self.Sigma = torch.mm(v.T / self.n_nodes, v)
        print(self.Sigma)

    def latent_lik(
        self,
        x,
    ):
        n_subnodes = x.shape[0]

        # 定数項
        lik = torch.ones(n_subnodes).to(self.device) * ((self.n_dim / 2) * torch.log(torch.tensor(2 * np.pi).to(self.device)) +
                                                        0.5 * torch.log(torch.det(self.Sigma) + 0.000001))

        # データから決まる項
        mu = torch.zeros((x.shape[0], self.n_dim + 1)).to(self.device)
        mu[:, 0] = 1
        v_ = log_map(x, mu)  # tangent vector
        v = v_[:, 1:]
        Sigma_pinv = torch.linalg.pinv(self.Sigma)  # Pseudo-inverse
        lik += 0.5 * torch.diag(v.mm(Sigma_pinv.mm(v.T)), 0)

        # -log Jacobian
        v_norm = tangent_norm(v_)
        v_norm = torch.where(
            v_norm <= 1e-6, torch.tensor(1e-6).to(self.device), v_norm)
        # print(v_norm)
        lik += (self.n_dim - 1) * (torch.log(1 - torch.exp(-2 * v_norm)) +
                                   v_norm - torch.tensor([np.log(2)]).to(self.device) - torch.log(v_norm))

        return lik

    def get_PC(
        self
    ):
        ret = 0
        ret += self.n_dim * np.log(2 / (self.n_dim - 1))
        ret += (1 - self.n_dim) * self.n_dim * np.log(self.eps_1) / 2
        ret += (self.n_nodes * self.n_dim / 2) * np.log(self.n_nodes /
                                                        (2 * np.e)) - multigamma_ln(self.n_dim / 2, self.n_dim)

        return ret


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
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    eps_1,
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

    # model
    model_hgg = PseudoUniform(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta=1.0,
        init_range=0.001,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_wnd = WrappedNormal(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        Sigma=torch.eye(model_n_dim),
        beta=1.0,
        eps_1=eps_1,
        init_range=0.001,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_naive = PseudoUniform(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta=1.0,
        init_range=0.001,
        sparse=sparse,
        device=device,
        calc_latent=False
    )

    # 最適化関数。
    rsgd_hgg = RSGD(
        model_hgg.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        device=device
    )
    rsgd_wnd = RSGD(
        model_wnd.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        device=device
    )
    rsgd_naive = RSGD(
        model_naive.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        device=device
    )

    model_hgg.to(device)
    model_wnd.to(device)
    model_naive.to(device)

    start = time.time()

    for epoch in range(burn_epochs):
        # if epoch != 0 and epoch % 30 == 0:  # 10 epochごとに学習率を減少
        #     rsgd.param_groups[0]["lr_embeddings"] /= 5
        if epoch == 10:
            # batchサイズに対応して学習率変更
            rsgd_hgg.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_wnd.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_naive.param_groups[0]["lr_embeddings"] = lr_epoch_10

        losses_hgg = []
        losses_wnd = []
        losses_naive = []

        # MLE
        model_hgg.params_mle()
        model_wnd.params_mle()

        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            # DNML-HGG
            rsgd_hgg.zero_grad()
            loss_hgg = model_hgg(pairs, labels).mean()
            loss_hgg.backward()
            rsgd_hgg.step()
            losses_hgg.append(loss_hgg)

            # DNML-WND
            rsgd_wnd.zero_grad()
            loss_wnd = model_wnd(pairs, labels).mean()
            loss_wnd.backward()
            rsgd_wnd.step()
            losses_wnd.append(loss_wnd)

            # Naive model
            rsgd_naive.zero_grad()
            loss_naive = model_naive(pairs, labels).mean()
            loss_naive.backward()
            rsgd_naive.step()
            losses_naive.append(loss_naive)

        print("epoch:", epoch, ", loss_hgg:",
              torch.Tensor(losses_hgg).mean().item())
        print("epoch:", epoch, ", loss_wnd:",
              torch.Tensor(losses_wnd).mean().item())
        print("loss_naive:",
              torch.Tensor(losses_naive).mean().item())

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

    # 尤度計算
    basescore_y_given_z_hgg = 0
    basescore_y_given_z_wnd = 0
    basescore_y_given_z_naive = 0
    for pairs, labels in dataloader_all:
        pairs = pairs.to(device)
        labels = labels.to(device)

        basescore_y_given_z_hgg += model_hgg.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_wnd += model_wnd.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_naive += model_naive.lik_y_given_z(
            pairs, labels).sum().item()

    basescore_z_hgg = model_hgg.z()
    basescore_z_wnd = model_wnd.z()

    basescore_y_given_z_hgg = basescore_y_given_z_hgg * \
        (n_data / len(lik_data)) / 2
    basescore_y_given_z_wnd = basescore_y_given_z_wnd * \
        (n_data / len(lik_data)) / 2
    basescore_y_given_z_naive = basescore_y_given_z_naive * \
        (n_data / len(lik_data)) / 2

    basescore_y_and_z_hgg = basescore_y_given_z_hgg + basescore_z_hgg
    basescore_y_and_z_wnd = basescore_y_given_z_wnd + basescore_z_wnd

    # Non-identifiable model
    AIC_naive = basescore_y_given_z_naive + \
        (params_dataset['n_nodes'] * model_n_dim + 1)
    BIC_naive = basescore_y_given_z_naive + ((params_dataset['n_nodes'] * model_n_dim + 1) / 2) * (
        np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] - 1) - np.log(2))

    # DNML-HGG
    pc_first, pc_second = model_hgg.get_PC(
        sigma_max, sigma_min, beta_max, beta_min, sampling=True)
    DNML_HGG = basescore_y_and_z_hgg + pc_first + pc_second

    AIC_HGG = basescore_y_and_z_hgg + 2
    BIC_HGG = basescore_y_and_z_hgg + 0.5 * (np.log(params_dataset['n_nodes']) + np.log(
        params_dataset['n_nodes'] - 1) - np.log(2)) + 0.5 * np.log(params_dataset['n_nodes'])

    # DNML-WND
    pc_wnd = model_wnd.get_PC()
    DNML_WND = basescore_y_and_z_wnd + pc_wnd
    AIC_WND = basescore_y_and_z_wnd + model_n_dim * (model_n_dim + 1) / 2 + 1
    BIC_WND = basescore_y_and_z_wnd + 0.5 * (np.log(params_dataset['n_nodes']) + np.log(
        params_dataset['n_nodes'] - 1) - np.log(2)) + (model_n_dim * (model_n_dim + 1) / 4) * np.log(params_dataset['n_nodes'])

    # Calculate AUC from probability
    def calc_AUC_from_prob(
        positive_dist,
        negative_dist
    ):

        pred = np.append(-positive_dist, -negative_dist)
        ground_truth = np.append(np.ones(len(positive_dist)),
                                 np.zeros(len(negative_dist)))
        AUC = metrics.roc_auc_score(ground_truth, pred)
        return AUC

    # latentを計算したものでのAUC
    AUC_HGG = calc_AUC_from_prob(
        model_hgg.calc_dist(positive_samples),
        model_hgg.calc_dist(negative_samples)
    )

    AUC_WND = calc_AUC_from_prob(
        model_wnd.calc_dist(positive_samples),
        model_wnd.calc_dist(negative_samples)
    )

    AUC_naive = calc_AUC_from_prob(
        model_naive.calc_dist(positive_samples),
        model_naive.calc_dist(negative_samples)
    )

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
        ground_truth = np.append(np.ones(len(p_positive)),
                                 np.zeros(len(p_negative)))

        AUC_GT = metrics.roc_auc_score(ground_truth, pred_g)
        print("AUC_GT:", AUC_GT)
        gt_r = torch.Tensor(x_lorentz[:, 0])
        gt_r = torch.max(gt_r, torch.Tensor([1.0 + 0.00001]))

        es_r_hgg = torch.Tensor(model_hgg.get_lorentz_table()[:, 0])
        es_r_wnd = torch.Tensor(model_wnd.get_lorentz_table()[:, 0])
        es_r_naive = torch.Tensor(model_naive.get_lorentz_table()[:, 0])
        # es_r = torch.max(es_r, torch.Tensor([1.0 + 0.00001]))
        # es_r = torch.where(es_r <= 1.0+0.00001, torch.Tensor([1.0+0.0001]), es_r)[:, 0]

        print(gt_r)
        print(es_r_hgg)
        print(es_r_wnd)
        print(es_r_naive)

        gt_r = arcosh(gt_r)
        es_r_hgg = arcosh(es_r_hgg)
        es_r_wnd = arcosh(es_r_wnd)
        es_r_naive = arcosh(es_r_naive)

        cor_hgg, _ = stats.spearmanr(gt_r, es_r_hgg)
        cor_wnd, _ = stats.spearmanr(gt_r, es_r_wnd)
        cor_naive, _ = stats.spearmanr(gt_r, es_r_naive)
        print("cor_hgg:", cor_hgg)
        print("cor_wnd:", cor_wnd)
        print("cor_naive:", cor_naive)

    else:
        AUC_GT = None
        cor_hgg = None
        cor_wnd = None
        cor_naive = None

    print("-log p_HGG(y, z):", basescore_y_and_z_hgg)
    print("-log p_WND(y, z):", basescore_y_and_z_wnd)
    print("-log p_HGG(y|z):", basescore_y_given_z_hgg)
    print("-log p_WND(y|z):", basescore_y_given_z_wnd)
    print("-log p_HGG(z):", basescore_z_hgg)
    print("-log p_WND(z):", basescore_z_wnd)
    print("-log p_naive(y; z):", basescore_y_given_z_naive)
    print("DNML-HGG:", DNML_HGG)
    print("DNML-WND:", DNML_WND)
    print("AIC_naive:", AIC_naive)
    print("BIC_naive:", BIC_naive)
    print("AIC_HGG:", AIC_HGG)
    print("BIC_HGG:", BIC_HGG)
    print("AIC_WND:", AIC_WND)
    print("BIC_WND:", BIC_WND)
    # print("AIC_naive_from_latent:", AIC_naive_from_latent)
    # print("BIC_naive_from_latent:", BIC_naive_from_latent)
    print("AUC_HGG:", AUC_HGG)
    print("AUC_WND:", AUC_WND)
    print("AUC_naive:", AUC_naive)
    print("AUC_GT:", AUC_GT)

    ret = {
        "DNML_HGG": DNML_HGG,
        "AIC_HGG": AIC_HGG,
        "BIC_HGG": BIC_HGG,
        "DNML_WND": DNML_WND,
        "AIC_WND": AIC_WND,
        "BIC_WND": BIC_WND,
        "AIC_naive": AIC_naive,
        "BIC_naive": BIC_naive,
        "AUC_HGG": AUC_HGG,
        "AUC_WND": AUC_WND,
        "AUC_naive": AUC_naive,
        "AUC_GT": AUC_GT,
        "cor_hgg": cor_hgg,
        "cor_wnd": cor_wnd,
        "cor_naive": cor_naive,
        "-log p_HGG(y, z)": basescore_y_and_z_hgg,
        "-log p_HGG(y|z)": basescore_y_given_z_hgg,
        "-log p_HGG(z)": basescore_z_hgg,
        "-log p_WND(y, z)": basescore_y_and_z_wnd,
        "-log p_WND(y|z)": basescore_y_given_z_wnd,
        "-log p_WND(z)": basescore_z_wnd,
        "-log p_naive(y; z)": basescore_y_given_z_naive,
        "pc_first": pc_first,
        "pc_second": pc_second,
        "model_hgg": model_hgg,
        "model_wnd": model_wnd,
        "model_naive": model_naive
    }

    return ret


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
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    eps_1,
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

    # model
    model_hgg = PseudoUniform(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta=1.0,
        init_range=0.001,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_wnd = WrappedNormal(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        Sigma=torch.eye(model_n_dim),
        beta=1.0,
        eps_1=eps_1,
        init_range=0.001,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_naive = PseudoUniform(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta=1.0,
        init_range=0.001,
        sparse=sparse,
        device=device,
        calc_latent=False
    )
    # 最適化関数。
    rsgd_hgg = RSGD(
        model_hgg.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        device=device
    )
    rsgd_wnd = RSGD(
        model_wnd.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        device=device
    )
    rsgd_naive = RSGD(
        model_naive.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        device=device
    )

    model_hgg.to(device)
    model_wnd.to(device)
    model_naive.to(device)

    start = time.time()

    for epoch in range(burn_epochs):
        # if epoch != 0 and epoch % 30 == 0:  # 10 epochごとに学習率を減少
            # rsgd.param_groups[0]["lr_embeddings"] /= 5
        if epoch == 10:
            rsgd_hgg.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_wnd.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_naive.param_groups[0]["lr_embeddings"] = lr_epoch_10

        losses_hgg = []
        losses_wnd = []
        losses_naive = []

        # MLE
        model_hgg.params_mle()
        model_wnd.params_mle()

        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            # DNML-HGG
            rsgd_hgg.zero_grad()
            loss_hgg = model_hgg(pairs, labels).mean()
            loss_hgg.backward()
            rsgd_hgg.step()
            losses_hgg.append(loss_hgg)

            # DNML-WND
            rsgd_wnd.zero_grad()
            loss_wnd = model_wnd(pairs, labels).mean()
            loss_wnd.backward()
            rsgd_wnd.step()
            losses_wnd.append(loss_wnd)

            # Naive model
            rsgd_naive.zero_grad()
            loss_naive = model_naive(pairs, labels).mean()
            loss_naive.backward()
            rsgd_naive.step()
            losses_naive.append(loss_naive)

        print("epoch:", epoch, ", loss_hgg:",
              torch.Tensor(losses_hgg).mean().item())
        print("epoch:", epoch, ", loss_wnd:",
              torch.Tensor(losses_wnd).mean().item())
        print("loss_naive:",
              torch.Tensor(losses_naive).mean().item())

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

    # 真のデータ数
    n_data = params_dataset['n_nodes'] * (params_dataset['n_nodes'] - 1)

    # # -2*log(p)の計算
    # basescore_y_and_z = 0
    # basescore_y_given_z = 0
    # basescore_y_given_z_naive = 0
    # for pairs, labels in dataloader_all:
    #     pairs = pairs.to(device)
    #     labels = labels.to(device)

    #     basescore_y_and_z += model_latent(pairs, labels).sum().item()
    #     basescore_y_given_z += model_latent.lik_y_given_z(
    #         pairs, labels).sum().item()
    #     basescore_y_given_z_naive += model_naive.lik_y_given_z(
    #         pairs, labels).sum().item()

    # basescore_z = model_latent.z()

    # AIC_naive = basescore_y_given_z_naive + \
    #     (params_dataset['n_nodes'] * model_n_dim + 1)
    # BIC_naive = basescore_y_given_z_naive + ((params_dataset['n_nodes'] * model_n_dim + 1) / 2) * (
    # np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] -
    # 1) - np.log(2))

    # AIC_naive_from_latent = basescore_y_given_z + \
    #     (params_dataset['n_nodes'] * model_n_dim + 1)
    # BIC_naive_from_latent = basescore_y_given_z + ((params_dataset['n_nodes'] * model_n_dim + 1) / 2) * (
    # np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] -
    # 1) - np.log(2))

    # pc_first, pc_second = model_latent.get_PC(
    #     sigma_max, sigma_min, beta_max, beta_min)
    # DNML_codelength = basescore_y_and_z + pc_first + pc_second

    # 尤度計算
    basescore_y_given_z_hgg = 0
    basescore_y_given_z_wnd = 0
    basescore_y_given_z_naive = 0
    for pairs, labels in dataloader_all:
        pairs = pairs.to(device)
        labels = labels.to(device)

        basescore_y_given_z_hgg += model_hgg.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_wnd += model_wnd.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_naive += model_naive.lik_y_given_z(
            pairs, labels).sum().item()

    basescore_z_hgg = model_hgg.z()
    basescore_z_wnd = model_wnd.z()

    # データを全部打ち込む場合はいらない
    # basescore_y_given_z_hgg = basescore_y_given_z_hgg * \
    #     (n_data / len(lik_data)) / 2
    # basescore_y_given_z_wnd = basescore_y_given_z_wnd * \
    #     (n_data / len(lik_data)) / 2
    # basescore_y_given_z_naive = basescore_y_given_z_naive * \
    #     (n_data / len(lik_data)) / 2

    basescore_y_and_z_hgg = basescore_y_given_z_hgg + basescore_z_hgg
    basescore_y_and_z_wnd = basescore_y_given_z_wnd + basescore_z_wnd

    # Non-identifiable model
    AIC_naive = basescore_y_given_z_naive + \
        (params_dataset['n_nodes'] * model_n_dim + 1)
    BIC_naive = basescore_y_given_z_naive + ((params_dataset['n_nodes'] * model_n_dim + 1) / 2) * (
        np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] - 1) - np.log(2))

    # DNML-HGG
    pc_first, pc_second = model_hgg.get_PC(
        sigma_max, sigma_min, beta_max, beta_min, sampling=True)
    DNML_HGG = basescore_y_and_z_hgg + pc_first + pc_second

    AIC_HGG = basescore_y_and_z_hgg + 2
    BIC_HGG = basescore_y_and_z_hgg + 0.5 * (np.log(params_dataset['n_nodes']) + np.log(
        params_dataset['n_nodes'] - 1) - np.log(2)) + 0.5 * np.log(params_dataset['n_nodes'])

    # DNML-WND
    pc_wnd = model_wnd.get_PC()
    DNML_WND = basescore_y_and_z_wnd + pc_wnd
    AIC_WND = basescore_y_and_z_wnd + model_n_dim * (model_n_dim + 1) / 2 + 1
    BIC_WND = basescore_y_and_z_wnd + 0.5 * (np.log(params_dataset['n_nodes']) + np.log(
        params_dataset['n_nodes'] - 1) - np.log(2)) + (model_n_dim * (model_n_dim + 1) / 4) * np.log(params_dataset['n_nodes'])

    print("-log p_HGG(y, z):", basescore_y_and_z_hgg)
    print("-log p_WND(y, z):", basescore_y_and_z_wnd)
    print("-log p_HGG(y|z):", basescore_y_given_z_hgg)
    print("-log p_WND(y|z):", basescore_y_given_z_wnd)
    print("-log p_HGG(z):", basescore_z_hgg)
    print("-log p_WND(z):", basescore_z_wnd)
    print("-log p_naive(y; z):", basescore_y_given_z_naive)
    print("DNML-HGG:", DNML_HGG)
    print("DNML-WND:", DNML_WND)
    print("AIC_HGG:", AIC_HGG)
    print("BIC_HGG:", BIC_HGG)
    print("AIC_WND:", AIC_WND)
    print("BIC_WND:", BIC_WND)
    print("AIC_naive:", AIC_naive)
    print("BIC_naive:", BIC_naive)

    ret = {
        "DNML_HGG": DNML_HGG,
        "AIC_HGG": AIC_HGG,
        "BIC_HGG": BIC_HGG,
        "DNML_WND": DNML_WND,
        "AIC_WND": AIC_WND,
        "BIC_WND": BIC_WND,
        "AIC_naive": AIC_naive,
        "BIC_naive": BIC_naive,
        "-log p_HGG(y, z)": basescore_y_and_z_hgg,
        "-log p_HGG(y|z)": basescore_y_given_z_hgg,
        "-log p_HGG(z)": basescore_z_hgg,
        "-log p_WND(y, z)": basescore_y_and_z_wnd,
        "-log p_WND(y|z)": basescore_y_given_z_wnd,
        "-log p_WND(z)": basescore_z_wnd,
        "-log p_naive(y; z)": basescore_y_given_z_naive,
        "pc_first": pc_first,
        "pc_second": pc_second,
        "model_hgg": model_hgg,
        "model_wnd": model_wnd,
        "model_naive": model_naive
    }

    # print("p(y, z; theta):", basescore_y_and_z)
    # print("p(y|z; theta):", basescore_y_given_z)
    # print("p(z; theta):", basescore_z)
    # print("p(y; z, theta):", basescore_y_given_z_naive)
    # print("DNML:", DNML_codelength)
    # print("AIC_naive:", AIC_naive)
    # print("BIC_naive:", BIC_naive)
    # print("AIC_naive_from_latent:", AIC_naive_from_latent)
    # print("BIC_naive_from_latent:", BIC_naive_from_latent)
    # # print("AUC_latent:", AUC_latent)
    # # print("AUC_naive:", AUC_naive)

    # ret = {
    #     "basescore_y_and_z": basescore_y_and_z,
    #     "basescore_y_given_z": basescore_y_given_z,
    #     "basescore_z": basescore_z,
    #     "basescore_y_given_z_naive": basescore_y_given_z_naive,
    #     "DNML_codelength": DNML_codelength,
    #     "pc_first": pc_first,
    #     "pc_second": pc_second,
    #     "AIC_naive": AIC_naive,
    #     "BIC_naive": BIC_naive,
    #     "AIC_naive_from_latent": AIC_naive_from_latent,
    #     "BIC_naive_from_latent": BIC_naive_from_latent,
    #     "model_latent": model_latent,
    #     "model_naive": model_naive
    # }

    return ret


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
    # burn_epochs = 5
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    lr_beta = 0.001
    sigma_max = 10.0
    sigma_min = 0.1
    beta_min = 0.1
    beta_max = 10.0
    eps_1 = 1e-6
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

    model_n_dims = [4, 8, 16, 32, 64]
    # model_n_dims = [16, 32, 64]
    # model_n_dims = [64]

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat=adj_mat,
        params_dataset=params_dataset
    )

    # negative samplingの比率を平均次数から決定
    pos_train_graph = len(np.where(train_graph == 1)[0])
    neg_train_graph = len(np.where(train_graph == 0)[0])
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
        ret = LinkPrediction(
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
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            eps_1=eps_1,
            device=device,
            loader_workers=16,
            shuffle=True,
            sparse=False,
            calc_groundtruth=True
        )

        torch.save(ret["model_hgg"].state_dict(),
                   "temp/result_" + str(model_n_dim) + "_hgg.pth")
        torch.save(ret["model_wnd"].state_dict(),
                   "temp/result_" + str(model_n_dim) + "_wnd.pth")
        torch.save(ret["model_naive"].state_dict(),
                   "temp/result_" + str(model_n_dim) + "_naive.pth")

        ret.pop('model_hgg')
        ret.pop('model_wnd')
        ret.pop('model_naive')

        ret["model_n_dims"] = model_n_dim
        ret["n_nodes"] = params_dataset["n_nodes"]
        ret["n_dim"] = params_dataset["n_dim"]
        ret["R"] = params_dataset["R"]
        ret["sigma"] = params_dataset["sigma"]
        ret["beta"] = params_dataset["beta"]
        ret["burn_epochs"] = burn_epochs
        ret["burn_batch_size"] = burn_batch_size
        ret["n_max_positives"] = n_max_positives
        ret["n_max_negatives"] = n_max_negatives
        ret["lr_embeddings"] = lr_embeddings
        ret["lr_epoch_10"] = lr_epoch_10
        ret["lr_beta"] = lr_beta
        ret["sigma_max"] = sigma_max
        ret["sigma_min"] = sigma_min
        ret["beta_max"] = beta_max
        ret["beta_min"] = beta_min
        ret["eps_1"] = eps_1

        row = pd.DataFrame(ret.values(), index=ret.keys()).T

        row = row.reindex(columns=[
            "model_n_dims",
            "n_nodes",
            "n_dim",
            "R",
            "sigma",
            "beta",
            "DNML_HGG",
            "AIC_HGG",
            "BIC_HGG",
            "DNML_WND",
            "AIC_WND",
            "BIC_WND",
            "AIC_naive",
            "BIC_naive",
            "AUC_HGG",
            "AUC_WND",
            "AUC_naive",
            "AUC_GT",
            "cor_hgg",
            "cor_wnd",
            "cor_naive",
            "-log p_HGG(y, z)",
            "-log p_HGG(y|z)",
            "-log p_HGG(z)",
            "-log p_WND(y, z)",
            "-log p_WND(y|z)",
            "-log p_WND(z)",
            "-log p_naive(y; z)",
            "pc_first",
            "pc_second",
            "burn_epochs",
            "n_max_positives",
            "n_max_negatives",
            "lr_embeddings",
            "lr_epoch_10",
            "lr_beta",
            "sigma_max",
            "sigma_min",
            "beta_max",
            "beta_min",
            "eps_1"
        ]
        )

        filepath = "result_lorentz.csv"

        if os.path.exists(filepath):
            result_previous = pd.read_csv(filepath)
            result = pd.concat([result_previous, row])
            result.to_csv(filepath, index=False)
        else:
            row.to_csv(filepath, index=False)
