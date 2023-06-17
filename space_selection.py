import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import os
import sys
import torch
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # this needs to come before other matplotlib imports
import matplotlib.pyplot as plt
import torch.multiprocessing as multi
import pandas as pd
import gc
import time
import math
from torch import nn, optim, Tensor
from tqdm import trange, tqdm
from collections import Counter
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from functools import partial
from scipy import integrate
from scipy.special import gammaln
from sklearn import metrics
from scipy import stats
from utils.utils import (
    arcosh,
    h_dist,
    e_dist,
    lorentz_scalar_product,
    tangent_norm,
    exp_map,
    log_map,
    set_dim0,
    calc_likelihood_list,
    calc_log_C_D,
    integral_sinh,
    calc_beta_hat,
    multigamma_ln,
    sin_k,
    cos_k,
    arcos_k,
    inner_product_k,
    dist_k,
    # s_dist_k,
    # h_dist_k,
    tangent_norm_k,
    exp_map_k,
    projection_k,
    log_map_k,
    approx_W_k,
    mle_truncated_normal,
    calc_spherical_complexity
)
from utils.utils_dataset import (
    get_unobserved,
    Graph,
    NegGraph,
    create_test_for_link_prediction,
    create_dataset_for_basescore,
    create_dataset
)
from datasets import (
    hyperbolic_geometric_graph,
    connection_prob,
    wrapped_normal_distribution,
    euclidean_geometric_graph,
    init_HGG
)

np.random.seed(0)
plt.style.use("ggplot")


class RSGD(optim.Optimizer):
    """
    Riemaniann Stochastic Gradient Descentを行う関数。
    """

    def __init__(
        self,
        params,
        lr_embeddings,
        lr_beta,
        lr_gamma,
        R,
        k,
        beta_max,
        beta_min,
        gamma_max,
        gamma_min,
        perturbation,
        device
    ):
        defaults = {
            "lr_embeddings": lr_embeddings,
            "lr_beta": lr_beta,
            "lr_gamma": lr_gamma,
            "R": R,
            "k": k,
            "beta_max": beta_max,
            "beta_min": beta_min,
            "gamma_max": gamma_max,
            "gamma_min": gamma_min,
            "perturbation": perturbation,
            "device": device
        }
        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:

            # betaとsigmaの更新
            # 全てのモデルで共通
            beta = group["params"][0]
            gamma = group["params"][1]

            beta_update = beta.data - \
                group["lr_beta"] * beta.grad.data
            beta_update = max(beta_update, group["beta_min"])
            beta_update = min(beta_update, group["beta_max"])
            if not math.isnan(beta_update):
                beta.data.copy_(torch.tensor(beta_update))

            gamma_update = gamma.data - \
                group["lr_gamma"] * gamma.grad.data
            gamma_update = max(gamma_update, group["gamma_min"])
            gamma_update = min(gamma_update, group["gamma_max"])
            if not math.isnan(gamma_update):
                gamma.data.copy_(torch.tensor(gamma_update))

            # うめこみの更新
            for p in group["params"][2:]:
                # print("p.grad:", p.grad)
                if p.grad is None:
                    continue

                if group["k"] < 0:  # hyperbolic case
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
                        - group["k"] * (
                            inner_product_k(p, h, group["k"])
                        ).unsqueeze(1)
                        * p
                    )
                    update = exp_map_k(
                        p, -group["lr_embeddings"] * proj, group["k"])
                    # print(update)
                    is_nan_inf = torch.isnan(update) | torch.isinf(update)
                    update = torch.where(is_nan_inf, p, update)
                    # update = set_dim0(update, group["R"] * 1.00)
                    update = projection_k(update, group["k"], group["R"])

                    p.data.copy_(update)

                    if group["perturbation"]:
                        r = arcosh(p[:, 0]).double()
                        r = torch.where(r <= 1e-5)
                        perturbation = torch.normal(
                            0.0, 0.0001, size=(len(r), D)).to(p.device)
                        p.data.copy_(
                            projection_k(p + perturbation, group["k"], group["R"]))

                elif group["k"] == 0:  # Euclidean case

                    grad_norm = torch.norm(p.grad.data)
                    grad_norm = torch.where(
                        grad_norm > 1, grad_norm, torch.tensor(1.0, device=p.device))
                    h = (p.grad.data / grad_norm)
                    update = p - group["lr_embeddings"] * h
                    is_nan_inf = torch.isnan(update) | torch.isinf(update)
                    update = torch.where(is_nan_inf, p, update)

                    p.data.copy_(update)

                else:
                    pass


class BaseEmbedding(nn.Module):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        R,
        k,
        init_range=0.01,
        device="cpu",
        sparse=False,
        calc_latent=True
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.R = R
        self.k = k
        self.device = device
        self.calc_latent = calc_latent

    def latent_lik(
        self,
        x
    ):
        pass

    def params_mle(
        self,
    ):
        pass

    def set_embedding(
        self,
        table
    ):
        self.table.weight.data = table

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
        dist = dist_k(us, vs, self.k)
        loss = torch.clone(labels).float()
        # 数値計算の問題をlogaddexpで回避
        # zを固定した下でのyのロス
        loss = torch.where(
            loss == 1,
            torch.logaddexp(torch.tensor([0.0]).to(
                self.device), self.beta * dist - self.gamma),
            torch.logaddexp(torch.tensor([0.0]).to(
                self.device), -self.beta * dist + self.gamma)
        )

        return loss

    def z(
        self
    ):
        z = self.table.weight.data
        lik_z = self.latent_lik(z).sum().item()

        return lik_z

    def calc_probability(
        self,
        samples,
    ):
        samples_ = torch.Tensor(samples).to(self.device).long()

        # 座標を取得
        us = self.table(samples_[:, 0])
        vs = self.table(samples_[:, 1])

        dist = dist_k(us, vs, k)
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

        dist = dist_k(us, vs, self.k)

        return dist.detach().cpu().numpy()

    def get_table(self):
        return self.table.weight.data.cpu().numpy()

    def get_PC(
        self,
        sampling=True
    ):
        pass


class Lorentz(BaseEmbedding):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        R,
        k,
        sigma,
        beta,
        gamma,
        init_range=0.01,
        sparse=True,
        device="cpu",
        calc_latent=True
    ):
        super().__init__(
            n_nodes=n_nodes,
            n_dim=n_dim,  # 次元より1つ多くデータを取る必要があることに注意
            R=R,
            k=k,
            init_range=init_range,
            sparse=sparse,
            device=device,
            calc_latent=calc_latent
        )
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.sigma = sigma.to(self.device)

        self.table = nn.Embedding(n_nodes, n_dim + 1, sparse=sparse)

        nn.init.uniform(self.table.weight, -init_range, init_range)

        # 0次元目をセット
        with torch.no_grad():
            projection_k(self.table.weight, self.k, self.R)

    def params_mle(
        self,
        sigma_min,
        sigma_max
    ):
        z = self.table.weight.data
        mu = torch.zeros((self.n_nodes, self.n_dim + 1)).to(self.device)
        mu[:, 0] = 1 / abs(self.k)

        v_ = log_map_k(z, mu, self.k)
        v = v_[:, 1:]

        self.sigma = torch.diag(torch.mm(v.T / self.n_nodes, v))
        self.sigma = torch.where(self.sigma < sigma_min, torch.tensor(
            sigma_min).to(self.device), self.sigma)
        self.sigma = torch.where(self.sigma > sigma_max, torch.tensor(
            sigma_max).to(self.device), self.sigma)
        print(self.sigma)
        print("beta:", self.beta)
        print("gamma:", self.gamma)

    def latent_lik(
        self,
        x,
    ):
        n_subnodes = x.shape[0]

        # 定数項
        # lik = torch.ones(n_subnodes).to(self.device) * ((self.n_dim / 2) * torch.log(torch.tensor(2 * np.pi).to(self.device)) +
        # 0.5 * torch.log(torch.det(self.Sigma) + 0.000001))
        lik = torch.ones(n_subnodes).to(self.device) * ((self.n_dim / 2) * torch.log(
            torch.tensor(2 * np.pi).to(self.device)) + 0.5 * torch.sum(torch.log(self.sigma)))
        # print(lik.shape)

        # データから決まる項
        mu = torch.zeros((x.shape[0], self.n_dim + 1)).to(self.device)
        mu[:, 0] = 1 / abs(self.k)
        v_ = log_map_k(x, mu, self.k)  # tangent vector
        v = v_[:, 1:]
        sigma_inv = (1 / self.sigma).reshape((-1, 1))
        # Sigma_pinv = torch.linalg.pinv(self.Sigma)  # Pseudo-inverse
        # lik += 0.5 * torch.diag(v.mm(Sigma_pinv.mm(v.T)), 0)
        # print(v.shape)
        # print((v*v).shape)
        # print(sigma_inv.shape)
        # print(((v * v).mm(sigma_inv)).shape)
        lik += 0.5 * (v * v).mm(sigma_inv)[:, 0]

        # -log Jacobian
        v_norm = tangent_norm_k(v_, self.k)
        v_norm = torch.where(
            v_norm <= 1e-6, torch.tensor(1e-6).to(self.device), v_norm)
        # print(v_norm)
        lik += (self.n_dim - 1) * (torch.log(1 - torch.exp(-2 * v_norm)) +
                                   v_norm - torch.tensor([np.log(2)]).to(self.device) - torch.log(v_norm))

        return lik

    def get_poincare_table(self):
        table = self.table.weight.data.cpu().numpy()
        return table[:, 1:] / (
            table[:, :1] + 1
        )  # diffeomorphism transform to poincare ball

    def get_PC(
        self,
        beta_min,
        beta_max,
        gamma_min,
        gamma_max,
        sigma_min,
        sigma_max,
        sampling
    ):
        if sampling == False:
            x_e = self.get_table()
        else:
            idx = np.array(range(self.n_nodes))
            idx = np.random.permutation(
                idx)[:min(int(self.n_nodes * 0.1), 100)]
            x_e = self.get_table()[idx, :]

        n_nodes_sample = len(x_e)
        print(n_nodes_sample)

        # lorentz scalar product
        first_term = - x_e[:, :1] * x_e[:, :1].T
        remaining = x_e[:, 1:].dot(x_e[:, 1:].T)
        adj_mat = - (first_term + remaining)

        for i in range(n_nodes_sample):
            adj_mat[i, i] = 1
        # distance matrix
        dist_mat = np.arccosh(adj_mat)

        is_nan_inf = np.isnan(dist_mat) | np.isinf(dist_mat)
        dist_mat = np.where(is_nan_inf, 2 * self.R, dist_mat)
        X = dist_mat
        # dist_mat
        # X = self.R - dist_mat
        for i in range(n_nodes_sample):
            X[i, i] = 0

        # I_n
        def sqrt_I_n(
            beta,
            gamma
        ):
            I_1_1 = np.sum(X**2 / ((np.cosh((beta * X - gamma) / 2.0) * 2)
                                   ** 2)) / (n_nodes_sample * (n_nodes_sample - 1))
            I_1_2 = np.sum(- X / ((np.cosh((beta * X - gamma) / 2.0) * 2)
                                  ** 2)) / (n_nodes_sample * (n_nodes_sample - 1))
            I_2_2 = 1 / ((np.cosh((beta * X - gamma) / 2.0) * 2)**2)
            for i in range(n_nodes_sample):
                I_2_2[i, i] = 0
            I_2_2 = np.sum(I_2_2) / (n_nodes_sample * (n_nodes_sample - 1))

            return np.sqrt(np.abs(I_1_1 * I_2_2 - I_1_2 * I_1_2))

        integral, _ = integrate.dblquad(sqrt_I_n, gamma_min,
                                        gamma_max, beta_min, beta_max)
        ret_1 = (np.log(self.n_nodes) + np.log(self.n_nodes - 1) - np.log(4 * np.pi)) + \
            np.log(integral)

        ret_2 = - self.n_dim * gammaln(self.n_nodes / 2) + (self.n_nodes * self.n_dim / 2) * np.log(
            self.n_nodes / np.e) + self.n_dim * (np.log(np.log(sigma_max) - np.log(sigma_min)))

        return ret_1, ret_2


def LinkPrediction(
    train_graph,
    positive_samples,
    negative_samples,
    lik_data,
    params_dataset,
    model_n_dim,
    burn_epochs,
    burn_batch_size,
    n_max_positives,
    n_max_negatives,
    lr_embeddings,
    lr_epoch_10,
    lr_beta,
    lr_gamma,
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    gamma_min,
    gamma_max,
    # eps_1,
    # eps_2,
    init_range,
    device,
    calc_lorentz=True,
    calc_euclidean=True,
    calc_spherical=True,
    calc_othermetrics=True,
    perturbation=True,
    change_learning_rate=100,
    loader_workers=16,
    shuffle=True,
    sparse=False
):

    print("model_n_dim:", model_n_dim)
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
    model_lorentz_latent = Lorentz(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        k=-1,
        sigma=torch.ones(model_n_dim) * 10,
        beta=1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )

    # 最適化関数。
    rsgd_lorentz_latent = RSGD(
        model_lorentz_latent.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        k=-1,
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )

    model_lorentz_latent.to(device)

    start = time.time()

    # change_learning_rate = 100

    for epoch in range(burn_epochs):
        if epoch == change_learning_rate:
            rsgd_lorentz_latent.param_groups[0]["lr_embeddings"] = lr_epoch_10

        losses_lorentz_latent = []

        # MLE
        if calc_lorentz:  # DNML-HGG
            print("Lorentz MLE")
            model_lorentz_latent.params_mle(
                sigma_min,
                sigma_max
            )

        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            if calc_lorentz:  # DNML-HGG
                rsgd_lorentz_latent.zero_grad()
                if epoch < change_learning_rate:
                    loss_lorentz_latent = model_lorentz_latent.lik_y_given_z(
                        pairs, labels).mean()
                else:
                    loss_lorentz_latent = model_lorentz_latent(
                        pairs, labels).mean()
                loss_lorentz_latent.backward()
                rsgd_lorentz_latent.step()
                losses_lorentz_latent.append(loss_lorentz_latent)

        print("epoch:", epoch)
        if calc_lorentz:  # DNML-HGG
            print("loss_hgg:",
                  torch.Tensor(losses_lorentz_latent).mean().item())

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # サンプリングしたデータのみで尤度を計算する。
    dataloader_all = DataLoader(
        Graph(lik_data),
        shuffle=shuffle,
        batch_size=burn_batch_size * (n_max_negatives + n_max_positives) * 10,
        num_workers=loader_workers,
        pin_memory=True
    )

    # 尤度計算
    basescore_y_given_z_lorentz = 0
    for pairs, labels in dataloader_all:
        pairs = pairs.to(device)
        labels = labels.to(device)

        basescore_y_given_z_lorentz += model_lorentz_latent.lik_y_given_z(
            pairs, labels).sum().item()

    basescore_z_lorentz = model_lorentz_latent.z()

    # the number of true data
    n_data = params_dataset['n_nodes'] * (params_dataset['n_nodes'] - 1) / 2

    basescore_y_given_z_lorentz = basescore_y_given_z_lorentz * \
        (n_data / len(lik_data))

    basescore_y_and_z_lorentz = basescore_y_given_z_lorentz + basescore_z_lorentz

    # Non-identifiable model
    # AIC_naive = basescore_y_given_z_naive + \
    #     (params_dataset['n_nodes'] * model_n_dim + 2)
    # BIC_naive = basescore_y_given_z_naive + ((params_dataset['n_nodes'] * model_n_dim + 2) / 2) * (
    # np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] -
    # 1) - np.log(2))

    # # DNML-HGG
    # pc_hgg_first, pc_hgg_second = model_hgg.get_PC(
    #     sigma_max,
    #     sigma_min,
    #     beta_min,
    #     beta_max,
    #     gamma_min,
    #     gamma_max,
    #     sampling=True
    # )
    # DNML_HGG = basescore_y_and_z_hgg + pc_hgg_first + pc_hgg_second

    # AIC_HGG = basescore_y_and_z_hgg + 3
    # BIC_HGG = basescore_y_and_z_hgg + 0.5 * (np.log(params_dataset['n_nodes']) + np.log(
    # params_dataset['n_nodes'] - 1) - np.log(2)) +
    # np.log(params_dataset['n_nodes'])

    # DNML-WND
    pc_lorentz_first, pc_lorentz_second = model_lorentz_latent.get_PC(
        beta_min,
        beta_max,
        gamma_min,
        gamma_max,
        sigma_min,
        sigma_max,
        sampling=True
    )
    DNML_lorentz = basescore_y_and_z_lorentz + pc_lorentz_first + pc_lorentz_second
    AIC_lorentz = basescore_y_and_z_lorentz + \
        model_n_dim * (model_n_dim + 1) / 2 + 2
    BIC_lorentz = basescore_y_and_z_lorentz + (np.log(params_dataset['n_nodes']) + np.log(
        params_dataset['n_nodes'] - 1) - np.log(2)) + (model_n_dim * (model_n_dim + 1) / 4) * np.log(params_dataset['n_nodes'])

    if calc_othermetrics:

        print("pos data", len(positive_samples))
        print("neg data", len(negative_samples))

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
        AUC_lorentz = calc_AUC_from_prob(
            model_lorentz_latent.calc_dist(positive_samples),
            model_lorentz_latent.calc_dist(negative_samples)
        )
    else:
        AUC_lorentz = None

    print("-log p_lorentz(y, z):", basescore_y_and_z_lorentz)
    print("-log p_lorentz(y|z):", basescore_y_given_z_lorentz)
    print("-log p_lorentz(z):", basescore_z_lorentz)
    print("pc_lorentz_first", pc_lorentz_first)
    print("pc_lorentz_second", pc_lorentz_second)
    print("DNML-lorentz:", DNML_lorentz)

    # print("AIC_naive:", AIC_naive)
    # print("BIC_naive:", BIC_naive)
    print("AUC_lorentz:", AUC_lorentz)

    ret = {
        "DNML_lorentz": DNML_lorentz,
        # "AIC_naive": AIC_naive,
        # "BIC_naive": BIC_naive,
        "AUC_lorentz": AUC_lorentz,
        # "AUC_WND": AUC_WND,
        # "AUC_naive": AUC_naive,
        # "AUC_GT": AUC_GT,
        "-log p_lorentz(y, z)": basescore_y_and_z_lorentz,
        "-log p_lorentz(y|z)": basescore_y_given_z_lorentz,
        "-log p_lorentz(z)": basescore_z_lorentz,
        # "-log p_naive(y; z)": basescore_y_given_z_naive,
        "pc_lorentz_first": pc_lorentz_first,
        "pc_lorentz_second": pc_lorentz_second,
        "model_lorentz_latent": model_lorentz_latent,
    }

    return ret

if __name__ == '__main__':
    # creating dataset
    # n_nodes = 6400
    n_nodes = 400

    print("R:", np.log(n_nodes))

    params_dataset = {
        'n_nodes': n_nodes,
        'n_dim': 4,
        'R': np.log(n_nodes),
        'sigma': 1,
        'beta': 0.4
    }

    # parameters
    burn_epochs = 800
    # burn_epochs = 5
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    lr_beta = 0.001
    lr_gamma = 0.001
    sigma_max = 100.0
    sigma_min = 0.001
    beta_min = 0.1
    beta_max = 10.0
    gamma_min = 0.1
    gamma_max = 10.0
    # eps_1 = 1e-6
    # eps_2 = 1e3
    init_range = 0.001
    # others
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

    # model_n_dims = [4, 8, 16, 32, 64]
    # model_n_dims = [16, 32, 64]
    # model_n_dims = [64]
    model_n_dims = [4]

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat=adj_mat,
        params_dataset=params_dataset
    )

    # negative samplingの比率を平均次数から決定
    pos_train_graph = len(np.where(train_graph == 1)[0])
    neg_train_graph = len(np.where(train_graph == 0)[0])
    ratio = neg_train_graph / pos_train_graph
    print("ratio:", ratio)

    n_max_negatives = int(n_max_positives * ratio)
    print("n_max_negatives:", n_max_negatives)
    lr_embeddings = 0.1
    lr_epoch_10 = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32 / 100  # batchサイズに対応して学習率変更

    for model_n_dim in model_n_dims:
        ret = LinkPrediction(
            train_graph=train_graph,
            positive_samples=positive_samples,
            negative_samples=negative_samples,
            lik_data=lik_data,
            params_dataset=params_dataset,
            model_n_dim=model_n_dim,
            burn_epochs=burn_epochs,
            burn_batch_size=burn_batch_size,
            n_max_positives=n_max_positives,
            n_max_negatives=n_max_negatives,
            lr_embeddings=lr_embeddings,
            lr_epoch_10=lr_epoch_10,
            lr_beta=lr_beta,
            lr_gamma=lr_gamma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            # eps_1=eps_1,
            # eps_2=eps_2,
            init_range=init_range,
            device=device,
            calc_lorentz=True,
            calc_euclidean=True,
            calc_spherical=True,
            calc_othermetrics=True,
            loader_workers=16,
            shuffle=True,
            sparse=False
        )

        torch.save(ret["model_lorentz_latent"],
                   "temp/result_" + str(model_n_dim) + "_lorentz_latent.pth")

        ret.pop('model_lorentz_latent')

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
        ret["lr_gamma"] = lr_gamma
        ret["sigma_max"] = sigma_max
        ret["sigma_min"] = sigma_min
        ret["beta_max"] = beta_max
        ret["beta_min"] = beta_min
        ret["gamma_max"] = gamma_max
        ret["gamma_min"] = gamma_min
        # ret["eps_1"] = eps_1
        # ret["eps_2"] = eps_2
        ret["init_range"] = init_range

        row = pd.DataFrame(ret.values(), index=ret.keys()).T

        row = row.reindex(columns=[
            "model_n_dims",
            "n_nodes",
            "n_dim",
            "R",
            "sigma",
            "beta",
            "DNML_lorentz",
            # "AIC_naive",
            # "BIC_naive",
            "AUC_lorentz",
            "-log p_lorentz(y, z)",
            "-log p_lorentz(y|z)",
            "-log p_lorentz(z)",
            # "-log p_naive(y; z)",
            "pc_lorentz_first",
            "pc_lorentz_second",
            "burn_epochs",
            "n_max_positives",
            "n_max_negatives",
            "lr_embeddings",
            "lr_epoch_10",
            "lr_beta",
            "lr_gamma",
            "sigma_max",
            "sigma_min",
            "beta_max",
            "beta_min",
            "gamma_max",
            "gamma_min",
            # "eps_1",
            # "eps_2",
            "init_range"
        ]
        )

        filepath = "result_space_selection.csv"

        if os.path.exists(filepath):
            result_previous = pd.read_csv(filepath)
            result = pd.concat([result_previous, row])
            result.to_csv(filepath, index=False)
        else:
            row.to_csv(filepath, index=False)
