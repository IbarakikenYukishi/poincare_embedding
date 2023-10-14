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
    tangent_norm_k,
    exp_map_k,
    projection_k,
    log_map_k,
    sqrt_I_n
)
from utils.utils_spherical import (
    approx_W_k,
    mle_truncated_normal,
    mle_truncated_normal_gpu,
    calc_spherical_complexity
)
from utils.utils_dataset import (
    get_unobserved,
    Graph,
    NegGraph,
    create_test_for_link_prediction,
    create_dataset_for_basescore,
    create_dataset,
    hyperbolic_geometric_graph
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
        lr_kappa,
        lr_gamma,
        R,
        k,
        k_max,
        gamma_max,
        gamma_min,
        perturbation,
        device
    ):
        defaults = {
            "lr_embeddings": lr_embeddings,
            "lr_kappa": lr_kappa,
            "lr_gamma": lr_gamma,
            "R": R,
            "k": k,
            "k_max": k_max,
            "gamma_max": gamma_max,
            "gamma_min": gamma_min,
            "perturbation": perturbation,
            "device": device
        }
        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:
            # update of beta and gamma
            # same for all models
            kappa = group["params"][0]
            gamma = group["params"][1]

            if kappa != 0:
                kappa_update = kappa.data - \
                    group["lr_kappa"] * kappa.grad.data
                if kappa > 0:
                    kappa_update = max(kappa_update, 1 / group["k_max"])
                elif kappa < 0:
                    kappa_update = min(kappa_update, - 1 / group["k_max"])
                if not math.isnan(kappa_update):
                    kappa.data.copy_(torch.tensor(kappa_update))

            gamma_update = gamma.data - \
                group["lr_gamma"] * gamma.grad.data
            gamma_update = max(gamma_update, group["gamma_min"])
            gamma_update = min(gamma_update, group["gamma_max"])
            if not math.isnan(gamma_update):
                gamma.data.copy_(torch.tensor(gamma_update))

            # update of the embedding
            for p in group["params"][2:]:
                if p.grad is None:
                    continue

                if group["k"] == 0:
                    B, D = p.size()
                    grad_norm = torch.norm(p.grad.data)
                    grad_norm = torch.where(
                        grad_norm > 1, grad_norm, torch.tensor(1.0, device=p.device))
                    h = (p.grad.data / grad_norm)
                    update = p - group["lr_embeddings"] * h

                    # print("embedding:", p)
                    # print("update:", update)
                    # print("grad:", p.grad)
                    is_nan_inf = torch.isnan(update) | torch.isinf(update)
                    update = torch.where(is_nan_inf, p, update)

                    p.data.copy_(update)

                    if group["perturbation"]:
                        # r = torch.sqrt(
                        #     dist_k(p, torch.zeros(D).to(group["device"]), k=0))
                        # r = torch.where(r <= 1e-5)
                        # perturbation = torch.normal(
                        #     0.0, 0.0001, size=(len(r), D)).to(p.device)
                        # p.data.copy_(
                        #     projection_k(p[r, :] + perturbation, group["k"], group["R"]))
                        # r = torch.sqrt(
                        #     dist_k(p, torch.zeros(D).to(group["device"]), k=0))
                        # r = torch.where(r <= 1e-5)
                        perturbation = torch.normal(
                            0.0, 0.0001, size=(len(p), D)).to(p.device)
                        p.data.copy_(
                            projection_k(p + perturbation, group["k"], group["R"]))

                else:  # hyperbolic and spherical case
                    B, D = p.size()
                    # gl = torch.eye(D, device=p.device, dtype=p.dtype)
                    gl = torch.ones(D, device=p.device, dtype=p.dtype)

                    # if group["k"] < 0:
                    #     gl[0, 0] = -1
                    if group["k"] < 0:
                        gl[0] = -1
                    grad_norm = torch.norm(p.grad.data)
                    grad_norm = torch.where(
                        grad_norm > 1, grad_norm, torch.tensor(1.0, device=p.device))
                    # normalize if and only if grad_norm is more than 1
                    h = (p.grad.data / grad_norm) * gl
                    proj = (
                        h
                        - group["k"] * (
                            inner_product_k(p, h, group["k"])
                        ).unsqueeze(1)
                        * p
                    )
                    if group["k"] > 0:
                        # update = exp_map_k(
                        # p, -group["lr_embeddings"] * 10 * proj, group["k"])
                        update = exp_map_k(
                            p, -group["lr_embeddings"] * proj, group["k"])
                    else:
                        update = exp_map_k(
                            p, -group["lr_embeddings"] * proj, group["k"])
                    is_nan_inf = torch.isnan(update) | torch.isinf(update)
                    update = torch.where(is_nan_inf, p, update)
                    update = projection_k(update, group["k"], group["R"])

                    # if group["k"] > 0:
                    #     print("embedding:", p)
                    #     print("update:", update)
                    #     print("grad:", p.grad)
                    p.data.copy_(update)

                    if group["perturbation"]:
                        # r = arcos_k(
                        #     1/np.sqrt(abs(group["k"])) * p[:, 0], group["k"]).double() / np.sqrt(abs(group["k"]))
                        # r = torch.where(r <= 1e-5)
                        perturbation = torch.normal(
                            0.0, 0.0001, size=(len(p), D)).to(p.device)
                        p.data.copy_(
                            projection_k(p + perturbation, group["k"], group["R"]))


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
        # likelihood of y given z
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

        if self.k == 0:
            # ロス計算
            dist = dist_k(us, vs, self.k)
            loss = torch.clone(labels).float()
            # 数値計算の問題をlogaddexpで回避
            # zを固定した下でのyのロス
            loss = torch.where(
                loss == 1,
                torch.logaddexp(torch.tensor([0.0]).to(
                    self.device), dist - self.gamma),
                torch.logaddexp(torch.tensor([0.0]).to(
                    self.device), -dist + self.gamma)
            )
        else:
            # ロス計算
            dist = dist_k(us, vs, self.k)
            loss = torch.clone(labels).float()
            # 数値計算の問題をlogaddexpで回避
            # zを固定した下でのyのロス
            loss = torch.where(
                loss == 1,
                torch.logaddexp(torch.tensor([0.0]).to(
                    self.device), torch.sqrt(torch.abs(self.kappa)) * dist - self.gamma),
                torch.logaddexp(torch.tensor([0.0]).to(
                    self.device), -torch.sqrt(torch.abs(self.kappa)) * dist + self.gamma)
            )

        return loss

    def z(
        self
    ):
        z = self.table.weight.data
        lik_z = self.latent_lik(z).sum().item()

        return lik_z

    def calc_dist(
        self,
        samples,
    ):
        samples_ = torch.Tensor(samples).to(self.device).long()

        # 座標を取得
        us = self.table(samples_[:, 0])
        vs = self.table(samples_[:, 1])

        if self.k == 0:
            dist = dist_k(us, vs, self.k)
        else:
            dist = 1 / torch.sqrt(torch.abs(self.kappa)) * \
                dist_k(us, vs, self.k)

        return dist.detach().cpu().numpy()

    def get_table(self):
        return self.table.weight.data.cpu().numpy()

    def get_PC(
        self,
        sampling=True
    ):
        pass


class Euclidean(BaseEmbedding):

    def __init__(
        self,
        n_nodes,
        n_dim,
        sigma,
        # kappa,
        gamma,
        init_range=0.01,
        sparse=True,
        device="cpu",
        calc_latent=True
    ):
        super().__init__(
            n_nodes=n_nodes,
            n_dim=n_dim,
            R=0,
            k=0,
            init_range=init_range,
            sparse=sparse,
            device=device,
            calc_latent=calc_latent
        )
        # 使わないがプログラムの統一感のためにダミーで入れておく
        self.kappa = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.sigma = sigma.to(self.device)

        # For the Euclidean case, the dimensionality of the ambient space is n_dim,
        # whereas that of hyperbolic and spherical space is n_dim+1.
        self.table = nn.Embedding(n_nodes, n_dim, sparse=sparse)

        nn.init.uniform(self.table.weight, -init_range, init_range)

    def params_mle(
        self,
        sigma_min,
        sigma_max
    ):
        z = self.table.weight.data
        self.sigma = torch.diag(torch.mm(z.T / self.n_nodes, z))

        self.sigma = torch.where(self.sigma < sigma_min, torch.tensor(
            sigma_min).to(self.device), self.sigma)
        self.sigma = torch.where(self.sigma > sigma_max, torch.tensor(
            sigma_max).to(self.device), self.sigma)

        print(self.sigma)
        print("kappa:", self.kappa, "k:", 1 / self.kappa)
        print("gamma:", self.gamma)

    def latent_lik(
        self,
        x,
    ):
        n_subnodes = x.shape[0]

        # 定数項
        lik = torch.ones(n_subnodes).to(self.device) * ((self.n_dim / 2) * torch.log(
            torch.tensor(2 * np.pi).to(self.device)) + 0.5 * torch.sum(torch.log(self.sigma)))

        # データから決まる項
        sigma_inv = (1 / self.sigma).reshape((-1, 1))
        lik += 0.5 * (x * x).mm(sigma_inv)[:, 0]

        return lik

    def get_PC(
        self,
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

        def distance_mat(X, Y):
            X = X[:, np.newaxis, :]
            Y = Y[np.newaxis, :, :]
            Z = np.sqrt(np.sum((X - Y) ** 2, axis=2))
            return Z

        # print(x_e)

        dist_mat = distance_mat(x_e, x_e)

        # print(dist_mat)

        is_nan_inf = np.isnan(dist_mat) | np.isinf(dist_mat)
        dist_mat = np.where(is_nan_inf, 2 * self.R, dist_mat)
        X = dist_mat
        for i in range(n_nodes_sample):
            X[i, i] = 0

        sqrt_I_n_ = partial(sqrt_I_n, X=X, n_nodes_sample=n_nodes_sample)

        # integral, _ = integrate.dblquad(sqrt_I_n_, gamma_min,
        #                                 gamma_max, beta_min, beta_max)
        integral, _ = integrate.quad(sqrt_I_n_, gamma_min, gamma_max)
        ret_1 = (np.log(self.n_nodes) + np.log(self.n_nodes - 1) - np.log(4 * np.pi)) + \
            np.log(integral)

        # ret_1 = 0

        ret_2 = - self.n_dim * gammaln(self.n_nodes / 2) + (self.n_nodes * self.n_dim / 2) * np.log(
            self.n_nodes / (2 * np.e)) + self.n_dim * (np.log(np.log(sigma_max) - np.log(sigma_min)))

        return ret_1, ret_2


class Lorentz(BaseEmbedding):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        R,
        # k,
        sigma,
        kappa,
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
            k=-1,
            init_range=init_range,
            sparse=sparse,
            device=device,
            calc_latent=calc_latent
        )
        self.kappa = nn.Parameter(torch.tensor(kappa))
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
        mu[:, 0] = 1 / np.sqrt(abs(self.k))

        v_ = log_map_k(z, mu, self.k)
        v = v_[:, 1:]

        self.sigma = torch.diag(torch.mm(v.T / self.n_nodes, v))
        self.sigma = torch.where(self.sigma < sigma_min, torch.tensor(
            sigma_min).to(self.device), self.sigma)
        self.sigma = torch.where(self.sigma > sigma_max, torch.tensor(
            sigma_max).to(self.device), self.sigma)
        print(self.sigma)
        print("kappa:", self.kappa)
        print("gamma:", self.gamma)

    def latent_lik(
        self,
        x,
    ):
        n_subnodes = x.shape[0]

        # 定数項
        lik = torch.ones(n_subnodes).to(self.device) * ((self.n_dim / 2) * torch.log(
            torch.tensor(2 * np.pi).to(self.device)) + 0.5 * torch.sum(torch.log(self.sigma)))
        lik += self.n_dim / 2 * torch.log(torch.abs(self.kappa))

        # データから決まる項
        mu = torch.zeros((x.shape[0], self.n_dim + 1)).to(self.device)
        mu[:, 0] = 1 / abs(self.k)
        v_ = log_map_k(x, mu, self.k)  # tangent vector
        v = v_[:, 1:]
        sigma_inv = (1 / self.sigma).reshape((-1, 1))
        lik += 0.5 * (v * v).mm(sigma_inv)[:, 0]

        # -log Jacobian
        v_norm_k = np.sqrt(abs(self.k)) * tangent_norm_k(v_, self.k)
        v_norm_k = torch.where(
            v_norm_k <= 1e-6, torch.tensor(1e-6).to(self.device), v_norm_k)
        # print(v_norm)
        lik += (self.n_dim - 1) * (torch.log(1 - torch.exp(-2 * v_norm_k)) +
                                   v_norm_k - torch.tensor([np.log(2)]).to(self.device) - torch.log(v_norm_k))

        return lik

    def get_poincare_table(self):
        table = self.table.weight.data.cpu().numpy()
        return table[:, 1:] / (
            table[:, :1] + 1
        )  # diffeomorphism transform to poincare ball

    def get_PC(
        self,
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
        for i in range(n_nodes_sample):
            X[i, i] = 0

        sqrt_I_n_ = partial(sqrt_I_n, X=X, n_nodes_sample=n_nodes_sample)

        # integral, _ = integrate.dblquad(sqrt_I_n_, gamma_min,
        #                                 gamma_max, beta_min, beta_max)
        integral, _ = integrate.quad(sqrt_I_n_, gamma_min, gamma_max)
        ret_1 = (np.log(self.n_nodes) + np.log(self.n_nodes - 1) - np.log(4 * np.pi)) + \
            np.log(integral)

        # ret_1 = 0

        ret_2 = - self.n_dim * gammaln(self.n_nodes / 2) + (self.n_nodes * self.n_dim / 2) * np.log(
            self.n_nodes / (2 * np.e)) + self.n_dim * (np.log(np.log(sigma_max) - np.log(sigma_min)))

        return ret_1, ret_2


class Spherical(BaseEmbedding):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        # R,
        # k,
        sigma,
        kappa,
        gamma,
        init_range=0.01,
        sparse=True,
        device="cpu",
        calc_latent=True
    ):
        super().__init__(
            n_nodes=n_nodes,
            n_dim=n_dim,  # 次元より1つ多くデータを取る必要があることに注意
            R=0,
            k=1,
            init_range=init_range,
            sparse=sparse,
            device=device,
            calc_latent=calc_latent
        )
        self.kappa = nn.Parameter(torch.tensor(kappa))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.sigma = sigma.to(self.device)

        self.table = nn.Embedding(n_nodes, n_dim + 1, sparse=sparse)

        nn.init.uniform(self.table.weight, -init_range, init_range)
        self.table.weight.data[:, 0] += 1 / np.sqrt(abs(self.k))

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
        mu[:, 0] = 1 / np.sqrt(abs(self.k))

        v_ = log_map_k(z, mu, self.k)
        v = v_[:, 1:]

        # MLE estimation of multivariate truncated normal
        self.sigma, _ = mle_truncated_normal_gpu(
            points=v,
            sigma_min=sigma_min,
            # sigma_min=0.001,
            sigma_max=sigma_max,
            k=self.k,
            sample_size=1000,
            n_iter=3000,
            learning_rate=0.001,
            alpha=0.9,
            device=self.device,
            sigma_init=self.sigma,
            early_stopping=50,
            # verbose=True
            verbose=False
        )
        # self.sigma = torch.tensor(sigma_mle).float().to(self.device)

        print(self.sigma)
        print("kappa:", self.kappa)
        print("gamma:", self.gamma)

    def latent_lik(
        self,
        x,
    ):
        n_subnodes = x.shape[0]

        # 定数項
        lik = torch.ones(n_subnodes).to(self.device) * ((self.n_dim / 2) * torch.log(
            torch.tensor(2 * np.pi).to(self.device)) + 0.5 * torch.sum(torch.log(self.sigma)))
        lik += self.n_dim / 2 * torch.log(torch.abs(self.kappa))

        # データから決まる項
        mu = torch.zeros((x.shape[0], self.n_dim + 1)).to(self.device)
        mu[:, 0] = 1 / np.sqrt(abs(self.k))
        v_ = log_map_k(x, mu, self.k)  # tangent vector
        v = v_[:, 1:]
        sigma_inv = (1 / self.sigma).reshape((-1, 1))
        lik += 0.5 * (v * v).mm(sigma_inv)[:, 0]
        # the normalization term is unnecessary to learn the embedding
        # lik += approx_W_k(Sigma, k, sample_size=1000)

        # -log Jacobian
        v_norm_k = np.sqrt(abs(self.k)) * tangent_norm_k(v_, self.k)
        v_norm_k = torch.where(
            v_norm_k <= 1e-6, torch.tensor(1e-6).to(self.device), v_norm_k)
        lik += torch.where(
            v_norm_k <= 1e-6, torch.tensor(0.0).to(self.device), (self.n_dim - 1) *
            (torch.log(torch.abs(sin_k(v_norm_k, self.k)) + 1e-6) - torch.log(v_norm_k)))
        # lik += (self.n_dim - 1) * \
        #     (torch.log(torch.abs(sin_k(v_norm_k, self.k)) + 1e-6) - torch.log(v_norm_k))

        return lik

    def z(
        self
    ):
        z = self.table.weight.data
        lik_z = self.latent_lik(z).sum().item()
        # add the logarithm of the normalization term when evaluating the
        # code-length
        lik_z += self.n_nodes * \
            np.log(approx_W_k(np.diag(self.sigma.cpu().numpy()),
                              k=self.k, sample_size=1000))

        return lik_z

    def get_PC(
        self,
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

        adj_mat = self.k * x_e.dot(x_e.T)

        for i in range(n_nodes_sample):
            adj_mat[i, i] = 1
        # distance matrix
        dist_mat = arcos_k(adj_mat, k=self.k,
                           use_torch=False) / np.sqrt(abs(self.k))

        is_nan_inf = np.isnan(dist_mat) | np.isinf(dist_mat)
        dist_mat = np.where(is_nan_inf, 2 * self.R, dist_mat)
        X = dist_mat
        # dist_mat
        for i in range(n_nodes_sample):
            X[i, i] = 0

        sqrt_I_n_ = partial(sqrt_I_n, X=X, n_nodes_sample=n_nodes_sample)

        # integral, _ = integrate.dblquad(sqrt_I_n_, gamma_min,
        #                                 gamma_max, beta_min, beta_max)
        integral, _ = integrate.quad(sqrt_I_n_, gamma_min, gamma_max)
        ret_1 = (np.log(self.n_nodes) + np.log(self.n_nodes - 1) - np.log(4 * np.pi)) + \
            np.log(integral)

        # ret_1 = 0

        ret_2 = - self.n_dim * gammaln(self.n_nodes / 2) + (self.n_nodes * self.n_dim / 2) * np.log(
            self.n_nodes / (2 * np.e)) + self.n_dim * (np.log(np.log(sigma_max) - np.log(sigma_min)))

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
    lr_kappa,
    lr_gamma,
    sigma_min,
    sigma_max,
    k_max,
    gamma_min,
    gamma_max,
    init_range,
    device,
    calc_lorentz_latent=True,
    calc_euclidean_latent=True,
    calc_spherical_latent=True,
    calc_lorentz_naive=True,
    calc_euclidean_naive=True,
    calc_spherical_naive=True,
    calc_othermetrics=True,
    perturbation=True,
    change_learning_rate=100,
    # change_learning_rate=10,
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
        n_dim=model_n_dim,
        R=params_dataset['R'],
        sigma=torch.ones(model_n_dim),
        kappa=-1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_euclidean_latent = Euclidean(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,
        sigma=torch.ones(model_n_dim),
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_spherical_latent = Spherical(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,
        sigma=torch.ones(model_n_dim) * sigma_min,
        # kappa=5.0,
        kappa=1.0,
        gamma=0.2,
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    # model
    model_lorentz_naive = Lorentz(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,
        R=params_dataset['R'],
        sigma=torch.ones(model_n_dim),
        kappa=-1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=False
    )
    model_euclidean_naive = Euclidean(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,
        sigma=torch.ones(model_n_dim),
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=False
    )
    model_spherical_naive = Spherical(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,
        sigma=torch.ones(model_n_dim) * sigma_min,
        kappa=1.0,
        # kappa=5.0,
        gamma=0.2,
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=False
    )

    # optimizer
    rsgd_lorentz_latent = RSGD(
        model_lorentz_latent.parameters(),
        lr_embeddings=lr_embeddings,
        lr_kappa=lr_kappa,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        k=-1,
        k_max=k_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )
    rsgd_euclidean_latent = RSGD(
        model_euclidean_latent.parameters(),
        lr_embeddings=lr_embeddings,
        lr_kappa=0,  # dummy argument
        lr_gamma=lr_gamma,
        R=0,  # dummy argument
        k=0,  # dummy argument
        k_max=k_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )
    rsgd_spherical_latent = RSGD(
        model_spherical_latent.parameters(),
        lr_embeddings=lr_embeddings,
        lr_kappa=lr_kappa,
        lr_gamma=lr_gamma,
        R=0,  # dummy argument
        k=1,
        k_max=k_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )
    rsgd_lorentz_naive = RSGD(
        model_lorentz_naive.parameters(),
        lr_embeddings=lr_embeddings,
        lr_kappa=lr_kappa,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        k=-1,
        k_max=k_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )
    rsgd_euclidean_naive = RSGD(
        model_euclidean_naive.parameters(),
        lr_embeddings=lr_embeddings,
        lr_kappa=0,  # dummy argument
        lr_gamma=lr_gamma,
        R=0,  # dummy argument
        k=0,  # dummy argument
        k_max=k_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )
    rsgd_spherical_naive = RSGD(
        model_spherical_naive.parameters(),
        lr_embeddings=lr_embeddings,
        lr_kappa=lr_kappa,
        lr_gamma=lr_gamma,
        R=0,  # dummy argument
        k=1,
        k_max=k_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )

    # move models to specific device
    model_lorentz_latent.to(device)
    model_euclidean_latent.to(device)
    model_spherical_latent.to(device)
    model_lorentz_naive.to(device)
    model_euclidean_naive.to(device)
    model_spherical_naive.to(device)

    start = time.time()

    early_stopping = 10

    # change_learning_rate = 100

    loss_lorentz_latent_best = 999999999999
    loss_euclidean_latent_best = 999999999999
    loss_spherical_latent_best = 999999999999
    loss_lorentz_naive_best = 999999999999
    loss_euclidean_naive_best = 999999999999
    loss_spherical_naive_best = 999999999999

    es_count_lorentz_latent = 0
    es_count_euclidean_latent = 0
    es_count_spherical_latent = 0
    es_count_lorentz_naive = 0
    es_count_euclidean_naive = 0
    es_count_spherical_naive = 0

    es_lorentz_latent = not calc_lorentz_latent
    es_euclidean_latent = not calc_euclidean_latent
    es_spherical_latent = not calc_spherical_latent
    es_lorentz_naive = not calc_lorentz_naive
    es_euclidean_naive = not calc_euclidean_naive
    es_spherical_naive = not calc_spherical_naive

    for epoch in range(burn_epochs):
        if epoch == change_learning_rate:
            rsgd_lorentz_latent.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_euclidean_latent.param_groups[
                0]["lr_embeddings"] = lr_epoch_10
            rsgd_spherical_latent.param_groups[
                0]["lr_embeddings"] = lr_epoch_10
            rsgd_lorentz_naive.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_euclidean_naive.param_groups[
                0]["lr_embeddings"] = lr_epoch_10
            rsgd_spherical_naive.param_groups[
                0]["lr_embeddings"] = lr_epoch_10

        losses_lorentz_latent = []
        losses_euclidean_latent = []
        losses_spherical_latent = []
        losses_lorentz_naive = []
        losses_euclidean_naive = []
        losses_spherical_naive = []

        if epoch == change_learning_rate - 1 or (epoch >= change_learning_rate and epoch % 20 == 0):
            if es_lorentz_latent and es_euclidean_latent and es_spherical_latent and es_lorentz_naive and es_euclidean_naive and es_spherical_naive:
                break
            # MLE
            if not es_lorentz_latent:  # Lorentz
                print("Lorentz MLE")
                model_lorentz_latent.params_mle(
                    sigma_min,
                    sigma_max
                )

            if not es_euclidean_latent:  # Euclidean
                print("Euclidean MLE")
                model_euclidean_latent.params_mle(
                    sigma_min,
                    sigma_max
                )

            if not es_spherical_latent:  # Spherical
                print("Spherical MLE")
                model_spherical_latent.params_mle(
                    sigma_min,
                    sigma_max
                )

        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            if not es_lorentz_latent:  # Lorentz
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

            if not es_euclidean_latent:  # Euclidean
                rsgd_euclidean_latent.zero_grad()
                if epoch < change_learning_rate:
                    loss_euclidean_latent = model_euclidean_latent.lik_y_given_z(
                        pairs, labels).mean()
                else:
                    loss_euclidean_latent = model_euclidean_latent(
                        pairs, labels).mean()
                loss_euclidean_latent.backward()
                rsgd_euclidean_latent.step()
                losses_euclidean_latent.append(loss_euclidean_latent)

            if not es_spherical_latent:  # Spherical
                rsgd_spherical_latent.zero_grad()
                if epoch < change_learning_rate:
                    loss_spherical_latent = model_spherical_latent.lik_y_given_z(
                        pairs, labels).mean()
                else:
                    loss_spherical_latent = model_spherical_latent(
                        pairs, labels).mean()
                loss_spherical_latent.backward()
                rsgd_spherical_latent.step()
                losses_spherical_latent.append(loss_spherical_latent)

            if not es_lorentz_naive:  # Lorentz
                rsgd_lorentz_naive.zero_grad()
                loss_lorentz_naive = model_lorentz_naive.lik_y_given_z(
                    pairs, labels).mean()
                loss_lorentz_naive.backward()
                rsgd_lorentz_naive.step()
                losses_lorentz_naive.append(loss_lorentz_naive)

            if not es_euclidean_naive:  # Euclidean
                rsgd_euclidean_naive.zero_grad()
                loss_euclidean_naive = model_euclidean_naive.lik_y_given_z(
                    pairs, labels).mean()
                loss_euclidean_naive.backward()
                rsgd_euclidean_naive.step()
                losses_euclidean_naive.append(loss_euclidean_naive)

            if not es_spherical_naive:  # Spherical
                rsgd_spherical_naive.zero_grad()
                loss_spherical_naive = model_spherical_naive.lik_y_given_z(
                    pairs, labels).mean()
                loss_spherical_naive.backward()
                rsgd_spherical_naive.step()
                losses_spherical_naive.append(loss_spherical_naive)

        print("epoch:", epoch)
        if not es_lorentz_latent:  # Lorentz
            loss_lorentz_latent = torch.Tensor(
                losses_lorentz_latent).mean().item()
            print("loss_lorentz_latent:", loss_lorentz_latent)
            if epoch > change_learning_rate:
                if loss_lorentz_latent < loss_lorentz_latent_best:
                    loss_lorentz_latent_best = loss_lorentz_latent
                    es_count_lorentz_latent = 0
                else:
                    es_count_lorentz_latent += 1
                print("es_count_lorentz_latent:", es_count_lorentz_latent)
                print("loss_lorentz_latent_best:", loss_lorentz_latent_best)
                if es_count_lorentz_latent > early_stopping:
                    print("early stopping for lorentz latent")
                    es_lorentz_latent = True

        if not es_euclidean_latent:  # Euclidean
            loss_euclidean_latent = torch.Tensor(
                losses_euclidean_latent).mean().item()
            print("loss_euclidean_latent:", loss_euclidean_latent)
            if epoch > change_learning_rate:
                if loss_euclidean_latent < loss_euclidean_latent_best:
                    loss_euclidean_latent_best = loss_euclidean_latent
                    es_count_euclidean_latent = 0
                else:
                    es_count_euclidean_latent += 1
                print("es_count_euclidean_latent:", es_count_euclidean_latent)
                print("loss_euclidean_latent_best:",
                      loss_euclidean_latent_best)
                if es_count_euclidean_latent > early_stopping:
                    print("early stopping for euclidean latent")
                    es_euclidean_latent = True

        if not es_spherical_latent:  # Spherical
            loss_spherical_latent = torch.Tensor(
                losses_spherical_latent).mean().item()
            print("loss_spherical_latent:", loss_spherical_latent)
            if epoch > change_learning_rate:
                if loss_spherical_latent < loss_spherical_latent_best:
                    loss_spherical_latent_best = loss_spherical_latent
                    es_count_spherical_latent = 0
                else:
                    es_count_spherical_latent += 1
                print("es_count_spherical_latent:", es_count_spherical_latent)
                print("loss_spherical_latent_best:",
                      loss_spherical_latent_best)
                if es_count_spherical_latent > early_stopping:
                    print("early stopping for spherical latent")
                    es_spherical_latent = True

        if not es_lorentz_naive:  # Lorentz
            loss_lorentz_naive = torch.Tensor(
                losses_lorentz_naive).mean().item()
            print("loss_lorentz_naive:", loss_lorentz_naive)
            if epoch > change_learning_rate:
                if loss_lorentz_naive < loss_lorentz_naive_best:
                    loss_lorentz_naive_best = loss_lorentz_naive
                    es_count_lorentz_naive = 0
                else:
                    es_count_lorentz_naive += 1
                print("es_count_lorentz_naive:", es_count_lorentz_naive)
                print("loss_lorentz_naive_best:", loss_lorentz_naive_best)
                if es_count_lorentz_naive > early_stopping:
                    print("early stopping for lorentz naive")
                    es_lorentz_naive = True

        if not es_euclidean_naive:  # Euclidean
            loss_euclidean_naive = torch.Tensor(
                losses_euclidean_naive).mean().item()
            print("loss_euclidean_naive:", loss_euclidean_naive)
            if epoch > change_learning_rate:
                if loss_euclidean_naive < loss_euclidean_naive_best:
                    loss_euclidean_naive_best = loss_euclidean_naive
                    es_count_euclidean_naive = 0
                else:
                    es_count_euclidean_naive += 1
                print("es_count_euclidean_naive:", es_count_euclidean_naive)
                print("loss_euclidean_naive_best:",
                      loss_euclidean_naive_best)
                if es_count_euclidean_naive > early_stopping:
                    print("early stopping for euclidean naive")
                    es_euclidean_naive = True

        if not es_spherical_naive:  # Spherical
            loss_spherical_naive = torch.Tensor(
                losses_spherical_naive).mean().item()
            print("loss_spherical_naive:", loss_spherical_naive)
            if epoch > change_learning_rate:
                if loss_spherical_naive < loss_spherical_naive_best:
                    loss_spherical_naive_best = loss_spherical_naive
                    es_count_spherical_naive = 0
                else:
                    es_count_spherical_naive += 1
                print("es_count_spherical_naive:", es_count_spherical_naive)
                print("loss_spherical_naive_best:",
                      loss_spherical_naive_best)
                if es_count_spherical_naive > early_stopping:
                    print("early stopping for spherical naive")
                    es_spherical_naive = True

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
    basescore_y_given_z_lorentz_latent = 0
    basescore_y_given_z_euclidean_latent = 0
    basescore_y_given_z_spherical_latent = 0
    basescore_y_given_z_lorentz_naive = 0
    basescore_y_given_z_euclidean_naive = 0
    basescore_y_given_z_spherical_naive = 0

    for pairs, labels in dataloader_all:
        pairs = pairs.to(device)
        labels = labels.to(device)

        basescore_y_given_z_lorentz_latent += model_lorentz_latent.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_euclidean_latent += model_euclidean_latent.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_spherical_latent += model_spherical_latent.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_lorentz_naive += model_lorentz_naive.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_euclidean_naive += model_euclidean_naive.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_spherical_naive += model_spherical_naive.lik_y_given_z(
            pairs, labels).sum().item()

    basescore_z_lorentz_latent = model_lorentz_latent.z()
    basescore_z_euclidean_latent = model_euclidean_latent.z()
    basescore_z_spherical_latent = model_spherical_latent.z()
    # basescore_z_lorentz_naive = model_lorentz_naive.z()
    # basescore_z_euclidean_naive = model_euclidean_naive.z()
    # basescore_z_spherical_naive = model_spherical_naive.z()

    # the number of true data
    n_data = params_dataset['n_nodes'] * (params_dataset['n_nodes'] - 1) / 2

    basescore_y_given_z_lorentz_latent = basescore_y_given_z_lorentz_latent * \
        (n_data / len(lik_data))
    basescore_y_given_z_euclidean_latent = basescore_y_given_z_euclidean_latent * \
        (n_data / len(lik_data))
    basescore_y_given_z_spherical_latent = basescore_y_given_z_spherical_latent * \
        (n_data / len(lik_data))
    basescore_y_given_z_lorentz_naive = basescore_y_given_z_lorentz_naive * \
        (n_data / len(lik_data))
    basescore_y_given_z_euclidean_naive = basescore_y_given_z_euclidean_naive * \
        (n_data / len(lik_data))
    basescore_y_given_z_spherical_naive = basescore_y_given_z_spherical_naive * \
        (n_data / len(lik_data))

    basescore_y_and_z_lorentz_latent = basescore_y_given_z_lorentz_latent + \
        basescore_z_lorentz_latent
    basescore_y_and_z_euclidean_latent = basescore_y_given_z_euclidean_latent + \
        basescore_z_euclidean_latent
    basescore_y_and_z_spherical_latent = basescore_y_given_z_spherical_latent + \
        basescore_z_spherical_latent
    # basescore_y_and_z_lorentz_naive = basescore_y_given_z_lorentz_naive + \
    #     basescore_z_lorentz_naive
    # basescore_y_and_z_euclidean_naive = basescore_y_given_z_euclidean_naive + \
    #     basescore_z_euclidean_naive
    # basescore_y_and_z_spherical_naive = basescore_y_given_z_spherical_naive + \
    #     basescore_z_spherical_naive

    # Lorentz
    # latent
    pc_lorentz_first, pc_lorentz_second = model_lorentz_latent.get_PC(
        gamma_min,
        gamma_max,
        sigma_min,
        sigma_max,
        sampling=True
    )
    DNML_lorentz_latent = basescore_y_and_z_lorentz_latent + \
        pc_lorentz_first + pc_lorentz_second
    AIC_lorentz_latent = basescore_y_and_z_lorentz_latent + model_n_dim + 1
    BIC_lorentz_latent = basescore_y_and_z_lorentz_latent + 0.5* (np.log(params_dataset["n_nodes"]) + np.log(
        params_dataset["n_nodes"] - 1) - np.log(2)) + 0.5 * model_n_dim * np.log(params_dataset["n_nodes"])
    # naive
    AIC_lorentz_naive = basescore_y_given_z_lorentz_naive + \
        params_dataset["n_nodes"] * model_n_dim + 1
    BIC_lorentz_naive = basescore_y_given_z_lorentz_naive + 0.5 * (params_dataset["n_nodes"] * model_n_dim + 1) * (
        np.log(params_dataset["n_nodes"]) + np.log(params_dataset["n_nodes"] - 1) - np.log(2))

    # Euclidean
    # latent
    pc_euclidean_first, pc_euclidean_second = model_euclidean_latent.get_PC(
        gamma_min,
        gamma_max,
        sigma_min,
        sigma_max,
        sampling=True
    )
    DNML_euclidean_latent = basescore_y_and_z_euclidean_latent + \
        pc_euclidean_first + pc_euclidean_second
    AIC_euclidean_latent = basescore_y_and_z_euclidean_latent + model_n_dim + 1
    BIC_euclidean_latent = basescore_y_and_z_euclidean_latent + 0.5 * (np.log(params_dataset["n_nodes"]) + np.log(
        params_dataset["n_nodes"] - 1) - np.log(2)) + 0.5 * model_n_dim * np.log(params_dataset["n_nodes"])
    # naive
    AIC_euclidean_naive = basescore_y_given_z_euclidean_naive + \
        params_dataset["n_nodes"] * model_n_dim + 1
    BIC_euclidean_naive = basescore_y_given_z_euclidean_naive + 0.5 * (params_dataset["n_nodes"] * model_n_dim + 1) * (
        np.log(params_dataset["n_nodes"]) + np.log(params_dataset["n_nodes"] - 1) - np.log(2))

    # Spherical
    # latent
    pc_spherical_first, pc_spherical_second = model_spherical_latent.get_PC(
        gamma_min,
        gamma_max,
        sigma_min,
        sigma_max,
        sampling=True
    )
    DNML_spherical_latent = basescore_y_and_z_spherical_latent + \
        pc_spherical_first + pc_spherical_second
    AIC_spherical_latent = basescore_y_and_z_spherical_latent + model_n_dim + 1
    BIC_spherical_latent = basescore_y_and_z_spherical_latent + 0.5 * (np.log(params_dataset["n_nodes"]) + np.log(
        params_dataset["n_nodes"] - 1) - np.log(2)) + 0.5 * model_n_dim * np.log(params_dataset["n_nodes"])
    # naive
    AIC_spherical_naive = basescore_y_given_z_spherical_naive + \
        params_dataset["n_nodes"] * model_n_dim + 1
    BIC_spherical_naive = basescore_y_given_z_spherical_naive + 0.5 * (params_dataset["n_nodes"] * model_n_dim + 1) * (
        np.log(params_dataset["n_nodes"]) + np.log(params_dataset["n_nodes"] - 1) - np.log(2))


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
        AUC_lorentz_latent = calc_AUC_from_prob(
            model_lorentz_latent.calc_dist(positive_samples),
            model_lorentz_latent.calc_dist(negative_samples)
        )
        # latentを計算したものでのAUC
        AUC_euclidean_latent = calc_AUC_from_prob(
            model_euclidean_latent.calc_dist(positive_samples),
            model_euclidean_latent.calc_dist(negative_samples)
        )
        AUC_spherical_latent = calc_AUC_from_prob(
            model_spherical_latent.calc_dist(positive_samples),
            model_spherical_latent.calc_dist(negative_samples)
        )
        # naive
        AUC_lorentz_naive = calc_AUC_from_prob(
            model_lorentz_naive.calc_dist(positive_samples),
            model_lorentz_naive.calc_dist(negative_samples)
        )
        # latentを計算したものでのAUC
        AUC_euclidean_naive = calc_AUC_from_prob(
            model_euclidean_naive.calc_dist(positive_samples),
            model_euclidean_naive.calc_dist(negative_samples)
        )
        AUC_spherical_naive = calc_AUC_from_prob(
            model_spherical_naive.calc_dist(positive_samples),
            model_spherical_naive.calc_dist(negative_samples)
        )
    else:
        AUC_lorentz_latent = None
        AUC_euclidean_latent = None
        AUC_spherical_latent = None
        AUC_lorentz_naive = None
        AUC_euclidean_naive = None
        AUC_spherical_naive = None

    print("DNML_lorentz_latent:", DNML_lorentz_latent)
    print("DNML_euclidean_latent:", DNML_euclidean_latent)
    print("DNML_spherical_latent:", DNML_spherical_latent)
    print("AIC_lorentz_latent:", AIC_lorentz_latent)
    print("AIC_euclidean_latent:", AIC_euclidean_latent)
    print("AIC_spherical_latent:", AIC_spherical_latent)
    print("BIC_lorentz_latent:", BIC_lorentz_latent)
    print("BIC_euclidean_latent:", BIC_euclidean_latent)
    print("BIC_spherical_latent:", BIC_spherical_latent)
    print("AIC_lorentz_naive:", AIC_lorentz_naive)
    print("AIC_euclidean_naive:", AIC_euclidean_naive)
    print("AIC_spherical_naive:", AIC_spherical_naive)
    print("BIC_lorentz_naive:", BIC_lorentz_naive)
    print("BIC_euclidean_naive:", BIC_euclidean_naive)
    print("BIC_spherical_naive:", BIC_spherical_naive)
    print("-log p_lorentz_latent(y, z):", basescore_y_and_z_lorentz_latent)
    print("-log p_lorentz_latent(y|z):", basescore_y_given_z_lorentz_latent)
    print("-log p_lorentz_latent(z):", basescore_z_lorentz_latent)
    print("-log p_lorentz_naive(y|z):", basescore_y_given_z_lorentz_naive)
    print("pc_lorentz_first", pc_lorentz_first)
    print("pc_lorentz_second", pc_lorentz_second)
    print("-log p_euclidean_latent(y, z):", basescore_y_and_z_euclidean_latent)
    print("-log p_euclidean_latent(y|z):", basescore_y_given_z_euclidean_latent)
    print("-log p_euclidean_latent(z):", basescore_z_euclidean_latent)
    print("-log p_euclidean_naive(y|z):", basescore_y_given_z_euclidean_naive)
    print("pc_euclidean_first", pc_euclidean_first)
    print("pc_euclidean_second", pc_euclidean_second)
    print("-log p_spherical_latent(y, z):", basescore_y_and_z_spherical_latent)
    print("-log p_spherical_latent(y|z):", basescore_y_given_z_spherical_latent)
    print("-log p_spherical_latent(z):", basescore_z_spherical_latent)
    print("-log p_spherical_naive(y|z):", basescore_y_given_z_spherical_naive)
    print("pc_spherical_first", pc_spherical_first)
    print("pc_spherical_second", pc_spherical_second)
    print("AUC_lorentz_latent:", AUC_lorentz_latent)
    print("AUC_euclidean_latent:", AUC_euclidean_latent)
    print("AUC_spherical_latent:", AUC_spherical_latent)
    print("AUC_lorentz_naive:", AUC_lorentz_naive)
    print("AUC_euclidean_naive:", AUC_euclidean_naive)
    print("AUC_spherical_naive:", AUC_spherical_naive)

    ret = {
        "DNML_lorentz_latent": DNML_lorentz_latent,
        "DNML_euclidean_latent": DNML_euclidean_latent,
        "DNML_spherical_latent": DNML_spherical_latent,
        "AIC_lorentz_latent": AIC_lorentz_latent,
        "AIC_euclidean_latent": AIC_euclidean_latent,
        "AIC_spherical_latent": AIC_spherical_latent,
        "AIC_lorentz_naive": AIC_lorentz_naive,
        "AIC_euclidean_naive": AIC_euclidean_naive,
        "AIC_spherical_naive": AIC_spherical_naive,
        "BIC_lorentz_latent": BIC_lorentz_latent,
        "BIC_euclidean_latent": BIC_euclidean_latent,
        "BIC_spherical_latent": BIC_spherical_latent,
        "BIC_lorentz_naive": BIC_lorentz_naive,
        "BIC_euclidean_naive": BIC_euclidean_naive,
        "BIC_spherical_naive": BIC_spherical_naive,
        "AUC_lorentz_latent": AUC_lorentz_latent,
        "AUC_euclidean_latent": AUC_euclidean_latent,
        "AUC_spherical_latent": AUC_spherical_latent,
        "AUC_lorentz_naive": AUC_lorentz_naive,
        "AUC_euclidean_naive": AUC_euclidean_naive,
        "AUC_spherical_naive": AUC_spherical_naive,
        "-log p_lorentz_latent(y, z)": basescore_y_and_z_lorentz_latent,
        "-log p_lorentz_latent(y|z)": basescore_y_given_z_lorentz_latent,
        "-log p_lorentz_latent(z)": basescore_z_lorentz_latent,
        "-log p_lorentz_naive(y|z)": basescore_y_given_z_lorentz_naive,
        "-log p_euclidean_latent(y, z)": basescore_y_and_z_euclidean_latent,
        "-log p_euclidean_latent(y|z)": basescore_y_given_z_euclidean_latent,
        "-log p_euclidean_latent(z)": basescore_z_euclidean_latent,
        "-log p_euclidean_naive(y|z)": basescore_y_given_z_euclidean_naive,
        "-log p_spherical_latent(y, z)": basescore_y_and_z_spherical_latent,
        "-log p_spherical_latent(y|z)": basescore_y_given_z_spherical_latent,
        "-log p_spherical_latent(z)": basescore_z_spherical_latent,
        "-log p_spherical_naive(y|z)": basescore_y_given_z_spherical_naive,
        "pc_lorentz_first": pc_lorentz_first,
        "pc_lorentz_second": pc_lorentz_second,
        "pc_euclidean_first": pc_euclidean_first,
        "pc_euclidean_second": pc_euclidean_second,
        "pc_spherical_first": pc_spherical_first,
        "pc_spherical_second": pc_spherical_second,
        "model_lorentz_latent": model_lorentz_latent,
        "model_euclidean_latent": model_euclidean_latent,
        "model_spherical_latent": model_spherical_latent,
        "model_lorentz_naive": model_lorentz_naive,
        "model_euclidean_naive": model_euclidean_naive,
        "model_spherical_naive": model_spherical_naive
    }

    return ret

if __name__ == '__main__':
    # creating dataset
    n_nodes = 400
    # n_nodes = 400

    print("R:", np.log(n_nodes))

    params_dataset = {
        'n_nodes': n_nodes,
        'n_dim': 4,
        'R': np.log(n_nodes) + 2,
        'sigma': 2,
        'beta': 0.4
    }

    # parameters
    burn_epochs = 800
    # burn_epochs = 20
    # burn_epochs = 5
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    lr_kappa = 1
    # lr_kappa = 0.01
    lr_gamma = 0.01
    sigma_max = 100.0
    sigma_min = 0.01
    k_max = 100.0
    gamma_min = 0.1
    gamma_max = 10.0
    init_range = 0.001
    change_learning_rate = 10
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
    print(adj_mat)

    result = pd.DataFrame()

    # model_n_dims = [64]
    # model_n_dims = [32]
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
            lr_kappa=lr_kappa,
            lr_gamma=lr_gamma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            k_max=k_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            change_learning_rate=change_learning_rate,
            init_range=init_range,
            device=device,
            calc_lorentz_latent=True,
            calc_euclidean_latent=True,
            calc_spherical_latent=True,
            calc_lorentz_naive=True,
            calc_euclidean_naive=True,
            calc_spherical_naive=True,
            calc_othermetrics=True,
            loader_workers=16,
            shuffle=True,
            sparse=False
        )

        torch.save(ret["model_lorentz_latent"],
                   "temp/result_" + str(model_n_dim) + "_lorentz_latent.pth")
        torch.save(ret["model_euclidean_latent"],
                   "temp/result_" + str(model_n_dim) + "_euclidean_latent.pth")
        torch.save(ret["model_spherical_latent"],
                   "temp/result_" + str(model_n_dim) + "_spherical_latent.pth")
        torch.save(ret["model_lorentz_naive"],
                   "temp/result_" + str(model_n_dim) + "_lorentz_naive.pth")
        torch.save(ret["model_euclidean_naive"],
                   "temp/result_" + str(model_n_dim) + "_euclidean_naive.pth")
        torch.save(ret["model_spherical_naive"],
                   "temp/result_" + str(model_n_dim) + "_spherical_naive.pth")

        ret.pop('model_lorentz_latent')
        ret.pop('model_euclidean_latent')
        ret.pop('model_spherical_latent')
        ret.pop('model_lorentz_naive')
        ret.pop('model_euclidean_naive')
        ret.pop('model_spherical_naive')

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
        ret["lr_kappa"] = lr_kappa
        ret["lr_gamma"] = lr_gamma
        ret["sigma_max"] = sigma_max
        ret["sigma_min"] = sigma_min
        ret["k_max"] = k_max
        ret["gamma_max"] = gamma_max
        ret["gamma_min"] = gamma_min
        ret["init_range"] = init_range

        row = pd.DataFrame(ret.values(), index=ret.keys()).T

        row = row.reindex(columns=[
            "model_n_dims",
            "n_nodes",
            "n_dim",
            "R",
            "sigma",
            "beta",
            "DNML_lorentz_latent",
            "DNML_euclidean_latent",
            "DNML_spherical_latent",
            "AIC_lorentz_latent",
            "AIC_euclidean_latent",
            "AIC_spherical_latent",
            "AIC_lorentz_naive",
            "AIC_euclidean_naive",
            "AIC_spherical_naive",
            "BIC_lorentz_latent",
            "BIC_euclidean_latent",
            "BIC_spherical_latent",
            "BIC_lorentz_naive",
            "BIC_euclidean_naive",
            "BIC_spherical_naive",
            "AUC_lorentz_latent",
            "AUC_euclidean_latent",
            "AUC_spherical_latent",
            "AUC_lorentz_naive",
            "AUC_euclidean_naive",
            "AUC_spherical_naive",
            "-log p_lorentz_latent(y, z)",
            "-log p_lorentz_latent(y|z)",
            "-log p_lorentz_latent(z)",
            "-log p_lorentz_naive(y|z)",
            "-log p_euclidean_latent(y, z)",
            "-log p_euclidean_latent(y|z)",
            "-log p_euclidean_latent(z)",
            "-log p_euclidean_naive(y|z)",
            "-log p_spherical_latent(y, z)",
            "-log p_spherical_latent(y|z)",
            "-log p_spherical_latent(z)",
            "-log p_spherical_naive(y|z)",
            "pc_lorentz_first",
            "pc_lorentz_second",
            "pc_euclidean_first",
            "pc_euclidean_second",
            "pc_spherical_first",
            "pc_spherical_second",
            "burn_epochs",
            "n_max_positives",
            "n_max_negatives",
            "lr_embeddings",
            "lr_epoch_10",
            "lr_kappa",
            "lr_gamma",
            "sigma_max",
            "sigma_min",
            "k_max",
            "gamma_max",
            "gamma_min",
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
