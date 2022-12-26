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
from datasets import hyperbolic_geometric_graph, connection_prob, wrapped_normal_distribution, euclidean_geometric_graph
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
from scipy import stats, special

np.random.seed(0)

import matplotlib

matplotlib.use("Agg")  # this needs to come before other matplotlib imports
import matplotlib.pyplot as plt

plt.style.use("ggplot")

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
    multigamma_ln
)
from utils.utils_dataset import (
    get_unobserved,
    Graph,
    NegGraph,
    create_test_for_link_prediction,
    create_dataset_for_basescore,
    create_dataset
)
from lorentz import(
    RSGD_WND,
    WrappedNormal
)


class SGD_Gaussian(optim.Optimizer):
    """
    Stochastic Gradient Descentを行う関数。
    """

    def __init__(
        self,
        params,
        lr_embeddings,
        lr_beta,
        lr_gamma,
        R,
        beta_max,
        beta_min,
        gamma_max,
        gamma_min,
        device
    ):
        defaults = {
            "lr_embeddings": lr_embeddings,
            "lr_beta": lr_beta,
            "lr_gamma": lr_gamma,
            'R': R,
            "beta_max": beta_max,
            "beta_min": beta_min,
            "gamma_max": gamma_max,
            "gamma_min": gamma_min,
            "device": device
        }
        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:

            # betaとsigmaの更新
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

                grad_norm = torch.norm(p.grad.data)
                grad_norm = torch.where(
                    grad_norm > 1, grad_norm, torch.tensor(1.0, device=p.device))
                h = (p.grad.data / grad_norm)
                update = p - group["lr_embeddings"] * h
                is_nan_inf = torch.isnan(update) | torch.isinf(update)
                update = torch.where(is_nan_inf, p, update)

                p.data.copy_(update)


class Euclidean(nn.Module):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        R,
        init_range=0.01,
        sparse=True,
        device="cpu",
        calc_latent=True
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_dim = n_dim
        self.R = R
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
        pass

    def z(
        self
    ):
        z = self.table.weight.data
        lik_z = self.latent_lik(z).sum().item()
        return lik_z

    def get_euclidean_table(self):
        return self.table.weight.data.cpu().numpy()

    def calc_probability(
        self,
        samples,
    ):
        samples_ = torch.Tensor(samples).to(self.device).long()

        # 座標を取得
        us = self.table(samples_[:, 0])
        vs = self.table(samples_[:, 1])

        dist = e_dist(us, vs)
        p = torch.exp(-torch.logaddexp(torch.tensor([0.0]).to(
            self.device), self.beta * (dist - self.R)))
        print(p)

        return p.detach().cpu().numpy()

    def get_PC(
        self,
        sampling=True
    ):
        pass


class Gaussian(Euclidean):

    def __init__(
        self,
        n_nodes,
        n_dim,  # 次元より1つ多くデータを取る必要があることに注意
        R,
        Sigma,
        beta,
        gamma,
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
            init_range=init_range,
            sparse=sparse,
            device=device,
            calc_latent=calc_latent
        )
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.Sigma = Sigma.to(self.device)
        self.eps_1 = eps_1

        self.table = nn.Embedding(n_nodes, n_dim, sparse=sparse)
        nn.init.normal(self.table.weight, 0, init_range)

    def params_mle(
        self
    ):
        z = self.table.weight.data
        self.Sigma = torch.mm(z.T / self.n_nodes, z)
        print(self.Sigma)
        print("beta:", self.beta)
        print("gamma:", self.gamma)

    def latent_lik(
        self,
        x,
    ):
        n_subnodes = x.shape[0]

        # 定数項
        lik = torch.ones(n_subnodes).to(self.device) * ((self.n_dim / 2) * torch.log(torch.tensor(2 * np.pi).to(self.device)) +
                                                        0.5 * torch.log(torch.det(self.Sigma) + 0.000001))

        # データから決まる項
        Sigma_pinv = torch.linalg.pinv(self.Sigma)  # Pseudo-inverse
        lik += 0.5 * torch.diag(x.mm(Sigma_pinv.mm(x.T)), 0)

        return lik

    def lik_y_given_z(
        self,
        pairs,
        labels
    ):
        # 座標を取得
        us = self.table(pairs[:, 0])
        vs = self.table(pairs[:, 1])

        # ロス計算
        dist = e_dist(us, vs)
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

    def calc_dist(
        self,
        samples,
    ):
        samples_ = torch.Tensor(samples).to(self.device).long()

        # 座標を取得
        us = self.table(samples_[:, 0])
        vs = self.table(samples_[:, 1])

        dist = e_dist(us, vs)

        return dist.detach().cpu().numpy()

    def get_PC(
        self,
        beta_min,
        beta_max,
        gamma_min,
        gamma_max,
        sampling
    ):
        if sampling == False:
            # DNMLのPCの計算
            x_e = self.get_euclidean_table()
        else:
            idx = np.array(range(self.n_nodes))
            idx = np.random.permutation(
                idx)[:min(int(self.n_nodes * 0.1), 100)]
            x_e = self.get_euclidean_table()[idx, :]

        n_nodes_sample = len(x_e)
        print(n_nodes_sample)

        # lorentz scalar product
        # first_term = - x_e[:, :1] * x_e[:, :1].T
        # remaining = x_e[:, 1:].dot(x_e[:, 1:].T)
        # adj_mat = - (first_term + remaining)

        def distance_mat(X, Y):
            X = X[:, np.newaxis, :]
            Y = Y[np.newaxis, :, :]
            Z = np.sqrt(np.sum((X - Y) ** 2, axis=2))
            return Z

        print(x_e)

        dist_mat = distance_mat(x_e, x_e)

        print(dist_mat)
        # for i in range(n_nodes_sample):
        #     adj_mat[i, i] = 1
        # distance matrix
        # dist_mat = np.arccosh(adj_mat)

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

        ret_1 = 0.5 * (np.log(self.n_nodes) + np.log(self.n_nodes - 1) - np.log(4 * np.pi)) + \
            np.log(integrate.dblquad(sqrt_I_n, gamma_min,
                                     gamma_max, beta_min, beta_max)[0])

        ret_2 = 0
        ret_2 += self.n_dim * np.log(2 / (self.n_dim - 1))
        ret_2 += (1 - self.n_dim) * self.n_dim * np.log(self.eps_1) / 2
        ret_2 += (self.n_nodes * self.n_dim / 2) * np.log(self.n_nodes /
                                                          (2 * np.e)) - multigamma_ln(self.n_dim / 2, self.n_dim)

        return ret_1, ret_2


def Euclidean_vs_Hyperbolic(
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
    lr_gamma,
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    gamma_min,
    gamma_max,
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
    model_wnd = WrappedNormal(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        Sigma=torch.eye(model_n_dim),
        beta=1.0,
        gamma=params_dataset['R'],
        eps_1=eps_1,
        # init_range=0.001,
        init_range=10,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_gaussian = Gaussian(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        Sigma=torch.eye(model_n_dim),
        beta=1.0,
        gamma=params_dataset['R'],
        eps_1=eps_1,
        # init_range=0.001,
        init_range=10,
        sparse=sparse,
        device=device,
        calc_latent=True
    )

    # 最適化関数。
    rsgd_wnd = RSGD_WND(
        model_wnd.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'] * 2,
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        device=device
    )
    sgd_gaussian = SGD_Gaussian(
        model_gaussian.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'] * 2,
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        device=device
    )

    model_wnd.to(device)
    model_gaussian.to(device)

    start = time.time()

    for epoch in range(burn_epochs):
        # if epoch != 0 and epoch % 30 == 0:  # 10 epochごとに学習率を減少
        #     rsgd.param_groups[0]["lr_embeddings"] /= 5
        if epoch == 10:
            # batchサイズに対応して学習率変更
            rsgd_wnd.param_groups[0]["lr_embeddings"] = lr_epoch_10
            sgd_gaussian.param_groups[0]["lr_embeddings"] = lr_epoch_10 * 1000

        losses_wnd = []
        losses_gaussian = []

        # MLE
        model_wnd.params_mle()
        model_gaussian.params_mle()

        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            # DNML-WND
            rsgd_wnd.zero_grad()
            loss_wnd = model_wnd(pairs, labels).mean()
            loss_wnd.backward()
            rsgd_wnd.step()
            losses_wnd.append(loss_wnd)

            # DNML-Gaussian
            sgd_gaussian.zero_grad()
            loss_gaussian = model_gaussian(pairs, labels).mean()
            loss_gaussian.backward()
            sgd_gaussian.step()
            losses_gaussian.append(loss_gaussian)

        print("epoch:", epoch, ", loss_wnd:",
              torch.Tensor(losses_wnd).mean().item())
        print("epoch:", epoch, "loss_gaussian:",
              torch.Tensor(losses_gaussian).mean().item())

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
    basescore_y_given_z_wnd = 0
    basescore_y_given_z_gaussian = 0
    for pairs, labels in dataloader_all:
        pairs = pairs.to(device)
        labels = labels.to(device)

        basescore_y_given_z_wnd += model_wnd.lik_y_given_z(
            pairs, labels).sum().item()
        basescore_y_given_z_gaussian += model_gaussian.lik_y_given_z(
            pairs, labels).sum().item()

    basescore_z_wnd = model_wnd.z()
    basescore_z_gaussian = model_gaussian.z()

    basescore_y_given_z_wnd = basescore_y_given_z_wnd * \
        (n_data / len(lik_data)) / 2
    basescore_y_given_z_gaussian = basescore_y_given_z_gaussian * \
        (n_data / len(lik_data)) / 2

    basescore_y_and_z_wnd = basescore_y_given_z_wnd + basescore_z_wnd
    basescore_y_and_z_gaussian = basescore_y_given_z_gaussian + basescore_z_gaussian

    # # Non-identifiable model
    # AIC_naive = basescore_y_given_z_naive + \
    #     (params_dataset['n_nodes'] * model_n_dim + 1)
    # BIC_naive = basescore_y_given_z_naive + ((params_dataset['n_nodes'] * model_n_dim + 1) / 2) * (
    # np.log(params_dataset['n_nodes']) + np.log(params_dataset['n_nodes'] -
    # 1) - np.log(2))

    # DNML-WND
    pc_wnd_first, pc_wnd_second = model_wnd.get_PC(
        beta_min,
        beta_max,
        gamma_min,
        gamma_max,
        sampling=True
    )
    DNML_WND = basescore_y_and_z_wnd + pc_wnd_first + pc_wnd_second
    AIC_WND = basescore_y_and_z_wnd + model_n_dim * (model_n_dim + 1) / 2 + 1
    BIC_WND = basescore_y_and_z_wnd + 0.5 * (np.log(params_dataset['n_nodes']) + np.log(
        params_dataset['n_nodes'] - 1) - np.log(2)) + (model_n_dim * (model_n_dim + 1) / 4) * np.log(params_dataset['n_nodes'])

    pc_gaussian_first, pc_gaussian_second = model_gaussian.get_PC(
        beta_min,
        beta_max,
        gamma_min,
        gamma_max,
        sampling=True
    )
    DNML_Gaussian = basescore_y_and_z_gaussian + \
        pc_gaussian_first + pc_gaussian_second
    AIC_Gaussian = basescore_y_and_z_gaussian + \
        model_n_dim * (model_n_dim + 1) / 2 + 1
    BIC_Gaussian = basescore_y_and_z_gaussian + 0.5 * (np.log(params_dataset['n_nodes']) + np.log(
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
    AUC_WND = calc_AUC_from_prob(
        model_wnd.calc_dist(positive_samples),
        model_wnd.calc_dist(negative_samples)
    )

    AUC_Gaussian = calc_AUC_from_prob(
        model_gaussian.calc_dist(positive_samples),
        model_gaussian.calc_dist(negative_samples)
    )

    # if calc_groundtruth:

    #     # 真の座標でのAUC
    #     # 座標を取得
    #     us = torch.Tensor(x_lorentz[positive_samples[:, 0], :])
    #     vs = torch.Tensor(x_lorentz[positive_samples[:, 1], :])

    #     dist = h_dist(us, vs)
    #     # p_positive = torch.exp(-torch.logaddexp(torch.tensor([0.0]), params_dataset["beta"] * (dist - params_dataset["R"]))).detach().cpu().numpy()
    #     p_positive = -dist.detach().cpu().numpy()

    #     # 座標を取得
    #     us = torch.Tensor(x_lorentz[negative_samples[:, 0], :])
    #     vs = torch.Tensor(x_lorentz[negative_samples[:, 1], :])

    #     dist = h_dist(us, vs)
    #     # p_negative = torch.exp(-torch.logaddexp(torch.tensor([0.0]), params_dataset["beta"] * (dist - params_dataset["R"]))).detach().cpu().numpy()
    #     p_negative = -dist.detach().cpu().numpy()

    #     pred_g = np.append(p_positive, p_negative)
    #     ground_truth = np.append(np.ones(len(p_positive)),
    #                              np.zeros(len(p_negative)))

    #     AUC_GT = metrics.roc_auc_score(ground_truth, pred_g)
    #     print("AUC_GT:", AUC_GT)
    #     gt_r = torch.Tensor(x_lorentz[:, 0])
    #     gt_r = torch.max(gt_r, torch.Tensor([1.0 + 0.00001]))

    #     es_r_hgg = torch.Tensor(model_hgg.get_lorentz_table()[:, 0])
    #     es_r_wnd = torch.Tensor(model_wnd.get_lorentz_table()[:, 0])
    #     es_r_naive = torch.Tensor(model_naive.get_lorentz_table()[:, 0])
    #     # es_r = torch.max(es_r, torch.Tensor([1.0 + 0.00001]))
    #     # es_r = torch.where(es_r <= 1.0+0.00001, torch.Tensor([1.0+0.0001]), es_r)[:, 0]

    #     print(gt_r)
    #     print(es_r_hgg)
    #     print(es_r_wnd)
    #     print(es_r_naive)

    #     gt_r = arcosh(gt_r)
    #     es_r_hgg = arcosh(es_r_hgg)
    #     es_r_wnd = arcosh(es_r_wnd)
    #     es_r_naive = arcosh(es_r_naive)

    #     cor_hgg, _ = stats.spearmanr(gt_r, es_r_hgg)
    #     cor_wnd, _ = stats.spearmanr(gt_r, es_r_wnd)
    #     cor_naive, _ = stats.spearmanr(gt_r, es_r_naive)
    #     print("cor_hgg:", cor_hgg)
    #     print("cor_wnd:", cor_wnd)
    #     print("cor_naive:", cor_naive)

    # else:
    #     AUC_GT = None
    #     cor_hgg = None
    #     cor_wnd = None
    #     cor_naive = None

    print("-log p_WND(y, z):", basescore_y_and_z_wnd)
    print("-log p_Gaussian(y, z):", basescore_y_and_z_gaussian)
    print("-log p_WND(y|z):", basescore_y_given_z_wnd)
    print("-log p_Gaussian(y|z):", basescore_y_given_z_gaussian)
    print("-log p_WND(z):", basescore_z_wnd)
    print("-log p_Gaussian(z):", basescore_z_gaussian)
    # print("pc_hgg_first", pc_hgg_first)
    # print("pc_hgg_second", pc_hgg_second)
    print("pc_wnd_first", pc_wnd_first)
    print("pc_wnd_second", pc_wnd_second)
    print("pc_gaussian_first", pc_gaussian_first)
    print("pc_gaussian_second", pc_gaussian_second)
    # print("-log p_naive(y; z):", basescore_y_given_z_naive)
    print("DNML-WND:", DNML_WND)
    print("DNML-Gaussian:", DNML_Gaussian)
    print("AIC_WND:", AIC_WND)
    print("BIC_WND:", BIC_WND)
    print("AIC_Gaussian:", AIC_Gaussian)
    print("BIC_Gaussian:", BIC_Gaussian)
    print("AUC_WND:", AUC_WND)
    print("AUC_Gaussian:", AUC_Gaussian)

    ret = {
        "DNML_WND": DNML_WND,
        "AIC_WND": AIC_WND,
        "BIC_WND": BIC_WND,
        "DNML_Gaussian": DNML_Gaussian,
        "AIC_Gaussian": AIC_Gaussian,
        "BIC_Gaussian": BIC_Gaussian,
        "AUC_WND": AUC_WND,
        "AUC_Gaussian": AUC_Gaussian,
        "-log p_WND(y, z)": basescore_y_and_z_wnd,
        "-log p_WND(y|z)": basescore_y_given_z_wnd,
        "-log p_WND(z)": basescore_z_wnd,
        "-log p_Gaussian(y, z)": basescore_y_and_z_gaussian,
        "-log p_Gaussian(y|z)": basescore_y_given_z_gaussian,
        "-log p_Gaussian(z)": basescore_z_gaussian,
        "pc_wnd_first": pc_wnd_first,
        "pc_wnd_second": pc_wnd_second,
        "pc_gaussian_first": pc_gaussian_first,
        "pc_gaussian_second": pc_gaussian_second,
        "model_wnd": model_wnd,
        "model_gaussian": model_gaussian,
    }

    return ret


if __name__ == '__main__':
    # データセット作成
    n_nodes = 400

    print("R:", np.log(n_nodes))

    params_dataset = {
        'n_nodes': n_nodes,
        'n_dim': 8,
        'R': np.log(n_nodes),
        'sigma': 0.5,
        'beta': 0.4
    }

    # パラメータ
    burn_epochs = 800
    # burn_epochs = 5
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 50)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    lr_beta = 0.001
    lr_gamma = 0.001
    sigma_max = 10.0
    sigma_min = 0.1
    beta_min = 0.1
    beta_max = 10.0
    gamma_min = 0.1
    gamma_max = 10.0
    eps_1 = 1e-6
    # それ以外
    loader_workers = 8
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 隣接行列
    adj_mat, x_lorentz = wrapped_normal_distribution(
        n_nodes=params_dataset['n_nodes'],
        n_dim=params_dataset['n_dim'],
        R=params_dataset['R'],
        Sigma=np.eye(params_dataset[
                     "n_dim"]) * ((np.log(params_dataset["n_nodes"]) * params_dataset["sigma"])**2),
        beta=params_dataset['beta']
    )

    # adj_mat, x_lorentz = euclidean_geometric_graph(
    #     n_nodes=params_dataset['n_nodes'],
    #     n_dim=params_dataset['n_dim'],
    #     R=params_dataset['R'],
    #     Sigma=np.eye(params_dataset[
    #                  "n_dim"]) * ((np.log(params_dataset["n_nodes"]) * params_dataset["sigma"])**2),
    #     beta=params_dataset['beta']
    # )

    # 平均次数が少なくなるように手で調整する用
    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    result = pd.DataFrame()

    # model_n_dims = [4, 8, 16, 32, 64]
    model_n_dims = [8]
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
        ret = Euclidean_vs_Hyperbolic(
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
            lr_gamma=lr_gamma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            eps_1=eps_1,
            device=device,
            loader_workers=16,
            shuffle=True,
            sparse=False,
            calc_groundtruth=False
        )

        # torch.save(ret["model_hgg"],
        #            "temp/result_" + str(model_n_dim) + "_hgg.pth")
        torch.save(ret["model_wnd"],
                   "temp/result_" + str(model_n_dim) + "_wnd.pth")
        torch.save(ret["model_gaussian"],
                   "temp/result_" + str(model_n_dim) + "_gaussian.pth")

        ret.pop('model_wnd')
        ret.pop('model_gaussian')

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
        ret["eps_1"] = eps_1

        row = pd.DataFrame(ret.values(), index=ret.keys()).T

        row = row.reindex(columns=[
            "model_n_dims",
            "n_nodes",
            "n_dim",
            "R",
            "sigma",
            "beta",
            "DNML_WND",
            "AIC_WND",
            "BIC_WND",
            "DNML_Gaussian",
            "AIC_Gaussian",
            "BIC_Gaussian",
            "AUC_WND",
            "AUC_Gaussian",
            "-log p_WND(y, z)",
            "-log p_WND(y|z)",
            "-log p_WND(z)",
            "-log p_Gaussian(y, z)",
            "-log p_Gaussian(y|z)",
            "-log p_Gaussian(z)",
            "pc_wnd_first",
            "pc_wnd_second",
            "pc_gaussian_first",
            "pc_gaussian_second",
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
            "eps_1"
        ]
        )

        filepath = "result_euclidean_vs_gaussian.csv"

        if os.path.exists(filepath):
            result_previous = pd.read_csv(filepath)
            result = pd.concat([result_previous, row])
            result.to_csv(filepath, index=False)
        else:
            row.to_csv(filepath, index=False)
