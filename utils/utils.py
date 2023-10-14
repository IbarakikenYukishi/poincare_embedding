import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import os
import sys
import torch
import random
import numpy as np
from torch import nn, optim, Tensor
from tqdm import trange, tqdm
from collections import Counter
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch.multiprocessing as multi
from functools import partial, lru_cache
import pandas as pd
import gc
import time
from scipy import integrate
from sklearn import metrics
import math
from scipy import stats, special
from sklearn.linear_model import LogisticRegression
from matplotlib.animation import ArtistAnimation, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, current_process
from PIL import Image

np.random.seed(0)

import matplotlib

matplotlib.use("Agg")  # this needs to come before other matplotlib imports
import matplotlib.pyplot as plt

plt.style.use("ggplot")


@lru_cache(maxsize=None)
def multigamma_ln(a, d):
    return special.multigammaln(a, d)


# def sqrt_I_n(
#     beta,
#     gamma,
#     X,
#     n_nodes_sample
# ):
#     I_1_1 = np.sum(X**2 / ((np.cosh((beta * X - gamma) / 2.0) * 2)
#                            ** 2)) / (n_nodes_sample * (n_nodes_sample - 1))
#     I_1_2 = np.sum(- X / ((np.cosh((beta * X - gamma) / 2.0) * 2)
#                           ** 2)) / (n_nodes_sample * (n_nodes_sample - 1))
#     I_2_2 = 1 / ((np.cosh((beta * X - gamma) / 2.0) * 2)**2)
#     for i in range(n_nodes_sample):
#         I_2_2[i, i] = 0
#     I_2_2 = np.sum(I_2_2) / (n_nodes_sample * (n_nodes_sample - 1))

#     return np.sqrt(np.abs(I_1_1 * I_2_2 - I_1_2 * I_1_2))

def sqrt_I_n(
    gamma,
    X,
    n_nodes_sample
):
    I_n = 1 / ((np.cosh((X - gamma) / 2.0) * 2)**2)
    for i in range(n_nodes_sample):
        I_n[i, i] = 0
    I_n = np.abs(np.sum(I_n) / (n_nodes_sample * (n_nodes_sample - 1)))

    return I_n


def sin_k(
    x,
    k,
    use_torch=True
):
    if k < 0:  # hyperbolic case
        if use_torch:
            return (torch.exp(x) - torch.exp(-x)) / 2
        else:
            return (np.exp(x) - np.exp(-x)) / 2
    else:  # spherical and Euclidean case
        if use_torch:
            return torch.sin(x)
        else:
            return np.sin(x)


def cos_k(
    x,
    k,
    use_torch=True
):
    if k < 0:  # hyperbolic case
        if use_torch:
            return (torch.exp(x) + torch.exp(-x)) / 2
        else:
            return (np.exp(x) + np.exp(-x)) / 2
    else:  # spherical and Euclidean case
        if use_torch:
            return torch.cos(x)
        else:
            return np.cos(x)


def arcos_k(
    x,
    k,
    use_torch=True
):
    if k < 0:  # hyperbolic case
        return arcosh(x, use_torch)
    else:  # spherical and Euclidean case
        if use_torch:
            return torch.arccos(x)
        else:
            return np.arccos(x)


def inner_product_k(
    x,
    y,
    k,
    use_torch=True
):
    # 内積
    # 2次元の入力を仮定している。
    # BD, BD -> B
    m = x * y

    if k < 0:  # hyperbolic case
        if use_torch:
            return m[:, 1:].sum(dim=1) - m[:, 0]
        else:
            return np.sum(m[:, 1:], axis=1) - m[:, 0]
    else:  # spherical and Euclidean case
        if use_torch:
            return m.sum(dim=1)
        else:
            return np.sum(m, axis=1)


def dist_k(
    x,
    y,
    k,
    use_torch=True
):
    if k < 0:  # hyperbolic case
        return h_dist_k(x, y, k, use_torch)
    elif k == 0:  # Euclidean case
        return e_dist(x, y, use_torch)
    else:  # spherical case
        return s_dist_k(x, y, k, use_torch)


def s_dist_k(
    u_e,
    v_e,
    k,
    use_torch=True
):
    dists_k = k * inner_product_k(u_e, v_e, k, use_torch)

    if use_torch:
        dists_k = torch.where(
            dists_k >= 1, torch.ones_like(dists_k) - 1e-6, dists_k)
        dists_k = torch.where(
            dists_k <= -1, -torch.ones_like(dists_k) + 1e-6, dists_k)
    else:
        dists_k = np.where(dists_k >= 1, np.ones_like(dists_k) - 1e-6, dists_k)
        dists_k = np.where(
            dists_k <= -1, -np.ones_like(dists_k) + 1e-6, dists_k)

    dists_k = (1 / np.sqrt(abs(k))) * arcos_k(dists_k, k, use_torch)

    return dists_k


def h_dist_k(
    u_e,
    v_e,
    k,
    use_torch=True
):
    dists_k = k * inner_product_k(u_e, v_e, k, use_torch)
    if use_torch:
        dists_k = torch.where(
            dists_k <= 1, torch.ones_like(dists_k) + 1e-6, dists_k)
    else:
        dists_k = np.where(dists_k <= 1, np.ones_like(dists_k) + 1e-6, dists_k)

    dists_k = (1 / np.sqrt(abs(k))) * arcos_k(dists_k, k, use_torch)

    return dists_k


def lorentz_scalar_product(
    x,
    y,
    use_torch=True
):
    m = x * y
    if use_torch:
        return m[:, 1:].sum(dim=1) - m[:, 0]
    else:
        return np.sum(m[:, 1:], axis=1) - m[:, 0]


def arcosh(
    x,
    use_torch=True
):
    if use_torch:
        return torch.log(x + torch.sqrt(x - 1) * torch.sqrt(x + 1))
    else:
        return np.log(x + np.sqrt(x - 1) * np.sqrt(x + 1))


def h_dist(
    u_e,
    v_e,
    use_torch=True
):
    dists = -lorentz_scalar_product(u_e, v_e, use_torch)
    if use_torch:
        dists = torch.where(dists <= 1, torch.ones_like(dists) + 1e-6, dists)
    else:
        dists = np.where(dists <= 1, np.ones_like(dists) + 1e-6, dists)

    dists = arcosh(dists, use_torch)

    return dists


def e_dist(
    u_e,
    v_e,
    use_torch=True
):
    if use_torch:
        return torch.sqrt(torch.sum((u_e - v_e)**2, dim=1))
    else:
        return np.sqrt(np.sum((u_e - v_e)**2, axis=1))


def tangent_norm(
    x,
    use_torch=True
):
    if use_torch:
        return torch.sqrt(lorentz_scalar_product(x, x, use_torch))
    else:
        return np.sqrt(lorentz_scalar_product(x, x, use_torch))


def tangent_norm_k(
    x,
    k,
    use_torch=True
):
    norm_2 = inner_product_k(x, x, k, use_torch)

    if use_torch:
        return torch.sqrt(norm_2)
    else:
        return np.sqrt(norm_2)


def exp_map_k(
    x,
    v,
    k
):
    if k == 0:  # Euclidean case
        return x + v
    else:  # hyperbolic and spherical case
        # Exponential Map
        tn = tangent_norm_k(v, k).unsqueeze(dim=1)
        tn_expand = tn.repeat(1, x.size()[-1])
        result = cos_k(np.sqrt(abs(k)) * tn, k) * x + \
            sin_k(np.sqrt(abs(k)) * tn, k) * (v / (np.sqrt(abs(k)) * tn))
        # if norm is larger than 0, then update x
        result = torch.where(tn_expand > 0, result, x)

        return result


def projection_k(
    x,
    k,
    R=0,  # is needed if hyperbolic case
):
    if k == 0:
        return x
    elif k > 0:
        return k * x / torch.norm(x, p=2, dim=1, keepdim=True)
    else:
        x[:, 1:] = torch.renorm(x[:, 1:], p=2, dim=0,
                                maxnorm=np.sinh(R) / np.sqrt(abs(k)))  # 半径Rの範囲に収めたい
        # 発散しないように気を使う。
        x_max = torch.max(torch.abs(x[:, 1:]), dim=1, keepdim=True)[0].double()
        x_max = torch.where(x_max < 1.0, 1.0, x_max)

        dim0 = x_max * torch.sqrt(-(1 / x_max)**2 / k +
                                  ((x[:, 1:] / x_max) ** 2).sum(dim=1, keepdim=True))
        x[:, 0] = dim0[:, 0]
        return x


def log_map_k(
    z,
    mu,
    k
):
    if k == 0:
        return z - mu
    else:
        alpha = k * inner_product_k(z, mu, k).reshape((-1, 1)).float()
        if k < 0:  # hyperbolic case
            alpha = torch.where(alpha > 1.0, alpha, torch.tensor(
                [1.0], dtype=alpha.dtype).to(alpha.device))
        else:  # spherical case
            alpha = torch.where(alpha > 1.0, torch.tensor(
                [1.0], dtype=alpha.dtype).to(alpha.device), alpha)
            alpha = torch.where(alpha < -1.0, torch.tensor(
                [-1.0], dtype=alpha.dtype).to(alpha.device), alpha)

        beta = arcos_k(alpha, k)

        coef = torch.where(beta < 1e-6, torch.tensor(
            [1.0], dtype=beta.dtype).to(beta.device), beta / sin_k(beta, k))

        u = coef * (z - alpha * mu)  # ゼロ除算を防ぐ
    return u


def exp_map(
    x,
    v
):
    # Exponential Map
    tn = tangent_norm(v).unsqueeze(dim=1)
    tn_expand = tn.repeat(1, x.size()[-1])
    result = torch.cosh(tn) * x + torch.sinh(tn) * (v / tn)
    result = torch.where(tn_expand > 0, result, x)
    return result


def log_map(
    z,
    mu
):
    alpha = -lorentz_scalar_product(z, mu).reshape((-1, 1)).float()
    alpha = torch.where(alpha > 1.0, alpha, torch.tensor(
        [1.0], dtype=alpha.dtype).to(alpha.device))
    u = (arcosh(alpha) / torch.sqrt(alpha**2 - 1 + 0.000001)) * \
        (z - alpha * mu)  # ゼロ除算を防ぐ
    return u


def set_dim0(x, R):
    x[:, 1:] = torch.renorm(x[:, 1:], p=2, dim=0,
                            maxnorm=np.sinh(R))  # 半径Rの範囲に収めたい
    # 発散しないように気を使う。
    x_max = torch.max(torch.abs(x[:, 1:]), dim=1, keepdim=True)[0].double()
    x_max = torch.where(x_max < 1.0, 1.0, x_max)

    dim0 = x_max * torch.sqrt((1 / x_max)**2 +
                              ((x[:, 1:] / x_max) ** 2).sum(dim=1, keepdim=True))
    x[:, 0] = dim0[:, 0]
    return x


def integral_sinh(n, n_dim, sigma, R, exp_C):  # (exp(exp_C)/2)^(D-1)で割った結果
    if n == 0:
        return R * (2 * np.exp(-exp_C))**(n_dim - 1)
    elif n == 1:
        return (1 / sigma) * (np.exp(sigma * R - exp_C) + np.exp(- sigma * R - exp_C) - 2 * np.exp(-exp_C)) * (2 * np.exp(-exp_C))**(n_dim - 2)
    else:
        ret = 1 / (sigma * n)
        ret = ret * (np.exp(sigma * R - exp_C) - np.exp(- sigma * R - exp_C)
                     )**(n - 1) * (np.exp(sigma * R - exp_C) + np.exp(- sigma * R - exp_C))
        ret = ret * (2 * np.exp(-exp_C)
                     )**(n_dim - 1 - n)
        return ret - (n - 1) / n * integral_sinh(n=n - 2, n_dim=n_dim, sigma=sigma, R=R, exp_C=exp_C)


def calc_log_C_D(n_dim, sigma, R):

    exp_C = max(1, sigma * R)  # sigma*Rが大きい時のみやる。

    log_C_D = (n_dim - 1) * exp_C - (n_dim - 1) * np.log(2)  # 支配項
    C = integral_sinh(n=n_dim - 1, n_dim=n_dim, sigma=sigma, R=R, exp_C=exp_C)
    # sigma*Rが十分小さく、次元が十分大きい時にCが負になることがあり、nanを生成するのでその対策
    # そもそもsigma*Rがそんなに小さくならないように　sigmaの範囲を制限した方がいいかもしれない
    C = max(C, 0.00000001)
    # print("C:", C)
    log_C_D = log_C_D + np.log(C)

    return log_C_D


def calc_likelihood_list(r, n_dim, R, sigma_min, sigma_max, DIV=1000):
    sigma_list = np.arange(sigma_min, sigma_max, (sigma_max - sigma_min) / DIV)

    ret = []

    for sigma_ in sigma_list:
        # rの尤度
        lik = -(n_dim - 1) * (np.log(1 - np.exp(-2 * sigma_ *
                                                r) + 0.00001) + sigma_ * r - np.log(2))

        log_C_D = calc_log_C_D(n_dim=n_dim, sigma=sigma_, R=R)

        # # rの正規化項
        # log_C_D = (n_dim - 1) * sigma_ * R - (n_dim - 1) * np.log(2)  # 支配項
        # # のこり。計算はwikipediaの再帰計算で代用してもいいかも
        # # https://en.wikipedia.org/wiki/List_of_integrals_of_hyperbolic_functions
        # C = integral_sinh(n=n_dim - 1, n_dim=n_dim, R=R, sigma=sigma_)
        # log_C_D = log_C_D + np.log(C)

        lik = lik + log_C_D
        ret.append(np.sum(lik))
        sigma_hat = float(sigma_list[np.argmin(ret)])

    return sigma_list, np.array(ret), sigma_hat


def calc_beta_hat(z, train_graph, n_samples, R, beta_min, beta_max):
    n_nodes = z.shape[0]
    idx = np.array(range(n_nodes))
    idx = np.random.permutation(idx)[:n_samples]
    # print("n_samples:", n_samples)

    z_ = z[idx, :]

    first_term = - z_[:, :1] * z_[:, :1].T

    # remaining = x_e_[:, 1:].dot(x_e_[:, 1:].T)
    remaining = torch.mm(z_[:, 1:], z_[:, 1:].T)
    adj_mat_hat = - (first_term + remaining)
    for i in range(n_samples):
        adj_mat_hat[i, i] = 1

    # 数値誤差対策
    adj_mat_hat = adj_mat_hat.double()
    adj_mat_hat = torch.where(adj_mat_hat <= 1 + 1e-6, 1 + 1e-6, adj_mat_hat)

    # distance matrix
    adj_mat_hat = arcosh(adj_mat_hat)
    for i in range(n_samples):
        adj_mat_hat[i, i] = -1

    train_graph_ = train_graph[idx][:, idx]

    for i in range(n_samples):
        train_graph_[i, :i + 1] = -1

    adj_mat_hat = adj_mat_hat.cpu().numpy()

    y = train_graph_.flatten()
    x = adj_mat_hat.flatten()

    non_empty_idx = np.where(y != -1)[0]
    y = y[non_empty_idx].reshape((-1, 1))
    x = -(x[non_empty_idx] - R).reshape((-1, 1))

    # print(y)
    # print(x)

    lr = LogisticRegression()  # ロジスティック回帰モデルのインスタンスを作成
    lr.fit(x, y)  # ロジスティック回帰モデルの重みを学習

    return max(min(lr.coef_[0, 0], beta_max), beta_min)


def plot_figure_training(adj_mat, tables_lorentz, tables_euclidean, tables_spherical, suffix):

    _adj_mat = deepcopy(adj_mat)
    for i in range(len(_adj_mat)):
        _adj_mat[i, 0:i + 1] = -1

    edges = np.array(np.where(_adj_mat == 1)).T

    def create_gif(
        tables,
        path
    ):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

        ims = []

        # print(len(poincares))

        for table in tables:
            # print(poincare)
            # plt.clf()
            plt.gca().set_xlim(-1, 1)
            plt.gca().set_ylim(-1, 1)
            plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, edgecolor="black"))

            plt_anims = []
            # for edge in edges:
            #     plt.plot(
            #         table[edge, 0],
            #         table[edge, 1],
            #         color="black",
            #         # marker="o",
            #         alpha=0.01,
            #     )

            plt_anims.append(plt.scatter(
                table[:, 0], table[:, 1], color="blue"))
            # ims.append([plt.scatter(poincare[:, 0], poincare[:, 1], color="blue")])
            ims.append(plt_anims)

        ani = ArtistAnimation(fig, ims, interval=100)
        ani.save(path, writer='pillow')

        plt.clf()

    def create_gif_3d(
        tables,
        path
    ):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        ims = []

        # print(len(poincares))

        # アニメーション更新関数
        def update(frame):
            ax.cla()  # プロットをクリア
            ax.set_xlim3d([-1, 1])  # x軸範囲を設定
            ax.set_ylim3d([-1, 1])  # y軸範囲を設定
            ax.set_zlim3d([-1, 1])  # z軸範囲を設定
            ax.set_title(f'Frame: {frame}')
            data = tables[frame]  # フレームのデータを取得
            # 3D散布図をプロット
            ax.scatter(data[:, 0], data[:, 1], data[:, 2])
            # プロットを画像に保存
            fig.canvas.draw()
            image = Image.frombytes(
                'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            return image

        # アニメーションの作成
        ani = FuncAnimation(
            fig, update, frames=len(tables), interval=100)

        # アニメーションの保存
        ani.save(path, writer='pillow')

        plt.clf()

    create_gif(tables_lorentz, suffix + "test_lorentz.gif")
    create_gif(tables_euclidean, suffix + "test_euclidean.gif")
    create_gif_3d(tables_spherical, suffix + "test_spherical.gif")


def plot_figure(adj_mat, table, path):
    # skip padding. plot x y

    print(table.shape)

    table_ = table[:, 1:] / (
        table[:, :1] + 1
    )  # diffeomorphism transform to poincare ball

    plt.figure(figsize=(7, 7))

    _adj_mat = deepcopy(adj_mat)
    for i in range(len(_adj_mat)):
        _adj_mat[i, 0:i + 1] = -1

    edges = np.array(np.where(_adj_mat == 1)).T

    # for edge in edges:
    #     plt.plot(
    #         table_[edge, 0],
    #         table_[edge, 1],
    #         color="black",
    #         # marker="o",
    #         alpha=0.5,
    #     )
    plt.scatter(table_[:, 0], table_[:, 1])
    plt.gca().set_xlim(-1, 1)
    plt.gca().set_ylim(-1, 1)
    plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, edgecolor="black"))
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    # All of the functions assumes the curvature of hyperbolic space is 1.
    # arcosh
    print("--------arcosh--------")
    x = torch.cosh(torch.Tensor([0., 2., 4., 8.]))
    print(arcosh(x))
    print(arcosh(x.numpy(), use_torch=False))

    # lorentz scalar product
    x = torch.Tensor([
        [1, 3, 0, 5],
        [-1, 2, 3, 0]
    ]
    )
    y = torch.Tensor([
        [5, 2, -1, 2],
        [8, 0, 1, 0]
    ]
    )

    print(lorentz_scalar_product(x, y))
    print(lorentz_scalar_product(x.numpy(), y.numpy(), use_torch=False))

    # h_dist
    print("--------h_dist--------")
    x = torch.Tensor([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]
    )
    y = torch.Tensor([
        [np.cosh(10), np.sinh(10), 0, 0],
        [np.cosh(10), 0, np.sinh(10), 0],
        [np.cosh(10), 0, 0, np.sinh(10)]
    ]
    )
    print(h_dist(x, y))
    print(h_dist(x.numpy(), y.numpy(), use_torch=False))

    # tangent norm
    print("--------tangent norm--------")
    x = torch.Tensor([
        [1, 2, -1, 0],
        [-1, 1, 2, 1],
        [1, -3, 1, 2]
    ]
    )
    print(tangent_norm(x))
    print(tangent_norm(x.numpy(), use_torch=False))

    # exp map
    print("--------exp map--------")
    x = torch.Tensor([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]
    )
    v = torch.Tensor([
        [0, np.sinh(1), 0, 0],
        [0, 0, np.sinh(1), 0],
        [0, 0, 0, np.sinh(1)]
    ]
    )

    print(exp_map(x, v))

    # log map
    print("--------log map--------")
    x = torch.Tensor([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]
    )
    v = torch.Tensor([
        [0, np.sinh(1), 0, 0],
        [0, 0, np.sinh(1), 0],
        [0, 0, 0, np.sinh(1)]
    ]
    )

    print(log_map(exp_map(x, v), x))

    # set_dim0
    print("--------set_dim0--------")
    x = torch.Tensor([
        [1, 2, -1, 0],
        [-1, 1, 2, 1],
        [1, -3, 1, 2]
    ]
    )
    print(set_dim0(x, R=3))

    # set_dim0
    print("--------projection_k--------")
    x = torch.Tensor([
        [1, 2, -1, 0],
        [-1, 1, 2, 1],
        [1, -3, 1, 2]
    ]
    )
    print(projection_k(x, R=3, k=-1))
    print(projection_k(x * np.sqrt(2), R=3, k=-1 / 2))
    x = torch.Tensor([
        [0.5, 0.5],
        [1, 1],
        [-1, -1],
        [4, -4]
    ]
    )
    print(projection_k(x, k=1))
    print(projection_k(x * np.sqrt(2), k=1 / 2))

    # sin_K, cos_K, arcos_k
    print("--------sin_k, cos_k, arcos_k--------")
    print(sin_k(np.pi / 2, 1, use_torch=False))
    print(cos_k(np.pi / 2, 1, use_torch=False))
    print(arcos_k(cos_k(np.pi / 2, 1, use_torch=False), 1, use_torch=False))
    print(sin_k(np.pi / 2, -1, use_torch=False))
    print(cos_k(np.pi / 2, -1, use_torch=False))
    print(arcos_k(cos_k(np.pi / 2, -1, use_torch=False), -1, use_torch=False))

    print(sin_k(torch.tensor(np.pi / 2), 1))
    print(cos_k(torch.tensor(np.pi / 2), 1))
    print(arcos_k(cos_k(torch.tensor(np.pi / 2), 1), 1))
    print(sin_k(torch.tensor(np.pi / 2), -1))
    print(cos_k(torch.tensor(np.pi / 2), -1))
    print(arcos_k(cos_k(torch.tensor(np.pi / 2), -1), -1))

    print(sin_k(np.pi / 4, 1, use_torch=False))
    print(cos_k(np.pi / 4, 1, use_torch=False))
    print(arcos_k(cos_k(np.pi / 4, 1, use_torch=False), 1, use_torch=False))
    print(sin_k(np.pi / 4, -1, use_torch=False))
    print(cos_k(np.pi / 4, -1, use_torch=False))
    print(arcos_k(cos_k(np.pi / 4, -1, use_torch=False), -1, use_torch=False))

    print(sin_k(torch.tensor(np.pi / 4), 1))
    print(cos_k(torch.tensor(np.pi / 4), 1))
    print(arcos_k(cos_k(torch.tensor(np.pi / 4), 1), 1))
    print(sin_k(torch.tensor(np.pi / 4), -1))
    print(cos_k(torch.tensor(np.pi / 4), -1))
    print(arcos_k(cos_k(torch.tensor(np.pi / 4), -1), -1))

    # dist_k
    print("--------dist_k--------")
    # spherical case
    x = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0]
    ])
    y = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [np.sqrt(2) / 2, np.sqrt(2) / 2]
    ])
    print(dist_k(x, y, k=1))
    print(dist_k(x * np.sqrt(2), y * np.sqrt(2), k=1 / 2))
    print(dist_k(x * np.sqrt(3), y * np.sqrt(3), k=1 / 3))
    print(dist_k(x.numpy(), y.numpy(), k=1, use_torch=False))
    print(dist_k(x.numpy() * np.sqrt(2), y.numpy()
                 * np.sqrt(2), k=1 / 2, use_torch=False))
    print(dist_k(x.numpy() * np.sqrt(3), y.numpy()
                 * np.sqrt(3), k=1 / 3, use_torch=False))

    # Euclidean case
    x = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0]
    ])
    y = torch.tensor([
        [0.0, 1.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [np.sqrt(2) / 2, np.sqrt(2) / 2]
    ])
    print(dist_k(x, y, k=0))
    print(dist_k(x * np.sqrt(2), y * np.sqrt(2), k=0))
    print(dist_k(x * np.sqrt(3), y * np.sqrt(3), k=0))
    print(dist_k(x.numpy(), y.numpy(), k=0, use_torch=False))
    print(dist_k(x.numpy() * np.sqrt(2), y.numpy()
                 * np.sqrt(2), k=0, use_torch=False))
    print(dist_k(x.numpy() * np.sqrt(3), y.numpy()
                 * np.sqrt(3), k=0, use_torch=False))

    # hyperbolic case
    x = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0]
    ]
    )
    y = torch.Tensor([
        [np.cosh(10.0), np.sinh(10.0), 0.0, 0.0],
        [np.cosh(10.0), 0.0, np.sinh(10.0), 0.0],
        [np.cosh(10.0), 0.0, 0.0, np.sinh(10.0)]
    ]
    )
    print(dist_k(x, y, k=-1))
    print(dist_k(x * np.sqrt(2), y * np.sqrt(2), k=-1 / 2))
    print(dist_k(x * np.sqrt(3), y * np.sqrt(3), k=-1 / 3))
    print(dist_k(x.numpy(), y.numpy(), k=-1, use_torch=False))
    print(dist_k(x.numpy() * np.sqrt(2), y.numpy()
                 * np.sqrt(2), k=-1 / 2, use_torch=False))
    print(dist_k(x.numpy() * np.sqrt(3), y.numpy()
                 * np.sqrt(3), k=-1 / 3, use_torch=False))

    # tangent norm
    print("--------tangent norm_k--------")
    x = torch.Tensor([
        [1, 2, -1, 0],
        [-1, 1, 2, 1],
        [1, -3, 1, 2]
    ]
    )
    print(tangent_norm_k(x, k=-1))
    print(tangent_norm_k(x.numpy(), k=-1, use_torch=False))
    print(tangent_norm_k(x, k=0))
    print(tangent_norm_k(x.numpy(), k=0, use_torch=False))
    print(tangent_norm_k(x, k=1))
    print(tangent_norm_k(x.numpy(), k=1, use_torch=False))

    # exp map
    print("--------exp_map_k and log_map_k--------")
    # hyperbolic case
    x = torch.Tensor([
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]
    )
    v = torch.Tensor([
        [0, np.sinh(1), 0, 0],
        [0, 0, np.sinh(1), 0],
        [0, 0, 0, np.sinh(1)]
    ]
    )
    print(exp_map_k(x, v, k=-1))
    print(exp_map_k(x * np.sqrt(2), v * np.sqrt(2), k=-1 / 2))
    print(exp_map_k(x * np.sqrt(3), v * np.sqrt(3), k=-1 / 3))
    # Euclidean case
    print(exp_map_k(x, v, k=0))
    # spherical case
    x = torch.Tensor([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ]
    )
    v = torch.Tensor([
        [0, np.pi / 2],
        [0, np.pi / 2],
        [np.pi / 2, 0],
        [np.pi / 2, 0],
        [0, np.pi / 4],
        [0, np.pi / 4],
        [np.pi / 4, 0],
        [np.pi / 4, 0],
        [0, np.pi / 2 + 2 * np.pi],
        [0, np.pi / 2 + 2 * np.pi],
        [np.pi / 2 + 2 * np.pi, 0],
        [np.pi / 2 + 2 * np.pi, 0],
        [0, np.pi / 4 + 2 * np.pi],
        [0, np.pi / 4 + 2 * np.pi],
        [np.pi / 4 + 2 * np.pi, 0],
        [np.pi / 4 + 2 * np.pi, 0]
    ]
    )
    print(exp_map_k(x, v, k=1))
    print(exp_map_k(x * np.sqrt(2), v * np.sqrt(2), k=1 / 2))
    print(exp_map_k(x * np.sqrt(3), v * np.sqrt(3), k=1 / 3))
    print(log_map_k(exp_map_k(x, v, k=1), x, k=1))
    print(log_map_k(exp_map_k(x * np.sqrt(2), v * np.sqrt(2), k=1 / 2), x, k=1 / 2))
    print(log_map_k(exp_map_k(x * np.sqrt(3), v * np.sqrt(3), k=1 / 3), x, k=1 / 3))
    x = torch.Tensor([
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, -1, 0, 0]
    ]
    )
    v = torch.Tensor([
        [0, np.pi / 2, 0, 0],
        [0, np.pi / 2, 0, 0],
        [np.pi / 2, 0, 0, 0],
        [np.pi / 2, 0, 0, 0],
        [0, np.pi / 4, 0, 0],
        [0, np.pi / 4, 0, 0],
        [np.pi / 4, 0, 0, 0],
        [np.pi / 4, 0, 0, 0],
        [0, np.pi / 2 + 2 * np.pi, 0, 0],
        [0, np.pi / 2 + 2 * np.pi, 0, 0],
        [np.pi / 2 + 2 * np.pi, 0, 0, 0],
        [np.pi / 2 + 2 * np.pi, 0, 0, 0],
        [0, np.pi / 4 + 2 * np.pi, 0, 0],
        [0, np.pi / 4 + 2 * np.pi, 0, 0],
        [np.pi / 4 + 2 * np.pi, 0, 0, 0],
        [np.pi / 4 + 2 * np.pi, 0, 0, 0]
    ]
    )
    print(exp_map_k(x, v, k=1))
    print(exp_map_k(x * np.sqrt(2), v * np.sqrt(2), k=1 / 2))
    print(exp_map_k(x * np.sqrt(3), v * np.sqrt(3), k=1 / 3))
    x = torch.Tensor([
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0]
    ]
    )
    v = torch.Tensor([
        [0, 1e-1],
        [0, 1e-2],
        [0, 1e-3],
        [0, 1e-4],
        [0, 1e-5],
        [0, 1e-6],
        [0, 1e-7],
        [0, 1e-8]
    ]
    )
    print(exp_map_k(x, v, k=1))

    # calc_log_C_D
    print("--------calc_log_C_D--------")
    # R*sigma>=1
    print(calc_log_C_D(n_dim=64, R=10, sigma=1))
    print(calc_log_C_D(n_dim=32, R=7, sigma=0.5))
    print(calc_log_C_D(n_dim=16, R=9, sigma=1.5))
    print(calc_log_C_D(n_dim=2, R=10, sigma=2))
    # R*sigma<1
    # wolfram alphaの出力などと比べると正しくない
    print(calc_log_C_D(n_dim=64, R=10, sigma=0.00001))
    print(calc_log_C_D(n_dim=32, R=7, sigma=0.00001))
    print(calc_log_C_D(n_dim=16, R=9, sigma=0.00001))
    print(calc_log_C_D(n_dim=2, R=10, sigma=0.00001))

    print(calc_log_C_D(n_dim=64, R=10, sigma=0.1))
    print(calc_log_C_D(n_dim=32, R=7, sigma=0.1))
    print(calc_log_C_D(n_dim=16, R=9, sigma=0.1))
    print(calc_log_C_D(n_dim=2, R=10, sigma=0.1))

    print(approx_W_k(np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]), 1, 1000))
    print(approx_W_k(np.array([
        [(np.pi / 2)**2],
    ]), 1, 1000000))
    print(approx_W_k(np.array([
        [(np.pi / 3)**2],
    ]), 1, 1000000))
    print(approx_W_k(np.array([
        [(np.pi / 2)**2, 0, 0],
        [0, (np.pi / 2)**2, 0],
        [0, 0, (np.pi / 2)**2]
    ]), 1, 1000))
    print(approx_W_k(np.array([
        [1, 0],
        [0, 1]
    ]), 1, 1000))
