import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate
from scipy.sparse import coo_matrix
from copy import deepcopy
from utils.utils import integral_sinh, calc_likelihood_list, arcosh, calc_beta_hat
from utils.utils_dataset import create_test_for_link_prediction, create_dataset
from sklearn.linear_model import LogisticRegression
import torch

np.random.seed(0)

INTEGRAL_DIV = 10000


def connection_prob(d, R, beta):
    """
    接続確率
    """
    return 1 / (1 + np.exp(beta * (d - R)))


def integral_sin(n, theta):
    if n == 0:
        return theta
    elif n == 1:
        return 1 - np.cos(theta)
    else:
        return -np.cos(theta) * (np.sin(theta)**(n - 1)) / n + ((n - 1) / n) * integral_sin(n - 2, theta)


def calc_dist_angle(n_dim, n, div=INTEGRAL_DIV):
    # nは1からn_dim-1となることが想定される。0次元目はr
    if n_dim - 1 == n:
        theta_array = 2 * np.pi * np.arange(0, div + 1) / div
        cum_dens = theta_array / (2 * np.pi)
    else:
        theta_array = np.pi * np.arange(0, div + 1) / div
        cum_dens = []
        for theta in theta_array:
            cum_dens.append(integral_sin(n_dim - 1 - n, theta))
        cum_dens = np.array(cum_dens) / np.max(cum_dens)
    return theta_array, cum_dens


def calc_dist_r(n_dim, sigma, R, div=INTEGRAL_DIV):
    # n_dimかRが大きくなると現状だと数値積分があまりうまくいかない。divを増やす必要がある。
    # 発散を防ぐために、exp(sigma*R*(n_dim-1))/(2**(n_dim-1))(分子の積分の支配項)で割ってある。

    def integral_sinh_(n, r):  # (exp(sigma*R)/2)^(D-1)で割った結果
        if n == 0:
            return r * (2 * np.exp(-sigma * R))**(n_dim - 1)
        elif n == 1:
            return 1 / sigma * (np.exp(sigma * (r - R)) + np.exp(- sigma * (r + R)) - 2 * np.exp(-sigma * R)) * (2 * np.exp(-sigma * r))**(n_dim - 2)
        else:
            ret = 1 / (sigma * n)
            ret = ret * (np.exp(sigma * (r - R)) - np.exp(-sigma * (r + R))
                         )**(n - 1) * (np.exp(sigma * (r - R)) + np.exp(-sigma * (r + R)))
            ret = ret * (2 * np.exp(-sigma * R)
                         )**(n_dim - 1 - n)
            return ret - (n - 1) / n * integral_sinh_(n - 2, r)
    r_array = R * np.arange(0, div + 1) / div
    cum_dens = []
    for r in r_array:
        cum_dens.append(integral_sinh_(n=n_dim - 1, r=r))
    cum_dens = np.array(cum_dens) / np.max(cum_dens)
    return r_array, cum_dens


def hyperbolic_geometric_graph(n_nodes, n_dim, R, sigma, beta):
    # TODO: プログラム前半部分も実行時間を短くする。
    # 現状は次元の2乗オーダーの計算量
    # n_dimは2以上で
    x_polar = np.random.uniform(0, 1, (n_nodes, n_dim))
    # 逆関数法で点を双曲空間からサンプリング
    # 双曲空間の意味での極座標で表示
    for i in range(n_dim):
        if i == 0:
            val_array, cum_dens = calc_dist_r(n_dim, sigma, R)
        else:
            val_array, cum_dens = calc_dist_angle(n_dim, i)
        for j in range(n_nodes):
            idx = np.max(np.where(cum_dens <= x_polar[j, i])[0])
            x_polar[j, i] = val_array[idx]
    # 直交座標に変換(Euclid)
    print('sampling ended')

    x_e = convert_euclid(x_polar)

    print('convert euclid ended')

    adj_mat = np.zeros((n_nodes, n_nodes))
    # サンプリング用の行列
    sampling_mat = np.random.uniform(0, 1, adj_mat.shape)
    sampling_mat = np.triu(
        sampling_mat) + np.triu(sampling_mat).T - np.diag(sampling_mat.diagonal())

    # lorentz scalar product
    first_term = - x_e[:, :1] * x_e[:, :1].T
    remaining = x_e[:, 1:].dot(x_e[:, 1:].T)
    adj_mat = - (first_term + remaining)

    for i in range(n_nodes):
        adj_mat[i, i] = 1
    # distance matrix
    adj_mat = np.arccosh(adj_mat)
    # probability matrix
    adj_mat = connection_prob(adj_mat, R, beta)

    for i in range(n_nodes):
        adj_mat[i, i] = 0

    adj_mat = np.where(sampling_mat < adj_mat, 1, 0)

    print(adj_mat)

    print("adj mat generated")

    return adj_mat, x_e


def convert_euclid(x_polar):
    n_nodes = x_polar.shape[0]
    n_dim = x_polar.shape[1]
    x_euclid = np.zeros((n_nodes, n_dim + 1))
    x_euclid[:, 0] = np.cosh(x_polar[:, 0])
    for i in range(n_dim):
        x_euclid[:, i + 1] = np.sinh(x_polar[:, 0])
        for j in range(0, i + 1):
            if j + 1 < n_dim:
                if j == i:
                    x_euclid[:, i + 1] *= np.cos(x_polar[:, j + 1])
                else:
                    x_euclid[:, i + 1] *= np.sin(x_polar[:, j + 1])
    return x_euclid


def calc_likelihood(n_nodes, n_dim, sigma, R, sigma_min, sigma_max):

    x_polar = np.random.uniform(0, 1, (n_nodes))
    # 逆関数法で点を双曲空間からサンプリング
    # 双曲空間の意味での極座標で表示
    val_array, cum_dens = calc_dist_r(n_dim, sigma, R)
    for j in range(n_nodes):
        idx = np.max(np.where(cum_dens <= x_polar[j])[0])
        x_polar[j] = val_array[idx]

    r = x_polar

    print(r)

    n_dim_list = [2, 4, 8, 16, 32, 64]

    for d in n_dim_list:
        sigma_list, lik, sigma_hat = calc_likelihood_list(
            r, d, R, sigma_min, sigma_max)

        print("d=", d, ":", sigma_hat)
        print("lik=", np.min(lik))

        plt.plot(sigma_list, lik, label="d=" + str(d))

    plt.legend()
    plt.savefig("test.png")


def beta_hat_test(n_nodes, n_dim, beta, sigma, beta_min, beta_max):
    R = np.log(n_nodes)

    params_adj_mat = {
        'n_nodes': n_nodes,
        'n_dim': n_dim,
        'R': R,
        'sigma': sigma,
        'beta': beta
    }

    adj_mat, x_e = hyperbolic_geometric_graph(
        n_nodes=params_adj_mat["n_nodes"],
        n_dim=params_adj_mat["n_dim"],
        R=params_adj_mat["R"],
        sigma=params_adj_mat["sigma"],
        beta=params_adj_mat["beta"]
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x_e = torch.Tensor(x_e).to(device)

    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    n_samples = int(n_nodes * 0.1)

    lr = calc_beta_hat(z=x_e, train_graph=adj_mat, n_samples=n_samples,
                       R=params_adj_mat["R"], beta_min=beta_min, beta_max=beta_max)
    print(lr)

    n_samples = int(n_nodes * 1.0)

    lr = calc_beta_hat(z=x_e, train_graph=adj_mat, n_samples=n_samples,
                       R=params_adj_mat["R"], beta_min=beta_min, beta_max=beta_max)
    print(lr)

    _, _, sigma_hat = calc_likelihood_list(
        arcosh(x_e[:, 0].to("cpu").numpy(), use_torch=False), n_dim, R, 0.1, 10)

    print(sigma_hat)


def beta_hat__(n_nodes, n_dim, beta, sigma, beta_min, beta_max):
    R = np.log(n_nodes)

    params_adj_mat = {
        'n_nodes': n_nodes,
        'n_dim': n_dim,
        'R': R,
        'sigma': sigma,
        'beta': beta
    }

    adj_mat, x_e = hyperbolic_geometric_graph(
        n_nodes=params_adj_mat["n_nodes"],
        n_dim=params_adj_mat["n_dim"],
        R=params_adj_mat["R"],
        sigma=params_adj_mat["sigma"],
        beta=params_adj_mat["beta"]
    )

    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    # lorentz scalar product
    first_term = - x_e[:, :1] * x_e[:, :1].T
    remaining = x_e[:, 1:].dot(x_e[:, 1:].T)
    adj_mat_hat = - (first_term + remaining)

    for i in range(n_nodes):
        adj_mat_hat[i, i] = 1
    # distance matrix
    adj_mat_hat = np.arccosh(adj_mat_hat)
    # probability matrix
    # adj_mat_hat = connection_prob(adj_mat_hat, R, beta)

    for i in range(n_nodes):
        # adj_mat_hat[i, i] = -1
        adj_mat_hat[i, :i + 1] = -1

    # print(adj_mat_hat)
    # print(adj_mat)

    y = adj_mat.flatten()
    x = adj_mat_hat.flatten()

    non_empty_idx = np.where(x != -1)[0]
    y = y[non_empty_idx].reshape((-1, 1))
    print(np.sum(y))
    x = -(x[non_empty_idx] - R).reshape((-1, 1))
    print(len(x))
    # x = np.exp(x)
    # print(empty_idx)
    # print(len(empty_idx))

    print(y)
    print(x)

    lr = LogisticRegression()  # ロジスティック回帰モデルのインスタンスを作成
    lr.fit(x, y)  # ロジスティック回帰モデルの重みを学習

    print(lr.coef_)

if __name__ == '__main__':
    n_dim_true_list = [16]
    n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
    sigma_list = [0.5, 1.0, 2.0]
    beta_list = [0.6, 0.8, 1.0, 1.2]

    for n_dim_true in n_dim_true_list:
        for n_nodes in n_nodes_list:
            count = 0
            for sigma in sigma_list:
                for beta in beta_list:
                    p_list = {
                        "n_dim_true": n_dim_true,
                        "n_nodes": n_nodes,
                        "sigma": sigma,
                        "beta": beta
                    }
                    print(p_list)
                    params_adj_mat = {
                        'n_nodes': n_nodes,
                        'n_dim': n_dim_true,
                        'R': np.log(n_nodes),
                        'sigma': sigma,
                        'beta': beta
                    }
                    adj_mat, x_e = hyperbolic_geometric_graph(
                        n_nodes=params_adj_mat["n_nodes"],
                        n_dim=params_adj_mat["n_dim"],
                        R=params_adj_mat["R"],
                        sigma=params_adj_mat["sigma"],
                        beta=params_adj_mat["beta"]
                    )

                    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
                        adj_mat, params_adj_mat)

                    # 平均次数が少なくなるように手で調整する用
                    print('average degree:', np.sum(adj_mat) / len(adj_mat))

                    adj_mat = coo_matrix(adj_mat)
                    train_graph = coo_matrix(train_graph)

                    graph_dict = {
                        "params_adj_mat": params_adj_mat,
                        "adj_mat": adj_mat,
                        "positive_samples": positive_samples,
                        "negative_samples": negative_samples,
                        "train_graph": train_graph,
                        "lik_data": lik_data,
                        "x_e": x_e
                    }

                    os.makedirs('dataset/dim_' +
                                str(params_adj_mat['n_dim']), exist_ok=True)
                    np.save('dataset/dim_' + str(params_adj_mat['n_dim']) + '/graph_' + str(
                        params_adj_mat['n_nodes']) + '_' + str(count) + '.npy', graph_dict)
                    count += 1
