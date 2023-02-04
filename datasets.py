import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy import integrate
from scipy.sparse import coo_matrix
from copy import deepcopy
from utils.utils_dataset import create_test_for_link_prediction, create_dataset
from sklearn.linear_model import LogisticRegression
from utils.utils import (
    integral_sinh,
    calc_likelihood_list,
    arcosh,
    calc_beta_hat,
    exp_map,
    plot_figure
)
from multiprocessing import Pool
import itertools

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


def init_HGG(n_nodes, n_dim, R, sigma, beta):
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

    return x_e


def hyperbolic_geometric_graph(n_nodes, n_dim, R, sigma, beta):
    # TODO: プログラム前半部分も実行時間を短くする。
    # 現状は次元の2乗オーダーの計算量
    # n_dimは2以上で
    x_e = init_HGG(n_nodes, n_dim, R, sigma, beta)

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

    # print(adj_mat)

    # print("adj mat generated")

    return adj_mat, x_e


def wrapped_normal_distribution(n_nodes, n_dim, R, Sigma, beta):
    v = np.random.multivariate_normal(np.zeros(n_dim), Sigma, size=n_nodes)
    print(v.shape)
    v_ = np.zeros((n_nodes, n_dim + 1))
    v_[:, 1:] = v  # tangent vector

    mean = np.zeros((n_nodes, n_dim + 1))
    mean[:, 0] = 1
    x_e = exp_map(torch.tensor(mean), torch.tensor(v_)).numpy()

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

    # print(adj_mat)

    # print("adj mat generated")

    return adj_mat, x_e


def euclidean_geometric_graph(n_nodes, n_dim, R, Sigma, beta):
    x_e = np.random.multivariate_normal(
        np.zeros(n_dim), Sigma, size=n_nodes)
    # print(v.shape)
    # v_ = np.zeros((n_nodes, n_dim + 1))
    # v_[:, 1:] = v  # tangent vector

    # mean = np.zeros((n_nodes, n_dim + 1))
    # mean[:, 0] = 1
    # x_e = exp_map(torch.tensor(mean), torch.tensor(v_)).numpy()

    print('convert euclid ended')

    adj_mat = np.zeros((n_nodes, n_nodes))
    # サンプリング用の行列
    sampling_mat = np.random.uniform(0, 1, adj_mat.shape)
    sampling_mat = np.triu(
        sampling_mat) + np.triu(sampling_mat).T - np.diag(sampling_mat.diagonal())

    # lorentz scalar product
    def distance_mat(X, Y):
        X = X[:, np.newaxis, :]
        Y = Y[np.newaxis, :, :]
        Z = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        return Z

    adj_mat = distance_mat(x_e, x_e)

    for i in range(n_nodes):
        adj_mat[i, i] = 1

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


def generate_wnd(inputs):
    n_graph, n_dim_true, n_nodes, sigma, beta = inputs
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
        'Sigma': np.eye(n_dim_true) * ((np.log(n_nodes) * sigma)**2),
        'beta': beta
    }
    adj_mat, x_e = wrapped_normal_distribution(
        n_nodes=params_adj_mat["n_nodes"],
        n_dim=params_adj_mat["n_dim"],
        R=params_adj_mat["R"],
        Sigma=params_adj_mat["Sigma"],
        beta=params_adj_mat["beta"]
    )

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat, params_adj_mat)

    print('average degree:', np.sum(adj_mat) / len(adj_mat))
    avg_deg = np.sum(adj_mat) / len(adj_mat)

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

    os.makedirs('dataset/WND/dim_' +
                str(params_adj_mat['n_dim']), exist_ok=True)
    np.save('dataset/WND/dim_' + str(params_adj_mat['n_dim']) + '/graph_' + str(
        params_adj_mat['n_nodes']) + '_' + str(n_graph) + '.npy', graph_dict)

    return inputs, avg_deg


def generate_hgg(inputs):
    n_graph, n_dim_true, n_nodes, sigma, beta = inputs
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

    print('average degree:', np.sum(adj_mat) / len(adj_mat), p_list)
    avg_deg = np.sum(adj_mat) / len(adj_mat)

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

    os.makedirs('dataset/HGG/dim_' +
                str(params_adj_mat['n_dim']), exist_ok=True)
    np.save('dataset/HGG/dim_' + str(params_adj_mat['n_dim']) + '/graph_' + str(
        params_adj_mat['n_nodes']) + '_' + str(n_graph) + '.npy', graph_dict)

    return inputs, avg_deg


def create_wnds(n_dim_true_list, n_nodes_list, sigma_list, beta_list):
    n_graph_list = list(np.array(range(len(n_dim_true_list) * len(n_nodes_list)
                                       * len(sigma_list) * len(beta_list))) % (len(sigma_list) * len(beta_list)))

    # values_ = []

    # for i, n_dim_true in enumerate(n_dim_true_list):
    #     values_n_dim_true = itertools.product(
    #         n_nodes_list, sigma_list[i], beta_list[i])

    #     for tuple_value in values_n_dim_true:
    #         v = (n_dim_true, tuple_value[0], tuple_value[1], tuple_value[2])
    #         values_.append(v)

    values_ = list(itertools.product(n_dim_true_list,
                                     n_nodes_list, sigma_list, beta_list))

    values = []
    for n_graph, value in zip(n_graph_list, values_):
        values.append((n_graph, value[0], value[1], value[2], value[3]))

    print(values)

    # multiprocessing
    p = Pool(12)

    results = p.map(generate_wnd, values)

    print("----multiprocessing ended----")
    for result in results:
        inputs = result[0]
        avg_deg = result[1]
        print("n_dim_true:", inputs[1], ", n_nodes:", inputs[
              2],  "sigma:", inputs[3],  "beta:", inputs[4])
        print("average degree:", avg_deg)


def create_hggs(n_dim_true_list, n_nodes_list, sigma_list, beta_list):

    n_graph_list = list(np.array(range(len(n_dim_true_list) * len(n_nodes_list)
                                       * len(sigma_list) * len(beta_list))) % (len(sigma_list) * len(beta_list)))

    values_ = list(itertools.product(n_dim_true_list,
                                     n_nodes_list, sigma_list, beta_list))

    values = []
    for n_graph, value in zip(n_graph_list, values_):
        values.append((n_graph, value[0], value[1], value[2], value[3]))

    print(values)

    # multiprocessing
    p = Pool(12)

    results = p.map(generate_hgg, values)

    print("----multiprocessing ended----")
    for result in results:
        inputs = result[0]
        avg_deg = result[1]
        print("n_dim_true:", inputs[1], ", n_nodes:", inputs[
              2],  "sigma:", inputs[3],  "beta:", inputs[4])
        print("average degree:", avg_deg)

if __name__ == '__main__':
    # WND
    # true dim 8
    n_dim_true_list = [8]
    sigma_list = [0.35, 0.375, 0.40]
    beta_list = [0.5, 0.6, 0.7, 0.8]
    n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
    create_wnds(n_dim_true_list, n_nodes_list, sigma_list, beta_list)

    # # true dim 16
    # n_dim_true_list = [16]
    # sigma_list = [0.225, 0.25, 0.275]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
    # create_wnds(n_dim_true_list, n_nodes_list, sigma_list, beta_list)

    # HGG

    # true dim 4
    # n_dim_true_list = [4]
    # sigma_list = [0.5, 1.0, 2.0]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
    # create_hggs(n_dim_true_list, n_nodes_list, sigma_list, beta_list)

    # true dim 8
    # n_dim_true_list = [8]
    # sigma_list = [0.5, 1.0, 2.0]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
    # create_hggs(n_dim_true_list, n_nodes_list, sigma_list, beta_list)

    # # true dim 16
    # n_dim_true_list = [16]
    # sigma_list = [0.5, 1.0, 2.0]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
    # create_hggs(n_dim_true_list, n_nodes_list, sigma_list, beta_list)
