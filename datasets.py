import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from scipy import integrate
from scipy.sparse import coo_matrix
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from utils.utils import (
    integral_sinh,
    calc_likelihood_list,
    arcosh,
    calc_beta_hat,
    exp_map,
    plot_figure,
    sin_k,
    cos_k,
    arcos_k,
    inner_product_k,
    dist_k,
    tangent_norm_k,
    exp_map_k,
    projection_k,
    log_map_k
)
from utils.utils_dataset import (
    create_test_for_link_prediction,
    create_dataset,
    # connection_prob,
    integral_sin,
    calc_dist_angle,
    calc_dist_r,
    init_HGG,
    hyperbolic_geometric_graph,
    convert_euclid
)
from multiprocessing import Pool
import itertools

np.random.seed(0)

def connection_prob(d, gamma):
    """
    接続確率
    """
    return 1 / (1 + np.exp(d - gamma))


def wrapped_normal_distribution_k(n_nodes, n_dim, gamma, Sigma, k):
    if k == 0:
        x_e = np.random.multivariate_normal(
            np.zeros(n_dim), Sigma, size=n_nodes)
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
        adj_mat = connection_prob(adj_mat, gamma)

        for i in range(n_nodes):
            adj_mat[i, i] = 0

        adj_mat = np.where(sampling_mat < adj_mat, 1, 0)

        print(adj_mat)

        print("adj mat generated")

        return adj_mat, x_e
    else:
        if k < 0:  # hyperbolic case
            v = np.random.multivariate_normal(
                np.zeros(n_dim), Sigma, size=n_nodes)
            print(v.shape)
            v_ = np.zeros((n_nodes, n_dim + 1))
            v_[:, 1:] = v  # tangent vector
        else:  # spherical case
            v = []
            while len(v) < n_nodes:
                # print(len(v))
                samples = np.random.multivariate_normal(np.zeros(n_dim), Sigma, size=n_nodes*3)
                norms = np.linalg.norm(samples, axis=1)
                valid_indices = np.where(norms <= np.pi/np.sqrt(k))[0]
                valid_samples = samples[valid_indices, :]
                v.extend(valid_samples)
            v = v[:n_nodes]
            v = np.array(v)
            # v = np.random.multivariate_normal(
            #     np.zeros(n_dim), Sigma, size=n_nodes)
            # dists = np.linalg.norm(v, axis=1)
            # v = v[np.where(dists <= np.pi / np.sqrt(k))[0], :]
            # print(v.shape)
            # v = v[:n_nodes, :]
            v_ = np.zeros((n_nodes, n_dim + 1))
            v_[:, 1:] = v  # tangent vector

        mean = np.zeros((n_nodes, n_dim + 1))
        mean[:, 0] = 1 / np.sqrt(abs(k))
        x_e = exp_map_k(torch.tensor(mean), torch.tensor(v_), k).numpy()

        print('convert euclid ended')

        adj_mat = np.zeros((n_nodes, n_nodes))
        # サンプリング用の行列
        sampling_mat = np.random.uniform(0, 1, adj_mat.shape)
        sampling_mat = np.triu(
            sampling_mat) + np.triu(sampling_mat).T - np.diag(sampling_mat.diagonal())

        if k < 0:
            # lorentz scalar product
            first_term = - x_e[:, :1] * x_e[:, :1].T
            remaining = x_e[:, 1:].dot(x_e[:, 1:].T)
            adj_mat = first_term + remaining
        else:
            adj_mat = x_e.dot(x_e.T)

        for i in range(n_nodes):
            adj_mat[i, i] = 1

        print(adj_mat)
        # distance matrix
        adj_mat = arcos_k(k * adj_mat, k, use_torch=False) / np.sqrt(abs(k))

        print("dist_mat:", adj_mat)
        # probability matrix
        adj_mat = connection_prob(adj_mat, gamma)

        for i in range(n_nodes):
            adj_mat[i, i] = 0

        adj_mat = np.where(sampling_mat < adj_mat, 1, 0)

        print(adj_mat)
        print("adj mat generated")

        return adj_mat, x_e


def wrapped_normal_distribution_k_(n_nodes, n_dim, beta, gamma, Sigma, k):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    if k == 0:
        x_e = np.random.multivariate_normal(
            np.zeros(n_dim), Sigma, size=n_nodes)
        print('convert euclid ended')
        plt.gca().set_xlim(-5, 5)
        plt.gca().set_ylim(-5, 5)
        plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, edgecolor="black"))
        plt.scatter(
            x_e[:, 0], x_e[:, 1], color="blue")
        plt.savefig("temp/euclid.png")
    else:
        if k < 0:  # hyperbolic case
            v = np.random.multivariate_normal(
                np.zeros(n_dim), Sigma, size=n_nodes)
            print(v.shape)
            v_ = np.zeros((n_nodes, n_dim + 1))
            v_[:, 1:] = v  # tangent vector
        else:  # spherical case
            v = np.random.multivariate_normal(
                np.zeros(n_dim), Sigma, size=n_nodes * 10)
            dists = np.linalg.norm(v, axis=1)
            v = v[np.where(dists <= np.pi / np.sqrt(k))[0], :]
            print(v.shape)
            v = v[:n_nodes, :]
            v_ = np.zeros((n_nodes, n_dim + 1))
            v_[:, 1:] = v  # tangent vector

        mean = np.zeros((n_nodes, n_dim + 1))
        mean[:, 0] = 1 / np.sqrt(abs(k))
        x_e = exp_map_k(torch.tensor(mean), torch.tensor(v_), k).numpy()

        if k > 0:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')

            ax.cla()  # プロットをクリア
            ax.set_xlim3d([-1, 1])  # x軸範囲を設定
            ax.set_ylim3d([-1, 1])  # y軸範囲を設定
            ax.set_zlim3d([-1, 1])  # z軸範囲を設定
            ax.scatter(x_e[:, 1], x_e[:, 2], x_e[:, 0])
            plt.savefig("temp/spherical.png")
        else:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')

            ax.cla()  # プロットをクリア
            ax.set_xlim3d([-2, 2])  # x軸範囲を設定
            ax.set_ylim3d([-2, 2])  # y軸範囲を設定
            ax.set_zlim3d([0, 4])  # z軸範囲を設定
            ax.scatter(x_e[:, 1], x_e[:, 2], x_e[:, 0])
            plt.savefig("temp/hyperbolic.png")


def generate_wnd_k(inputs):
    n_graph, n_dim_true, n_nodes, sigma, gamma, k = inputs
    p_list = {
        "n_dim_true": n_dim_true,
        "n_nodes": n_nodes,
        "sigma": sigma,
        # "beta": beta,
        "gamma": gamma,
        "k": k
    }
    print(p_list)
    params_adj_mat = {
        "n_nodes": n_nodes,
        "n_dim": n_dim_true,
        "R": np.log(n_nodes)+6,
        "Sigma": np.eye(n_dim_true) * sigma,
        # "beta": beta,
        "gamma": gamma,
        "k": k
    }
    adj_mat, x_e = wrapped_normal_distribution_k(
        n_nodes=params_adj_mat["n_nodes"],
        n_dim=params_adj_mat["n_dim"],
        Sigma=params_adj_mat["Sigma"],
        # beta=params_adj_mat["beta"],
        gamma=params_adj_mat["gamma"],
        k=params_adj_mat["k"]
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

    if k > 0:
        dataset_name = "spherical"
    elif k < 0:
        dataset_name = "hyperbolic"
    else:
        dataset_name = "euclidean"
    os.makedirs("dataset/" + dataset_name + "/dim_" +
                str(params_adj_mat['n_dim']), exist_ok=True)
    np.save("dataset/" + dataset_name + "/dim_" + str(params_adj_mat['n_dim']) + '/graph_' + str(
        params_adj_mat['n_nodes']) + '_' + str(n_graph) + '.npy', graph_dict)

    return inputs, avg_deg


# def create_wnds_non_euclidean(n_dim_true_list, n_nodes_list, sigma_list, gamma_list, k_list):
def create_wnds_non_euclidean(n_dim_true_list, n_nodes_list, gamma_list, k_list, k):
    # n_graph_list = list(np.array(range(len(n_dim_true_list) * len(n_nodes_list)
    #                                    * len(sigma_list) * len(k_list))) % (len(sigma_list) * len(k_list)))
    n_graph_list = list(np.array(range(len(n_dim_true_list) * len(n_nodes_list)
                                       * len(gamma_list) * len(k_list))) % (len(gamma_list) * len(k_list)))

    # values_ = []

    # for i, n_dim_true in enumerate(n_dim_true_list):
    #     values_n_dim_true = itertools.product(
    #         n_nodes_list, sigma_list[i], beta_list[i])

    #     for tuple_value in values_n_dim_true:
    #         v = (n_dim_true, tuple_value[0], tuple_value[1], tuple_value[2])
    #         values_.append(v)

    # values_ = list(itertools.product(n_dim_true_list,
    #                                  n_nodes_list, sigma_list, k_list))
    values_ = list(itertools.product(n_dim_true_list,
                                     n_nodes_list, gamma_list, k_list))

    values = []
    if k<0:
        for n_graph, value in zip(n_graph_list, values_):
            # values.append((n_graph, value[0], value[1], 2.0/np.sqrt(np.abs(value[3])), value[
            #               2], value[3]))
            values.append((n_graph, value[0], value[1], 5.5/np.sqrt(np.abs(value[3])), value[
                          2], value[3]))
    else:
        for n_graph, value in zip(n_graph_list, values_):
            values.append((n_graph, value[0], value[1], 4.5/np.sqrt(np.abs(value[3])), value[
                          2], value[3]))

    print(values)

    # multiprocessing
    p = Pool(12)

    results = p.map(generate_wnd_k, values)

    print("----multiprocessing ended----")
    for result in results:
        inputs = result[0]
        avg_deg = result[1]
        print("n_dim_true:", inputs[1], ", n_nodes:", inputs[
              2],  "sigma:", inputs[3],  "gamma:", inputs[4], "k:", inputs[5])
        print("average degree:", avg_deg)


def create_wnds_euclidean(n_dim_true_list, n_nodes_list, sigma_list, gamma_list):
    n_graph_list = list(np.array(range(len(n_dim_true_list) * len(n_nodes_list)
                                       * len(sigma_list) * len(gamma_list))) % (len(sigma_list) * len(gamma_list)))

    # values_ = []

    # for i, n_dim_true in enumerate(n_dim_true_list):
    #     values_n_dim_true = itertools.product(
    #         n_nodes_list, sigma_list[i], beta_list[i])

    #     for tuple_value in values_n_dim_true:
    #         v = (n_dim_true, tuple_value[0], tuple_value[1], tuple_value[2])
    #         values_.append(v)

    values_ = list(itertools.product(n_dim_true_list,
                                     n_nodes_list, sigma_list, gamma_list))

    values = []
    for n_graph, value in zip(n_graph_list, values_):
        values.append((n_graph, value[0], value[1], value[
                      2], value[3], 0))

    print(values)

    # multiprocessing
    p = Pool(12)

    results = p.map(generate_wnd_k, values)

    print("----multiprocessing ended----")
    for result in results:
        inputs = result[0]
        avg_deg = result[1]
        print("n_dim_true:", inputs[1], ", n_nodes:", inputs[
              2],  "sigma:", inputs[3], "gamma:", inputs[4])
        print("average degree:", avg_deg)



if __name__ == '__main__':
    # Euclidean
    # true dim 8
    n_dim_true_list = [8]
    sigma_list = [10.0, 12.0, 14.0]
    gamma_list = [4.5, 5.0, 5.5, 6.0]
    n_nodes_list = [400, 800, 1600, 3200, 6400]
    create_wnds_euclidean(n_dim_true_list, n_nodes_list, sigma_list, gamma_list)

    # Lorentz
    # true dim 8
    n_dim_true_list = [8]
    gamma_list = [5.0, 5.25, 5.5, 5.75]
    k_list = [-0.75, -1, -1.25]
    n_nodes_list = [400, 800, 1600, 3200, 6400]
    create_wnds_non_euclidean(n_dim_true_list, n_nodes_list, gamma_list, k_list, k=-1)

    # Spherical
    # true dim 8
    n_dim_true_list = [8]
    gamma_list = [0.1, 0.125, 0.15, 0.175]
    k_list = [0.05, 0.1, 0.2]
    n_nodes_list = [400, 800, 1600, 3200, 6400]
    # n_nodes_list = [400]
    create_wnds_non_euclidean(n_dim_true_list, n_nodes_list, gamma_list, k_list, k=1)

    # # Euclidean
    # # true dim 16
    # n_dim_true_list = [16]
    # sigma_list = [1.5, 2.0, 2.5]
    # gamma_list = [2.25, 2.5, 2.75, 3.0]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    # create_wnds_euclidean(n_dim_true_list, n_nodes_list, sigma_list, gamma_list)

    # # Lorentz
    # # true dim 16
    # n_dim_true_list = [16]
    # gamma_list = [3.00, 3.25, 3.5, 3.75]
    # k_list = [-0.5, -1, -2]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    # # n_nodes_list = [400]
    # create_wnds_non_euclidean(n_dim_true_list, n_nodes_list, gamma_list, k_list, k=-1)

    # # Spherical
    # # true dim 16
    # n_dim_true_list = [16]
    # gamma_list = [0.1, 0.15, 0.2, 0.25]
    # k_list = [0.1, 0.2, 0.3]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    # # n_nodes_list = [400]
    # create_wnds_non_euclidean(n_dim_true_list, n_nodes_list, gamma_list, k_list, k=1)

    # adj_mat, _ = wrapped_normal_distribution_k(
    #     n_nodes=3200, n_dim=8, gamma=0.1, Sigma=np.eye(8)*3, k=0.2)
    # print(_)
    # print(adj_mat)
    # print('average degree:', np.sum(adj_mat) / len(adj_mat))

    # wrapped_normal_distribution_k_(
    #     n_nodes=600, n_dim=2, beta=0.5, gamma=0.5, Sigma=np.eye(2) * 0.5, k=0)
    # wrapped_normal_distribution_k_(
    #     n_nodes=600, n_dim=2, beta=0.5, gamma=0.5, Sigma=np.eye(2) * 0.5, k=1)
    # wrapped_normal_distribution_k_(
    # n_nodes=600, n_dim=2, beta=0.5, gamma=0.5, Sigma=np.eye(2) * 0.5, k=-1)

    # Lorentz
    # true dim 8
    # n_dim_true_list = [8]
    # sigma_list = [0.35, 0.375, 0.40]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    # create_wnds_k(n_dim_true_list, n_nodes_list, sigma_list, beta_list, k=-1)

    # # true dim 16
    # n_dim_true_list = [16]
    # sigma_list = [0.225, 0.25, 0.275]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    # create_wnds_k(n_dim_true_list, n_nodes_list, sigma_list, beta_list, k=-1)

    # # HGG
    # # true dim 8
    # n_dim_true_list = [8]
    # sigma_list = [0.5, 1.0, 2.0]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    # create_hggs(n_dim_true_list, n_nodes_list, sigma_list, beta_list)

    # # true dim 16
    # n_dim_true_list = [16]
    # sigma_list = [0.5, 1.0, 2.0]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    # create_hggs(n_dim_true_list, n_nodes_list, sigma_list, beta_list)

    # # WND
    # # true dim 8
    # n_dim_true_list = [8]
    # sigma_list = [0.35, 0.375, 0.40]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    # create_wnds(n_dim_true_list, n_nodes_list, sigma_list, beta_list)

    # # true dim 16
    # n_dim_true_list = [16]
    # sigma_list = [0.225, 0.25, 0.275]
    # beta_list = [0.5, 0.6, 0.7, 0.8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    # create_wnds(n_dim_true_list, n_nodes_list, sigma_list, beta_list)
