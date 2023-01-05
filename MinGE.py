import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import pandas as pd
from scipy import integrate
from copy import deepcopy
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix
from multiprocessing import Pool
from functools import partial

RESULTS = "results"


def integral_sin(n, theta):
    if n == 0:
        return theta
    elif n == 1:
        return 1 - np.cos(theta)
    else:
        return -np.cos(theta) * (np.sin(theta)**(n - 1)) / n + ((n - 1) / n) * integral_sin(n - 2, theta)


def calc_GE_realworld(dataset_name, n_dim_list, weight_entropy):
    adj_mat = np.load('dataset/' + dataset_name + '/' + dataset_name + '.npy')
    n_nodes = len(adj_mat)
    print(n_nodes)

    params_dataset = {
        'n_nodes': n_nodes,
    }

    N = params_dataset["n_nodes"]
    # Structure Entropy
    D = np.sum(adj_mat, axis=0) + 1
    adj_mat_2 = coo_matrix(adj_mat) * \
        coo_matrix(adj_mat).toarray().astype(np.float)

    t = np.sum(adj_mat_2, axis=1).astype(np.float)
    for i in range(N):
        if t[i] != 0:
            adj_mat_2[i, :] /= t[i]

    D_r = coo_matrix(adj_mat_2) * coo_matrix(D.reshape((-1, 1))
                                             ).toarray().astype(np.float).reshape(-1)

    z = np.sum(D_r)
    p = D_r / z
    p_ = p[np.where(p > 0.00001)[0]]
    log_p = np.log(p_)
    H_s = -np.dot(p_, log_p)

    GE_list = []

    # Feature Entropy
    for n_dim in n_dim_list:

        H_f = 2 * np.log(N)
        term_1_ = lambda theta: np.exp(
            n_dim * np.cos(theta)) * (np.sin(theta)**(n_dim - 2)) / integral_sin(n_dim - 2, np.pi)
        term_1 = integrate.quad(term_1_, 0, np.pi)[0]
        H_f += np.log(term_1)

        term_2_ = lambda theta: np.exp(n_dim * np.cos(theta)) * n_dim * np.cos(
            theta) * (np.sin(theta)**(n_dim - 2)) / integral_sin(n_dim - 2, np.pi)
        term_2 = integrate.quad(term_2_, 0, np.pi)[0] / term_1
        H_f -= term_2

        GE_list.append(H_f + weight_entropy * H_s)

    return GE_list


def calc_GE(dim_true, n_dim, n_nodes, n_graph, weight_entropy):
    dataset = np.load('dataset/dim_' + str(dim_true) + '/graph_' + str(n_nodes) + '_' + str(n_graph) +
                      '.npy', allow_pickle='TRUE').item()  # object型なので、itemを付けないと辞書として使えない。
    adj_mat = dataset["adj_mat"]
    params_dataset = dataset["params_adj_mat"]

    N = params_dataset["n_nodes"]
    # Structure Entropy
    D = np.sum(adj_mat, axis=0) + 1
    adj_mat_2 = coo_matrix(adj_mat) * \
        coo_matrix(adj_mat).toarray().astype(np.float)

    t = np.sum(adj_mat_2, axis=1).astype(np.float)
    for i in range(N):
        if t[i] != 0:
            adj_mat_2[i, :] /= t[i]

    D_r = coo_matrix(adj_mat_2) * coo_matrix(D.reshape((-1, 1))
                                             ).toarray().astype(np.float).reshape(-1)

    z = np.sum(D_r)
    p = D_r / z
    p_ = p[np.where(p > 0.00001)[0]]
    log_p = np.log(p_)
    H_s = -np.dot(p_, log_p)

    # Feature Entropy
    H_f = 2 * np.log(N)
    term_1_ = lambda theta: np.exp(
        n_dim * np.cos(theta)) * (np.sin(theta)**(n_dim - 2)) / integral_sin(n_dim - 2, np.pi)
    term_1 = integrate.quad(term_1_, 0, np.pi)[0]
    H_f += np.log(term_1)

    term_2_ = lambda theta: np.exp(n_dim * np.cos(theta)) * n_dim * np.cos(
        theta) * (np.sin(theta)**(n_dim - 2)) / integral_sin(n_dim - 2, np.pi)
    term_2 = integrate.quad(term_2_, 0, np.pi)[0] / term_1
    H_f -= term_2

    return H_f + weight_entropy * H_s


def calc_artificial(inputs, n_dim_list, weight_entropy):
    print(inputs)
    dataset, dim_true, n_nodes, n_graph = inputs
    result = pd.DataFrame()
    entropy_list = []
    for n_dim in n_dim_list:
        entropy_list.append(
            calc_GE(dim_true, n_dim, n_nodes, n_graph, weight_entropy))
    entropy_list = np.array(entropy_list)
    result["model_n_dims"] = n_dim_list
    result["MinGE"] = entropy_list

    os.makedirs(RESULTS + "/" + dataset + "/dim_" +
                str(dim_true) + "/", exist_ok=True)
    result.to_csv(RESULTS + "/" + dataset + "/dim_" + str(dim_true) + "/result_" + str(n_nodes) +
                  "_" + str(n_graph) + "_MinGE.csv", index=False)
    print("estimated dimensionality:",
          n_dim_list[np.argmin(np.abs(entropy_list))])


def artificial_dataset(
    dataset_list,
    n_dim_true_list,
    n_nodes_list,
    n_dim_list,
    n_graphs,
    weight_entropy
):
    values = list(itertools.product(
        dataset_list,
        n_dim_true_list,
        n_nodes_list,
        range(n_graphs)
    ))

    calc_artificial_ = partial(
        calc_artificial,
        n_dim_list=n_dim_list,
        weight_entropy=weight_entropy
    )

    # multiprocessing
    p = Pool(12)

    results = p.map(calc_artificial_, values)

def calc_GE_lexical(n_dim, weight_entropy, dataset_name):
    data = np.load("dataset/wn_dataset/" + dataset_name +
                   "_data.npy", allow_pickle=True).item()
    adj_mat = data["adj_mat"]

    N = len(adj_mat)
    # Structure Entropy
    D = np.sum(adj_mat, axis=0) + 1
    adj_mat_2 = coo_matrix(adj_mat) * \
        coo_matrix(adj_mat).toarray().astype(np.float)

    t = np.sum(adj_mat_2, axis=1).astype(np.float)
    for i in range(N):
        if t[i] != 0:
            adj_mat_2[i, :] /= t[i]

    D_r = coo_matrix(adj_mat_2) * coo_matrix(D.reshape((-1, 1))
                                             ).toarray().astype(np.float).reshape(-1)

    z = np.sum(D_r)
    p = D_r / z
    p_ = p[np.where(p > 0.00001)[0]]
    log_p = np.log(p_)
    H_s = -np.dot(p_, log_p)

    # Feature Entropy
    H_f = 2 * np.log(N)
    term_1_ = lambda theta: np.exp(
        n_dim * np.cos(theta)) * (np.sin(theta)**(n_dim - 2)) / integral_sin(n_dim - 2, np.pi)
    term_1 = integrate.quad(term_1_, 0, np.pi)[0]
    H_f += np.log(term_1)

    term_2_ = lambda theta: np.exp(n_dim * np.cos(theta)) * n_dim * np.cos(
        theta) * (np.sin(theta)**(n_dim - 2)) / integral_sin(n_dim - 2, np.pi)
    term_2 = integrate.quad(term_2_, 0, np.pi)[0] / term_1
    H_f -= term_2

    return H_f + weight_entropy * H_s


def lexical_dataset(n_dim_list, dataset_name):
    print(dataset_name)
    weight_entropy = 1.0

    result = pd.DataFrame()
    entropy_list = []
    for n_dim in n_dim_list:
        entropy_list.append(
            calc_GE_lexical(n_dim, weight_entropy, dataset_name))
    entropy_list = np.array(entropy_list)
    result["model_n_dims"] = n_dim_list
    result["MinGE"] = entropy_list
    result.to_csv(RESULTS + "/" + dataset_name +
                  "/result_MinGE.csv", index=False)
    print("estimated dimensionality:",
          n_dim_list[np.argmin(entropy_list)])

if __name__ == "__main__":
    # artificial dataset
    print("Results of Artificial Datasets")
    n_dim_true_list = [16]
    dataset_list = ["HGG", "WND"]
    n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
    n_dim_list = [2, 4, 8, 16, 32, 64]
    n_graphs = 12
    weight_entropy = 1.0
    artificial_dataset(
        dataset_list=dataset_list,
        n_dim_true_list=n_dim_true_list,
        n_nodes_list=n_nodes_list,
        n_dim_list=n_dim_list,
        n_graphs=n_graphs,
        weight_entropy=weight_entropy
    )

    # # link prediction
    # n_dim_list = [2, 4, 8, 16, 32, 64]
    # dataset_name_list = ["ca-AstroPh", "ca-HepPh", "ca-CondMat", "ca-GrQc"]

    # for dataset_name in dataset_name_list:
    #     result = pd.DataFrame()
    #     entropy_list = calc_GE_realworld(
    #         dataset_name, n_dim_list, weight_entropy)

    #     result["model_n_dims"] = n_dim_list
    #     result["MinGE"] = entropy_list

    #     result.to_csv(RESULTS + "/" + dataset_name +
    #                   "/result_MinGE.csv", index=False)

    # lexical dataset
    # n_dim_list = [2, 3, 4, 5, 6, 7, 8, 16, 32, 64]
    # dataset_name_list = [
    #     "animal",
    #     "group",
    #     "mammal",
    #     "solid",
    #     "tree",
    #     "verb",
    #     "worker"
    # ]
    # for dataset_name in dataset_name_list:
    #     lexical_dataset(n_dim_list, dataset_name)
