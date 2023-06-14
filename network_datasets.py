import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate, stats
from copy import deepcopy
import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score
import matplotlib.pyplot as plt
import matplotlib
from experiment_wn import is_a_score
import torch
from multiprocessing import Pool
from functools import partial
import stellargraph as sg
from utils.utils_dataset import create_test_for_link_prediction

import networkx as nx
import pandas as pd
import wget
import tarfile
import sys
import pickle as pkl
from scipy.sparse import coo_matrix


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    return adj


def load_citation_data(dataset_str, data_path, split_seed=None):
    with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, "graph")), 'rb') as f:
        if sys.version_info > (3, 0):
            graph = pkl.load(f, encoding='latin1')
        else:
            graph = pkl.load(f)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj


def load_data_lp(dataset, data_path):
    if dataset in ['cora', 'pubmed']:
        adj = load_citation_data(dataset, data_path)
    elif dataset == 'airport':
        adj = load_data_airport(dataset, data_path, return_label=False)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = adj
    return data


def data_generation(dataset_name):
    print(dataset_name)
    adj_mat = load_data_lp(dataset_name, "dataset/" + dataset_name)
    adj_mat = adj_mat.toarray().astype(np.int)

    n_nodes = adj_mat.shape[0]

    # 無向グラフなので有向グラフに変換する
    adj_u = np.triu(adj_mat) + np.triu(adj_mat).T
    adj_l = np.tril(adj_mat) + np.tril(adj_mat).T

    for i in range(n_nodes):
        adj_u[i, i] = 0
        adj_l[i, i] = 0

    adj_mat = np.maximum(adj_u, adj_l)
    adj_mat = np.minimum(adj_mat, 1)
    adj_mat = np.maximum(adj_mat, 0)

    print("n_nodes:", n_nodes)
    print("n_edges:", np.sum(adj_mat) / 2)
    print(adj_mat)

    params_dataset = {
        "n_nodes": n_nodes
    }

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat, params_dataset)

    data = {
        "adj_mat": coo_matrix(adj_mat),
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "train_graph": coo_matrix(train_graph),
        "lik_data": lik_data,
    }

    np.save('dataset/' + dataset_name + "/data.npy", data)

if __name__ == "__main__":
    data_generation("airport")
    data_generation("cora")
    data_generation("pubmed")
