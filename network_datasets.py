import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate, stats
# from embed import create_dataset
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
# import networkx as nx

import networkx as nx
import pandas as pd
import wget
import tarfile
import sys
import pickle as pkl
from scipy.sparse import coo_matrix

# if not os.path.exists("dataset/cora"):
#     wget.download(
#         "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz", out="dataset")
#     tar_path = 'dataset/cora.tgz'

#     with tarfile.open(tar_path, 'r:gz') as tar:
#         tar.extractall(path="dataset")

# # airport dataset
# os.makedirs("dataset/airport", exist_ok=True)
# wget.download(
#     "https://github.com/HazyResearch/hgcn/raw/master/data/airport/airport.p", out="dataset/airport")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/raw/master/data/airport/airport_alldata.p", out="dataset/airport")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/raw/master/data/airport/routes.dat", out="dataset/airport")

# # pubmed dataset
# os.makedirs("dataset/pubmed", exist_ok=True)
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/pubmed/ind.pubmed.allx", out="dataset/pubmed")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/pubmed/ind.pubmed.ally", out="dataset/pubmed")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/pubmed/ind.pubmed.graph", out="dataset/pubmed")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/pubmed/ind.pubmed.test.index", out="dataset/pubmed")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/pubmed/ind.pubmed.tx", out="dataset/pubmed")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/pubmed/ind.pubmed.ty", out="dataset/pubmed")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/pubmed/ind.pubmed.x", out="dataset/pubmed")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/pubmed/ind.pubmed.y", out="dataset/pubmed")

# # cora dataset
# os.makedirs("dataset/cora", exist_ok=True)
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/cora/ind.cora.allx", out="dataset/cora")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/cora/ind.cora.ally", out="dataset/cora")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/cora/ind.cora.graph", out="dataset/cora")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/cora/ind.cora.test.index", out="dataset/cora")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/cora/ind.cora.tx", out="dataset/cora")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/cora/ind.cora.ty", out="dataset/cora")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/cora/ind.cora.x", out="dataset/cora")
# wget.download(
#     "https://github.com/HazyResearch/hgcn/blob/master/data/cora/ind.cora.y", out="dataset/cora")


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
    adj_mat = load_data_lp(dataset_name, "dataset/"+dataset_name)
    adj_mat = adj_mat.toarray().astype(np.int)

    n_nodes = adj_mat.shape[0]
    print("n_nodes:", n_nodes)

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

    # print(load_data_lp("airport", "dataset/airport"))
    # print(load_data_lp("pubmed", "dataset/pubmed"))
    # print(load_data_lp("cora", "dataset/cora"))

    # # Coraデータの読み込み
    # cora_content = pd.read_csv('dataset/cora/cora.content', sep='\t', header=None)
    # cora_cites = pd.read_csv('dataset/cora/cora.cites', sep='\t', header=None)

    # # ノード属性を表示
    # print("Node attributes:")
    # print(cora_content.head())

    # # エッジ（引用関係）を表示
    # print("\nEdge (citation) relationships:")
    # print(cora_cites.head())

    # # NetworkXグラフの作成
    # G = nx.from_pandas_edgelist(cora_cites, source=0, target=1)

    # # グラフの情報を表示
    # print("\nGraph information:")
    # print(nx.info(G))
