import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import torch
import numpy as np
import pandas as pd
import gc
import time
import matplotlib.pyplot as plt
import matplotlib
import torch.multiprocessing as multi
import os
from functools import partial
from scipy.io import mmread
from copy import deepcopy
from torch.utils.data import DataLoader
# from lorentz import LinkPrediction
from utils.utils import arcosh, h_dist
from utils.utils_dataset import create_dataset_for_basescore, create_test_for_link_prediction
from scipy import stats
from scipy.sparse import coo_matrix


RESULTS = "results"


def create_wn_dataset(dataset_name):
    df = pd.read_csv("dataset/wn_dataset/" + dataset_name + "_closure.csv")

    node_names = set(df["id1"]) | set(df["id2"])
    node_names = np.array(list(node_names))

    n_nodes = len(node_names)

    adj_mat = np.zeros((n_nodes, n_nodes))
    is_a = np.zeros((len(df), 2))

    for index, r in df.iterrows():
        u = np.where(node_names == r[0])[0]
        v = np.where(node_names == r[1])[0]
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1

        is_a[index, 0] = u
        is_a[index, 1] = v

    adj_mat = adj_mat.astype(np.int)
    is_a = is_a.astype(np.int)

    params_dataset = {
        "n_nodes": n_nodes
    }

    print(node_names)
    print(adj_mat)
    print(np.sum(adj_mat))
    print(is_a)

    data = {
        "adj_mat": adj_mat,
        "is_a": is_a
    }

    np.save("dataset/wn_dataset/" + dataset_name + "_data.npy", data)


def create_wn_dataset_link(dataset_name):
    df = pd.read_csv("dataset/wn_dataset/" + dataset_name + "_closure.csv")

    node_names = set(df["id1"]) | set(df["id2"])
    node_names = np.array(list(node_names))

    n_nodes = len(node_names)

    adj_mat = np.zeros((n_nodes, n_nodes))
    is_a = np.zeros((len(df), 2))

    for index, r in df.iterrows():
        u = np.where(node_names == r[0])[0]
        v = np.where(node_names == r[1])[0]
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1

        is_a[index, 0] = u
        is_a[index, 1] = v

    adj_mat = adj_mat.astype(np.int)
    is_a = is_a.astype(np.int)

    params_dataset = {
        "n_nodes": n_nodes
    }

    print(node_names)
    print(adj_mat)
    print(np.sum(adj_mat))
    print(is_a)

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat, params_dataset)

    data = {
        "adj_mat": coo_matrix(adj_mat),
        "positive_samples": positive_samples,
        "negative_samples": negative_samples,
        "train_graph": coo_matrix(train_graph),
        "lik_data": lik_data,
    }

    # print(data)

    os.makedirs("dataset/" + dataset_name, exist_ok=True)
    np.save('dataset/' + dataset_name + "/data.npy", data)


def is_a_score(is_a, n_dim, lorentz_table, alpha=100, print_stats=False, print_is_a_score=False):
    if print_stats:
        r = arcosh(torch.Tensor(lorentz_table[:, 0]))
        # torch.set_printoptions(edgeitems=1000)
        print(r)

    score_sum = 0
    for r in is_a:
        u = r[0]
        v = r[1]
        c_u = torch.Tensor(lorentz_table[u])
        c_v = torch.Tensor(lorentz_table[v])
        r_u = arcosh(c_u[0])
        r_v = arcosh(c_v[0])
        dst = h_dist(c_u.reshape((1, -1)), c_v.reshape((1, -1)))
        score = -(1 + alpha * (r_v - r_u)) * dst
        score_sum += score[0]
        pass

    if print_is_a_score:
        print("Dim ", n_dim, ": ", score_sum / len(is_a))
        print("nodes:", len(lorentz_table))
        print("is-a:", len(is_a))

    return score_sum / len(is_a)


def calc_metrics_realworld(device_idx, model_n_dim, dataset_name):
    if not os.path.exists("dataset/wn_dataset/" + dataset_name + "_data.npy"):
        create_wn_dataset(dataset_name)

    data = np.load("dataset/wn_dataset/" + dataset_name +
                   "_data.npy", allow_pickle=True).item()
    adj_mat = data["adj_mat"]
    is_a = data["is_a"]

    print("n_nodes:", len(adj_mat))
    print("n_edges:", np.sum(adj_mat) / 2)
    n_nodes = len(adj_mat)

    params_dataset = {
        'n_nodes': n_nodes,
        'R': np.log(n_nodes),
    }

    # data for calculating likelihood
    lik_data = create_dataset_for_basescore(
        adj_mat=adj_mat,
        n_max_samples=int((n_nodes - 1) * 0.1)
    )

    # parameter
    burn_epochs = 800
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    n_max_negatives = n_max_positives * 10
    lr_embeddings = 0.1
    lr_epoch_10 = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32 / 100  # batchサイズに対応して学習率変更
    lr_beta = 0.001
    lr_gamma = 0.001
    sigma_max = 1.0
    sigma_min = 0.001
    beta_min = 0.1
    beta_max = 10.0
    gamma_min = 0.1
    gamma_max = 10.0
    eps_1 = 1e-6
    eps_2 = 1e3
    init_range = 0.001
    perturbation = False
    # others
    loader_workers = 16
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = False

    device = "cuda:" + str(device_idx)

    # 平均次数が少なくなるように手で調整する用
    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    result = pd.DataFrame()

    ret = LinkPrediction(
        train_graph=adj_mat,
        positive_samples=None,
        negative_samples=None,
        lik_data=lik_data,
        x_lorentz=None,
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
        eps_2=eps_2,
        init_range=init_range,
        device=device,
        calc_HGG=True,
        calc_WND=True,
        calc_naive=True,
        calc_othermetrics=False,
        calc_groundtruth=False,
        perturbation=perturbation,
        loader_workers=16,
        shuffle=True,
        sparse=False
    )

    torch.save(ret["model_hgg"],
               RESULTS + "/" + dataset_name + "/result_" + str(model_n_dim) + "_hgg.pth")
    torch.save(ret["model_wnd"],
               RESULTS + "/" + dataset_name + "/result_" + str(model_n_dim) + "_wnd.pth")
    torch.save(ret["model_naive"],
               RESULTS + "/" + dataset_name + "/result_" + str(model_n_dim) + "_naive.pth")

    lorentz_table_hgg = ret["model_hgg"].get_lorentz_table()
    lorentz_table_wnd = ret["model_wnd"].get_lorentz_table()
    lorentz_table_naive = ret["model_naive"].get_lorentz_table()

    ret["is-a-score_hgg"] = is_a_score(
        is_a, model_n_dim, lorentz_table_hgg).cpu().numpy()
    ret["is-a-score_wnd"] = is_a_score(
        is_a, model_n_dim, lorentz_table_wnd).cpu().numpy()
    ret["is-a-score_naive"] = is_a_score(
        is_a, model_n_dim, lorentz_table_naive).cpu().numpy()

    ret.pop('model_hgg')
    ret.pop('model_wnd')
    ret.pop('model_naive')

    ret["model_n_dims"] = model_n_dim
    ret["n_nodes"] = params_dataset["n_nodes"]
    ret["R"] = params_dataset["R"]
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
    ret["eps_2"] = eps_2
    ret["init_range"] = init_range

    row = pd.DataFrame(ret.values(), index=ret.keys()).T

    row = row.reindex(columns=[
        "model_n_dims",
        "n_nodes",
        # "n_dim",
        "R",
        # "sigma",
        # "beta",
        "DNML_HGG",
        "AIC_HGG",
        "BIC_HGG",
        "DNML_WND",
        "AIC_WND",
        "BIC_WND",
        "AIC_naive",
        "BIC_naive",
        "is-a-score_hgg",
        "is-a-score_wnd",
        "is-a-score_naive",
        "-log p_HGG(y, z)",
        "-log p_HGG(y|z)",
        "-log p_HGG(z)",
        "-log p_WND(y, z)",
        "-log p_WND(y|z)",
        "-log p_WND(z)",
        "-log p_naive(y; z)",
        "pc_hgg_first",
        "pc_hgg_second",
        "pc_wnd_first",
        "pc_wnd_second",
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
        "eps_1",
        "eps_2",
        "init_range"
    ]
    )

    row.to_csv(RESULTS + "/" + dataset_name + "/result_" +
               str(model_n_dim) + ".csv", index=False)

if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(
        description='Hyperbolic Graph Embedding with LVM')
    parser.add_argument('dataset', help='dataset')
    parser.add_argument('n_dim', help='n_dim')
    parser.add_argument('device', help='device')

    args = parser.parse_args()
    print(args)

    device_idx = int(args.device)
    model_n_dim = int(args.n_dim)

    if int(args.dataset) == 0:
        dataset_name = "mammal"
    elif int(args.dataset) == 1:
        dataset_name = "solid"
    elif int(args.dataset) == 2:
        dataset_name = "tree"
    elif int(args.dataset) == 3:
        dataset_name = "worker"
    elif int(args.dataset) == 4:
        dataset_name = "adult"
    elif int(args.dataset) == 5:
        dataset_name = "instrument"
    elif int(args.dataset) == 6:
        dataset_name = "leader"
    elif int(args.dataset) == 7:
        dataset_name = "implement"

    print(dataset_name)

    os.makedirs(RESULTS + "/" + dataset_name + "/", exist_ok=True)

    # create_wn_dataset_link(dataset_name)
    create_wn_dataset_link("mammal")
    create_wn_dataset_link("solid")

    # calc_metrics_realworld(device_idx=device_idx,
    # model_n_dim=model_n_dim, dataset_name=dataset_name)
