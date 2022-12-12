import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import torch
import numpy as np
import pandas as pd
import gc
import time
from copy import deepcopy
from torch.utils.data import DataLoader
from lorentz import CV_HGG, DNML_HGG, LinkPrediction, create_test_for_link_prediction
import torch.multiprocessing as multi
from functools import partial
from scipy.io import mmread
from scipy.sparse import coo_matrix
import os


RESULTS = "results"


def calc_metrics_realworld(dataset_name, device_idx, model_n_dim):
    data = np.load('dataset/' + dataset_name +
                   '/data.npy', allow_pickle=True).item()
    adj_mat = data["adj_mat"].toarray()
    positive_samples = data["positive_samples"]
    negative_samples = data["negative_samples"]
    train_graph = data["train_graph"].toarray()
    lik_data = data["lik_data"]

    print("n_nodes:", len(adj_mat))
    print("n_edges:", np.sum(adj_mat))
    n_nodes = len(adj_mat)

    params_dataset = {
        'n_nodes': n_nodes,
        'R': np.log(n_nodes),
    }

    # パラメータ
    burn_epochs = 800
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    n_max_negatives = n_max_positives * 10
    lr_embeddings = 0.1
    lr_epoch_10 = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32 / 100  # batchサイズに対応して学習率変更
    lr_beta = 0.001
    # lr_sigma = 0.001
    sigma_max = 1.0
    sigma_min = 0.1
    beta_min = 0.1
    beta_max = 10.0
    eps_1 = 1e-6
    # それ以外
    loader_workers = 16
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = False

    device = "cuda:" + str(device_idx)

    # 平均次数が少なくなるように手で調整する用
    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    result = pd.DataFrame()
    # basescore_y_and_z_list = []
    # basescore_y_given_z_list = []
    # basescore_z_list = []
    # basescore_y_given_z_naive_list = []
    # DNML_codelength_list = []
    # pc_first_list = []
    # pc_second_list = []
    # AIC_naive_list = []
    # BIC_naive_list = []
    # AIC_naive_from_latent_list = []
    # BIC_naive_from_latent_list = []
    # CV_score_list = []
    # AUC_latent_list = []
    # AUC_naive_list = []

    ret = LinkPrediction(
        adj_mat=adj_mat,
        train_graph=train_graph,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
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
        # lr_sigma=lr_sigma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta_min=beta_min,
        beta_max=beta_max,
        eps_1=eps_1,
        device=device,
        loader_workers=16,
        shuffle=True,
        sparse=False,
        calc_groundtruth=False
    )
    torch.save(ret["model_hgg"].state_dict(), RESULTS + "/" + dataset_name +
               "/result_" + str(model_n_dim) + "_hgg.pth")
    torch.save(ret["model_wnd"].state_dict(), RESULTS + "/" + dataset_name +
               "/result_" + str(model_n_dim) + "_wnd.pth")
    torch.save(ret["model_naive"].state_dict(), RESULTS + "/" + dataset_name +
               "/result_" + str(model_n_dim) + "_naive.pth")

    ret.pop('model_hgg')
    ret.pop('model_wnd')
    ret.pop('model_naive')

    ret["model_n_dims"] = model_n_dim
    ret["n_nodes"] = params_dataset["n_nodes"]
    # ret["n_dim"] = params_dataset["n_dim"]
    ret["R"] = params_dataset["R"]
    # ret["sigma"] = params_dataset["sigma"]
    # ret["beta"] = params_dataset["beta"]
    ret["burn_epochs"] = burn_epochs
    ret["burn_batch_size"] = burn_batch_size
    ret["n_max_positives"] = n_max_positives
    ret["n_max_negatives"] = n_max_negatives
    ret["lr_embeddings"] = lr_embeddings
    ret["lr_epoch_10"] = lr_epoch_10
    ret["lr_beta"] = lr_beta
    ret["sigma_max"] = sigma_max
    ret["sigma_min"] = sigma_min
    ret["beta_max"] = beta_max
    ret["beta_min"] = beta_min
    ret["eps_1"] = eps_1

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
        "AUC_HGG",
        "AUC_WND",
        "AUC_naive",
        "AUC_GT",
        "cor_hgg",
        "cor_wnd",
        "cor_naive",
        "-log p_HGG(y, z)",
        "-log p_HGG(y|z)",
        "-log p_HGG(z)",
        "-log p_WND(y, z)",
        "-log p_WND(y|z)",
        "-log p_WND(z)",
        "-log p_naive(y; z)",
        "pc_first",
        "pc_second",
        "burn_epochs",
        "n_max_positives",
        "n_max_negatives",
        "lr_embeddings",
        "lr_epoch_10",
        "lr_beta",
        "sigma_max",
        "sigma_min",
        "beta_max",
        "beta_min",
        "eps_1"
    ]
    )

    row.to_csv(RESULTS + "/" + dataset_name + "/result_" +
               str(model_n_dim) + ".csv", index=False)


def data_generation(dataset_name):
    # データセット生成
    edges_ids = np.loadtxt('dataset/' + dataset_name +
                           "/" + dataset_name + ".txt", dtype=int)

    ids_all = set(edges_ids[:, 0]) & set(edges_ids[:, 1])
    n_nodes = len(ids_all)
    adj_mat = np.zeros((n_nodes, n_nodes))
    ids_all = list(ids_all)

    for i in range(len(edges_ids)):
        print(i)
        u = np.where(ids_all == edges_ids[i, 0])[0]
        v = np.where(ids_all == edges_ids[i, 1])[0]
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1

    adj_mat = adj_mat.astype(np.int)
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


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(description='HGG')
    parser.add_argument('dataset', help='dataset')
    parser.add_argument('n_dim', help='n_dim')
    parser.add_argument('device', help='device')
    args = parser.parse_args()
    print(args)

    if int(args.dataset) == 0:
        dataset_name = "ca-AstroPh"
    elif int(args.dataset) == 1:
        dataset_name = "ca-HepPh"
    elif int(args.dataset) == 2:
        dataset_name = "ca-GrQc"
    elif int(args.dataset) == 3:
        dataset_name = "ca-CondMat"

    os.makedirs(RESULTS + "/" + dataset_name, exist_ok=True)

    if not os.path.exists('dataset/' + dataset_name + "/data.npy"):
        data_generation(dataset_name)

    calc_metrics_realworld(dataset_name=dataset_name, device_idx=int(
        args.device), model_n_dim=int(args.n_dim))
