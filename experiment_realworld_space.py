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
from space_selection import LinkPrediction
from utils.utils_dataset import create_test_for_link_prediction
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
        'R': np.log(n_nodes) + 4,
    }

    # パラメータ
    burn_epochs = 800
    # burn_epochs = 15
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    n_max_negatives = n_max_positives * 10
    lr_embeddings = 0.1
    lr_epoch_10 = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32 / 100  # batchサイズに対応して学習率変更
    # lr_kappa = 0.004
    # lr_gamma = 0.004
    lr_kappa = 1.0
    lr_gamma = 1.0
    # lr_gamma = 0.01
    sigma_max = 100.0
    sigma_min = 0.2
    k_max = 100
    gamma_min = 0.1
    gamma_max = 10.0
    init_range = 0.001
    perturbation = True
    change_learning_rate = 100
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
        train_graph=train_graph,
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        lik_data=lik_data,
        params_dataset=params_dataset,
        model_n_dim=model_n_dim,
        burn_epochs=burn_epochs,
        burn_batch_size=burn_batch_size,
        n_max_positives=n_max_positives,
        n_max_negatives=n_max_negatives,
        lr_embeddings=lr_embeddings,
        lr_epoch_10=lr_epoch_10,
        lr_kappa=lr_kappa,
        lr_gamma=lr_gamma,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        k_max=k_max,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        init_range=init_range,
        device=device,
        calc_lorentz_latent=True,
        calc_euclidean_latent=True,
        calc_spherical_latent=True,
        calc_lorentz_naive=True,
        calc_euclidean_naive=True,
        calc_spherical_naive=True,
        calc_othermetrics=True,
        perturbation=perturbation,
        change_learning_rate=change_learning_rate,
        loader_workers=16,
        shuffle=True,
        sparse=False
    )
    torch.save(ret["model_lorentz_latent"], RESULTS + "/" + dataset_name +
               "/result_" + str(model_n_dim) + "_lorentz_latent.pth")
    torch.save(ret["model_euclidean_latent"], RESULTS + "/" + dataset_name +
               "/result_" + str(model_n_dim) + "_euclidean_latent.pth")
    torch.save(ret["model_spherical_latent"], RESULTS + "/" + dataset_name +
               "/result_" + str(model_n_dim) + "_spherical_latent.pth")
    torch.save(ret["model_lorentz_naive"], RESULTS + "/" + dataset_name +
               "/result_" + str(model_n_dim) + "_lorentz_naive.pth")
    torch.save(ret["model_euclidean_naive"], RESULTS + "/" + dataset_name +
               "/result_" + str(model_n_dim) + "_euclidean_naive.pth")
    torch.save(ret["model_spherical_naive"], RESULTS + "/" + dataset_name +
               "/result_" + str(model_n_dim) + "_spherical_naive.pth")

    ret.pop('model_lorentz_latent')
    ret.pop('model_euclidean_latent')
    ret.pop('model_spherical_latent')
    ret.pop('model_lorentz_naive')
    ret.pop('model_euclidean_naive')
    ret.pop('model_spherical_naive')

    ret["model_n_dims"] = model_n_dim
    ret["n_nodes"] = params_dataset["n_nodes"]
    ret["R"] = params_dataset["R"]
    ret["burn_epochs"] = burn_epochs
    ret["burn_batch_size"] = burn_batch_size
    ret["n_max_positives"] = n_max_positives
    ret["n_max_negatives"] = n_max_negatives
    ret["lr_embeddings"] = lr_embeddings
    ret["lr_epoch_10"] = lr_epoch_10
    ret["lr_kappa"] = lr_kappa
    ret["lr_gamma"] = lr_gamma
    ret["sigma_max"] = sigma_max
    ret["sigma_min"] = sigma_min
    ret["k_max"] = k_max
    ret["gamma_max"] = gamma_max
    ret["gamma_min"] = gamma_min
    ret["init_range"] = init_range

    row = pd.DataFrame(ret.values(), index=ret.keys()).T

    row = row.reindex(columns=[
        "model_n_dims",
        "n_nodes",
        "R",
        "DNML_lorentz_latent",
        "DNML_euclidean_latent",
        "DNML_spherical_latent",
        "AIC_lorentz_latent",
        "AIC_euclidean_latent",
        "AIC_spherical_latent",
        "AIC_lorentz_naive",
        "AIC_euclidean_naive",
        "AIC_spherical_naive",
        "BIC_lorentz_latent",
        "BIC_euclidean_latent",
        "BIC_spherical_latent",
        "BIC_lorentz_naive",
        "BIC_euclidean_naive",
        "BIC_spherical_naive",
        "AUC_lorentz_latent",
        "AUC_euclidean_latent",
        "AUC_spherical_latent",
        "AUC_lorentz_naive",
        "AUC_euclidean_naive",
        "AUC_spherical_naive",
        "-log p_lorentz_latent(y, z)",
        "-log p_lorentz_latent(y|z)",
        "-log p_lorentz_latent(z)",
        "-log p_lorentz_naive(y|z)",
        "-log p_euclidean_latent(y, z)",
        "-log p_euclidean_latent(y|z)",
        "-log p_euclidean_latent(z)",
        "-log p_euclidean_naive(y|z)",
        "-log p_spherical_latent(y, z)",
        "-log p_spherical_latent(y|z)",
        "-log p_spherical_latent(z)",
        "-log p_spherical_naive(y|z)",
        "pc_lorentz_first",
        "pc_lorentz_second",
        "pc_euclidean_first",
        "pc_euclidean_second",
        "pc_spherical_first",
        "pc_spherical_second",
        "burn_epochs",
        "n_max_positives",
        "n_max_negatives",
        "lr_embeddings",
        "lr_epoch_10",
        "lr_kappa",
        "lr_gamma",
        "sigma_max",
        "sigma_min",
        "k_max",
        "gamma_max",
        "gamma_min",
        "init_range"
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

    # if int(args.dataset) == 0:
    #     dataset_name = "ca-AstroPh"
    # elif int(args.dataset) == 1:
    #     dataset_name = "ca-CondMat"
    # elif int(args.dataset) == 2:
    #     dataset_name = "ca-GrQc"
    # elif int(args.dataset) == 3:
    #     dataset_name = "ca-HepPh"
    # elif int(args.dataset) == 4:
    #     dataset_name = "airport"
    # elif int(args.dataset) == 5:
    #     dataset_name = "cora"
    # elif int(args.dataset) == 6:
    #     dataset_name = "pubmed"
    # elif int(args.dataset) == 7:
    #     dataset_name = "bio-yeast-protein-inter"
    # elif int(args.dataset) == 8:
    #     dataset_name = "mammal"
    # elif int(args.dataset) == 9:
    #     dataset_name = "solid"
    # elif int(args.dataset) == 10:
    #     dataset_name = "tree"
    # elif int(args.dataset) == 11:
    #     dataset_name = "worker"
    # elif int(args.dataset) == 12:
    #     dataset_name = "adult"
    # elif int(args.dataset) == 13:
    #     dataset_name = "instrument"
    # elif int(args.dataset) == 14:
    #     dataset_name = "leader"
    # elif int(args.dataset) == 15:
    #     dataset_name = "implement"
    # elif int(args.dataset) == 16:
    #     dataset_name = "inf-euroroad"
    # elif int(args.dataset) == 17:
    #     dataset_name = "inf-power"

    if int(args.dataset) == 0:
        dataset_name = "ca-AstroPh"
    elif int(args.dataset) == 1:
        dataset_name = "ca-HepPh"
    elif int(args.dataset) == 2:
        dataset_name = "airport"
    elif int(args.dataset) == 3:
        dataset_name = "mammal"
    elif int(args.dataset) == 4:
        dataset_name = "solid"




    os.makedirs(RESULTS + "/" + dataset_name, exist_ok=True)

    if not os.path.exists('dataset/' + dataset_name + "/data.npy"):
        data_generation(dataset_name)

    calc_metrics_realworld(dataset_name=dataset_name, device_idx=int(
        args.device), model_n_dim=int(args.n_dim))
