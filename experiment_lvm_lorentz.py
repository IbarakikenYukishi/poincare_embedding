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
from lorentz import LinkPrediction
import torch.multiprocessing as multi
from functools import partial
from scipy.sparse import coo_matrix
import os

RESULTS = "results"


def calc_metrics(
    dataset_name,
    partition_idx,
    n_dim,
    n_nodes,
    n_graphs,
    n_partitions,
    n_devices,
    model_n_dims
):
    for n_graph in range(int(n_graphs * partition_idx / n_partitions), int(n_graphs * (partition_idx + 1) / n_partitions)):
        print(n_graph)

        dataset = np.load('dataset/' + dataset_name + '/dim_' + str(n_dim) + '/graph_' + str(n_nodes) + '_' + str(n_graph) +
                          '.npy', allow_pickle='TRUE').item()  # object型なので、itemを付けないと辞書として使えない。
        adj_mat = dataset["adj_mat"]
        adj_mat = adj_mat.toarray()
        params_dataset = dataset["params_adj_mat"]
        positive_samples = dataset["positive_samples"]
        negative_samples = dataset["negative_samples"]
        train_graph = dataset["train_graph"]
        train_graph = train_graph.toarray()
        lik_data = dataset["lik_data"]
        x_lorentz = dataset["x_e"]

        # パラメータ
        burn_epochs = 800
        # burn_epochs = 2
        burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
        n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
        n_max_negatives = n_max_positives * 10
        lr_embeddings = 0.1
        lr_epoch_10 = 10.0 * \
            (burn_batch_size * (n_max_positives + n_max_negatives)) / \
            32 / 100  # batchサイズに対応して学習率変更
        lr_beta = 0.001
        lr_sigma = 0.001
        lr_gamma = 0.001
        sigma_max = 1.0
        sigma_min = 0.1
        beta_max = 10.0
        beta_min = 0.1
        gamma_min = 0.1
        gamma_max = 10.0
        eps_1 = 1e-6
        eps_2 = 1e3
        # それ以外
        loader_workers = 16
        print("loader_workers: ", loader_workers)
        shuffle = True
        sparse = False

        device = "cuda:" + str(partition_idx % n_devices)

        # 平均次数が少なくなるように手で調整する用
        print('average degree:', np.sum(adj_mat) / len(adj_mat))

        result = pd.DataFrame()

        for model_n_dim in model_n_dims:
            if dataset_name == "HGG":
                calc_HGG = True
                calc_WND = False
            elif dataset_name == "WND":
                calc_HGG = False,
                calc_WND = True,

            ret = LinkPrediction(
                train_graph=train_graph,
                positive_samples=positive_samples,
                negative_samples=negative_samples,
                lik_data=lik_data,
                x_lorentz=x_lorentz,
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
                device=device,
                calc_HGG=calc_HGG,
                calc_WND=calc_WND,
                calc_naive=True,
                calc_othermetrics=True,
                calc_groundtruth=True,
                loader_workers=16,
                shuffle=True,
                sparse=False,
            )

            torch.save(ret["model_hgg"], RESULTS + "/" + dataset_name + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_hgg.pth")
            torch.save(ret["model_naive"], RESULTS + "/" + dataset_name + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_naive.pth")

            ret.pop('model_hgg')
            ret.pop('model_wnd')
            ret.pop('model_naive')

            ret["model_n_dims"] = model_n_dim
            ret["n_nodes"] = params_dataset["n_nodes"]
            ret["n_dim"] = params_dataset["n_dim"]
            ret["R"] = params_dataset["R"]
            if dataset_name == "HGG":
                ret["sigma"] = params_dataset["sigma"]
            ret["beta"] = params_dataset["beta"]
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

            row = pd.DataFrame(ret.values(), index=ret.keys()).T

            if dataset_name == "HGG":
                row = row.reindex(columns=[
                    "model_n_dims",
                    "n_nodes",
                    "n_dim",
                    "R",
                    "sigma",
                    "beta",
                    "DNML_HGG",
                    "AIC_HGG",
                    "BIC_HGG",
                    # "DNML_WND",
                    # "AIC_WND",
                    # "BIC_WND",
                    "AIC_naive",
                    "BIC_naive",
                    "AUC_HGG",
                    # "AUC_WND",
                    "AUC_naive",
                    "AUC_GT",
                    "cor_hgg",
                    # "cor_wnd",
                    "cor_naive",
                    "-log p_HGG(y, z)",
                    "-log p_HGG(y|z)",
                    "-log p_HGG(z)",
                    # "-log p_WND(y, z)",
                    # "-log p_WND(y|z)",
                    # "-log p_WND(z)",
                    "-log p_naive(y; z)",
                    "pc_hgg_first",
                    "pc_hgg_second",
                    # "pc_wnd_first",
                    # "pc_wnd_second",
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
                    # "eps_1",
                    # "eps_2"
                ]
                )
            if dataset_name == "WND":
                row = row.reindex(columns=[
                    "model_n_dims",
                    "n_nodes",
                    "n_dim",
                    "R",
                    # "sigma",
                    "beta",
                    # "DNML_HGG",
                    # "AIC_HGG",
                    # "BIC_HGG",
                    "DNML_WND",
                    "AIC_WND",
                    "BIC_WND",
                    "AIC_naive",
                    "BIC_naive",
                    # "AUC_HGG",
                    "AUC_WND",
                    "AUC_naive",
                    "AUC_GT",
                    # "cor_hgg",
                    "cor_wnd",
                    "cor_naive",
                    # "-log p_HGG(y, z)",
                    # "-log p_HGG(y|z)",
                    # "-log p_HGG(z)",
                    "-log p_WND(y, z)",
                    "-log p_WND(y|z)",
                    "-log p_WND(z)",
                    "-log p_naive(y; z)",
                    # "pc_hgg_first",
                    # "pc_hgg_second",
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
                    "eps_2"
                ]
                )

            result = pd.concat([result, row])

        result.to_csv(RESULTS + "/" + dataset_name + "/dim_" + str(n_dim) + "/result_" + str(n_nodes) +
                      "_" + str(n_graph) + ".csv", index=False)


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(description='HGG')
    parser.add_argument('dataset_name', help='dataset_name')
    parser.add_argument('n_nodes', help='n_nodes')
    parser.add_argument('n_dim', help='n_dim')
    parser.add_argument('partition', help='partition')
    args = parser.parse_args()
    print(args)

    # if args.dataset_name == "0":
    #     dataset_name = "HGG"
    # elif args.dataset_name == "1":
    #     dataset_name = "WND"

    if args.n_nodes == "0":
        n_nodes_list = [400, 800, 1600, 3200, 6400]
    elif args.n_nodes == "1":
        n_nodes_list = [12800]

    model_n_dims = [2, 4, 8, 16, 32, 64]

    n_partitions = 12
    n_devices = 4
    n_graphs = 12

    os.makedirs(RESULTS + "/" + args.dataset_name + "/dim_" +
                args.n_dim + "/", exist_ok=True)

    for n_nodes in n_nodes_list:
        calc_metrics(
            dataset_name=args.dataset_name,
            partition_idx=int(args.partition),
            n_dim=int(args.n_dim),
            n_nodes=n_nodes,
            n_graphs=n_graphs,
            n_partitions=n_partitions,
            n_devices=n_devices,
            model_n_dims=model_n_dims
        )
