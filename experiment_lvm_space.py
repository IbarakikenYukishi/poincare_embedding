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
        # burn_epochs = 20
        burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
        n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
        n_max_negatives = n_max_positives * 10
        lr_embeddings = 0.1
        lr_epoch_10 = 10.0 * \
            (burn_batch_size * (n_max_positives + n_max_negatives)) / \
            32 / 100  # batchサイズに対応して学習率変更
        if dataset_name == "euclidean":
            lr_kappa = 0.004
            lr_gamma = 0.004
        else:
            lr_kappa = 1.0
            lr_gamma = 1.0
        sigma_max = 100.0
        sigma_min = 0.2
        k_max = 100
        gamma_min = 0.1
        gamma_max = 10.0
        init_range = 0.001
        perturbation = True
        change_learning_rate = 100
        # change_learning_rate = 10
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
                gamma_min=gamma_min,
                gamma_max=gamma_max,
                k_max=k_max,
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
                sparse=False,
            )

            torch.save(ret["model_lorentz_latent"], RESULTS + "/" + dataset_name + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_lorentz_latent.pth")
            torch.save(ret["model_euclidean_latent"], RESULTS + "/" + dataset_name + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_euclidean_latent.pth")
            torch.save(ret["model_spherical_latent"], RESULTS + "/" + dataset_name + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_spherical_latent.pth")
            torch.save(ret["model_lorentz_naive"], RESULTS + "/" + dataset_name + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_lorentz_naive.pth")
            torch.save(ret["model_euclidean_naive"], RESULTS + "/" + dataset_name + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_euclidean_naive.pth")
            torch.save(ret["model_spherical_naive"], RESULTS + "/" + dataset_name + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_spherical_naive.pth")

            ret.pop('model_lorentz_latent')
            ret.pop('model_euclidean_latent')
            ret.pop('model_spherical_latent')
            ret.pop('model_lorentz_naive')
            ret.pop('model_euclidean_naive')
            ret.pop('model_spherical_naive')

            ret["model_n_dims"] = model_n_dim
            ret["n_nodes"] = params_dataset["n_nodes"]
            ret["n_dim"] = params_dataset["n_dim"]
            ret["R"] = params_dataset["R"]
            # ret["sigma"] = params_dataset["sigma"]
            ret["k"] = params_dataset["k"]
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
            ret["perturbation"] = perturbation
            ret["change_learning_rate"] = change_learning_rate

            row = pd.DataFrame(ret.values(), index=ret.keys()).T

            row = row.reindex(columns=[
                "model_n_dims",
                "n_nodes",
                "n_dim",
                "R",
                # "sigma",
                # "beta",
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
                "init_range",
                "perturbation",
                "change_learning_rate"
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

    # if args.n_nodes == "0":
    #     n_nodes_list = [400, 800, 1600]
    # elif args.n_nodes == "1":
    #     n_nodes_list = [3200]
    # elif args.n_nodes == "2":
    #     n_nodes_list = [6400]

    if args.n_nodes == "0":
        n_nodes_list = [400]
    elif args.n_nodes == "1":
        n_nodes_list = [800]
    elif args.n_nodes == "2":
        n_nodes_list = [1600]
    elif args.n_nodes == "3":
        n_nodes_list = [3200]

    # model_n_dims = [2, 4, 8, 16, 32, 64]
    model_n_dims = [2, 4, 8, 16, 32]

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
