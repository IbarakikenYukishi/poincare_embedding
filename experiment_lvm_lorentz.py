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

        dataset = np.load('dataset/dim_' + str(n_dim) + '/graph_' + str(n_nodes) + '_' + str(n_graph) +
                          '.npy', allow_pickle='TRUE').item()  # object型なので、itemを付けないと辞書として使えない。
        adj_mat = dataset["adj_mat"]
        # print(adj_mat)
        adj_mat = adj_mat.toarray()
        # print(adj_mat)
        params_dataset = dataset["params_adj_mat"]
        positive_samples = dataset["positive_samples"]
        negative_samples = dataset["negative_samples"]
        train_graph = dataset["train_graph"]
        train_graph = train_graph.toarray()
        lik_data = dataset["lik_data"]
        x_lorentz = dataset["x_e"]

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
        lr_sigma = 0.001
        sigma_max = 1.0
        sigma_min = 0.1
        beta_max = 10.0
        beta_min = 0.1

        # それ以外
        loader_workers = 16
        print("loader_workers: ", loader_workers)
        shuffle = True
        sparse = False

        device = "cuda:" + str(partition_idx % n_devices)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 平均次数が少なくなるように手で調整する用
        print('average degree:', np.sum(adj_mat) / len(adj_mat))

        result = pd.DataFrame()
        basescore_y_and_z_list = []
        basescore_y_given_z_list = []
        basescore_z_list = []
        basescore_y_given_z_naive_list = []
        DNML_codelength_list = []
        pc_first_list = []
        pc_second_list = []
        AIC_naive_list = []
        BIC_naive_list = []
        AIC_naive_from_latent_list = []
        BIC_naive_from_latent_list = []
        CV_score_list = []
        AUC_latent_list = []
        AUC_naive_list = []
        GT_AUC_list = []
        cor_latent_list = []
        cor_naive_list = []

        for model_n_dim in model_n_dims:
            ret = LinkPrediction(
                adj_mat=adj_mat,
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
                lr_sigma=lr_sigma,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                beta_min=beta_min,
                beta_max=beta_max,
                device=device,
                loader_workers=16,
                shuffle=True,
                sparse=False,
                calc_groundtruth=True
            )
            basescore_y_and_z_list.append(ret["basescore_y_and_z"])
            basescore_y_given_z_list.append(ret["basescore_y_given_z"])
            basescore_z_list.append(ret["basescore_z"])
            basescore_y_given_z_naive_list.append(
                ret["basescore_y_given_z_naive"])
            DNML_codelength_list.append(ret["DNML_codelength"])
            pc_first_list.append(ret["pc_first"])
            pc_second_list.append(ret["pc_second"])
            AIC_naive_list.append(ret["AIC_naive"])
            BIC_naive_list.append(ret["BIC_naive"])
            AIC_naive_from_latent_list.append(ret["AIC_naive_from_latent"])
            BIC_naive_from_latent_list.append(ret["BIC_naive_from_latent"])
            AUC_latent_list.append(ret["AUC_latent"])
            AUC_naive_list.append(ret["AUC_naive"])
            GT_AUC_list.append(ret["GT_AUC"])
            cor_latent_list.append(ret["cor_latent"])
            cor_naive_list.append(ret["cor_naive"])
            torch.save(ret["model_latent"].state_dict(), RESULTS + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_latent.pth")
            torch.save(ret["model_naive"].state_dict(), RESULTS + "/dim_" + str(n_dim) + "/result_" + str(model_n_dim) + "_" + str(n_nodes) +
                       "_" + str(n_graph) + "_naive.pth")

        result["model_n_dims"] = model_n_dims
        result["n_nodes"] = params_dataset["n_nodes"]
        result["n_dim"] = params_dataset["n_dim"]
        result["R"] = params_dataset["R"]
        result["sigma"] = params_dataset["sigma"]
        result["beta"] = params_dataset["beta"]
        result["DNML_codelength"] = DNML_codelength_list
        result["AIC_naive"] = AIC_naive_list
        result["BIC_naive"] = BIC_naive_list
        result["AIC_naive_from_latent"] = AIC_naive_from_latent_list
        result["BIC_naive_from_latent"] = BIC_naive_from_latent_list
        result["AUC_latent"] = AUC_latent_list
        result["AUC_naive"] = AUC_naive_list
        result["GT_AUC"] = GT_AUC_list
        result["cor_latent"] = cor_latent_list
        result["cor_naive"] = cor_naive_list
        result["basescore_y_and_z"] = basescore_y_and_z_list
        result["basescore_y_given_z"] = basescore_y_given_z_list
        result["basescore_z"] = basescore_z_list
        result["basescore_y_given_z_naive"] = basescore_y_given_z_naive_list
        result["pc_first"] = pc_first_list
        result["pc_second"] = pc_second_list
        result["burn_epochs"] = burn_epochs
        result["burn_batch_size"] = burn_batch_size
        result["n_max_positives"] = n_max_positives
        result["n_max_negatives"] = n_max_negatives
        result["lr_embeddings"] = lr_embeddings
        result["lr_epoch_10"] = lr_epoch_10
        result["lr_beta"] = lr_beta
        result["lr_sigma"] = lr_sigma
        result["sigma_max"] = sigma_max
        result["sigma_min"] = sigma_min
        result["beta_max"] = beta_max
        result["beta_min"] = beta_min

        result.to_csv(RESULTS + "/dim_" + str(n_dim) + "/result_" + str(n_nodes) +
                      "_" + str(n_graph) + ".csv", index=False)


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(description='HGG')
    parser.add_argument('n_dim', help='n_dim')
    parser.add_argument('partition', help='partition')
    args = parser.parse_args()
    print(args)

    n_nodes_list = [400, 800, 1600, 3200, 6400]
    model_n_dims = [2, 4, 8, 16, 32, 64]

    n_partitions = 12
    n_devices = 4
    n_graphs = 12

    # if int(args.n_dim) == 8:
    #     model_n_dims = [2, 4, 8, 16]
    # elif int(args.n_dim) == 16:
    #     model_n_dims = [2, 4, 8, 16, 32]
    os.makedirs(RESULTS + "/dim_" + args.n_dim + "/", exist_ok=True)

    for n_nodes in n_nodes_list:
        calc_metrics(
            partition_idx=int(args.partition),
            n_dim=int(args.n_dim),
            n_nodes=n_nodes,
            n_graphs=n_graphs,
            n_partitions=n_partitions,
            n_devices=n_devices,
            model_n_dims=model_n_dims
        )
