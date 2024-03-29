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
# from datasets import hyperbolic_geometric_sgraph
# from embed import create_dataset, get_unobserved, Graph, SamplingGraph, RSGD, Poincare, calc_lik_pc_cpu, calc_lik_pc_gpu
from embed_lvm import CV_HGG, DNML_HGG
import torch.multiprocessing as multi
from functools import partial


def calc_metrics(device_idx, n_dim, n_nodes, n_graphs, n_devices, model_n_dims):

    for n_graph in range(int(n_graphs * device_idx / n_devices), int(n_graphs * (device_idx + 1) / n_devices)):

        dataset = np.load('dataset/dim_' + str(n_dim) + '/graph_' + str(n_nodes) + '_' + str(n_graph) +
                          '.npy', allow_pickle='TRUE').item()  # object型なので、itemを付けないと辞書として使えない。
        adj_mat = dataset["adj_mat"]
        params_dataset = dataset["params_adj_mat"]

        # パラメータ
        burn_epochs = 200
        burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
        n_max_positives = int(params_dataset["n_nodes"] * 0.02)
        n_max_negatives = n_max_positives * 10
        learning_rate = 10.0 * \
            (burn_batch_size * (n_max_positives + n_max_negatives)) / \
            32  # batchサイズに対応して学習率変更
        sigma_max = 1.0
        sigma_min = 0.001
        beta_min = 0.1
        beta_max = 10.0
        # それ以外
        loader_workers = 8
        print("loader_workers: ", loader_workers)
        shuffle = True
        sparse = False

        device = "cuda:" + str(device_idx)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 平均次数が少なくなるように手で調整する用
        print('average degree:', np.sum(adj_mat) / len(adj_mat))

        result = pd.DataFrame()
        basescore_y_and_z_list = []
        basescore_y_given_z_list = []
        DNML_codelength_list = []
        AIC_naive_list = []
        BIC_naive_list = []
        CV_score_list = []

        for model_n_dim in model_n_dims:
            # basescore_y_and_z, basescore_y_given_z, DNML_codelength, AIC_naive, BIC_naive = DNML_HGG(
            #     adj_mat=adj_mat,
            #     params_dataset=params_dataset,
            #     model_n_dim=model_n_dim,
            #     burn_epochs=burn_epochs,
            #     burn_batch_size=burn_batch_size,
            #     n_max_positives=n_max_positives,
            #     n_max_negatives=n_max_negatives,
            #     learning_rate=learning_rate,
            #     sigma_min=sigma_min,
            #     sigma_max=sigma_max,
            #     beta_min=beta_min,
            #     beta_max=beta_max,
            #     device=device,
            #     loader_workers=16,
            #     shuffle=True,
            #     sparse=False
            # )
            # basescore_y_and_z_list.append(basescore_y_and_z)
            # basescore_y_given_z_list.append(basescore_y_given_z)
            # DNML_codelength_list.append(DNML_codelength)
            # AIC_naive_list.append(AIC_naive)
            # BIC_naive_list.append(BIC_naive)

            CV_score = CV_HGG(
                adj_mat=adj_mat,
                params_dataset=params_dataset,
                model_n_dim=model_n_dim,
                burn_epochs=burn_epochs,
                burn_batch_size=burn_batch_size,
                n_max_positives=n_max_positives,
                n_max_negatives=n_max_negatives,
                learning_rate=learning_rate,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                beta_min=beta_min,
                beta_max=beta_max,
                device=device,
                k_folds=5,
                loader_workers=16,
                shuffle=True,
                sparse=False
            )
            CV_score_list.append(CV_score)

        result["model_n_dims"] = model_n_dims
        # result["DNML_codelength"] = DNML_codelength_list
        # result["AIC_naive"] = AIC_naive_list
        # result["BIC_naive"] = BIC_naive_list
        result["CV_score"] = CV_score_list
        # result["basescore_y_and_z"] = basescore_y_and_z_list
        # result["basescore_y_given_z"] = basescore_y_given_z_list

        result.to_csv("results/result_" + str(n_dim) + "_" + str(n_nodes) +
                      "_" + str(n_graph) + ".csv", index=False)


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(description='HGG')
    parser.add_argument('n_dim', help='n_dim')
    parser.add_argument('device', help='device')
    args = parser.parse_args()
    print(args)

    # HGG-5
    n_nodes_list = [100, 200, 400, 800, 1600]
    model_n_dims = [2, 3, 4, 5, 6, 7, 8]

    # HGG-50
    # n_nodes_list = [200, 400, 800, 1600, 3200]
    # model_n_dims = [20, 30, 40, 50, 60, 70, 80]

    for n_nodes in n_nodes_list:

        calc_metrics(device_idx=int(args.device), n_dim=int(args.n_dim),
                     n_nodes=n_nodes, n_graphs=10, n_devices=4, model_n_dims=model_n_dims)
