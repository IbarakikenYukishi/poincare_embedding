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
from datasets import wrapped_normal_distribution, euclidean_geometric_graph
from lorentz import CV_HGG, DNML_HGG, LinkPrediction, create_test_for_link_prediction, Euclidean_vs_Hyperbolic
import torch.multiprocessing as multi
from functools import partial
from scipy.io import mmread
from scipy.sparse import coo_matrix
import os


RESULTS = "results"


def calc_hyperbolic_euclidean(n_nodes, data_type, device_idx):
    # データセット作成
    # n_nodes = 400

    print("R:", np.log(n_nodes))

    params_dataset = {
        'n_nodes': n_nodes,
        'n_dim': 8,
        'R': np.log(n_nodes),
        'sigma': 0.5,
        'beta': 0.4
    }

    # パラメータ
    burn_epochs = 800
    # burn_epochs = 5
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 50)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    lr_beta = 0.001
    lr_gamma = 0.001
    sigma_max = 10.0
    sigma_min = 0.1
    beta_min = 0.1
    beta_max = 10.0
    gamma_min = 0.1
    gamma_max = 10.0
    eps_1 = 1e-6
    # それ以外
    loader_workers = 8
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if data_type == 'WND':
        # 隣接行列
        adj_mat, x_lorentz = wrapped_normal_distribution(
            n_nodes=params_dataset['n_nodes'],
            n_dim=params_dataset['n_dim'],
            R=params_dataset['R'],
            Sigma=np.eye(params_dataset[
                         "n_dim"]) * ((np.log(params_dataset["n_nodes"]) * params_dataset["sigma"])**2),
            beta=params_dataset['beta']
        )
    elif data_type == 'Gaussian':
        adj_mat, x_lorentz = euclidean_geometric_graph(
            n_nodes=params_dataset['n_nodes'],
            n_dim=params_dataset['n_dim'],
            R=params_dataset['R'],
            Sigma=np.eye(params_dataset[
                         "n_dim"]) * ((np.log(params_dataset["n_nodes"]) * params_dataset["sigma"])**2),
            beta=params_dataset['beta']
        )

    # 平均次数が少なくなるように手で調整する用
    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    result = pd.DataFrame()

    model_n_dims = [8]

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat=adj_mat,
        params_dataset=params_dataset
    )

    # negative samplingの比率を平均次数から決定
    pos_train_graph = len(np.where(train_graph == 1)[0])
    neg_train_graph = len(np.where(train_graph == 0)[0])
    ratio = neg_train_graph / pos_train_graph
    print("ratio:", ratio)

    # ratio=10

    n_max_negatives = int(n_max_positives * ratio)
    print("n_max_negatives:", n_max_negatives)
    lr_embeddings = 0.1
    lr_epoch_10 = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32 / 100  # batchサイズに対応して学習率変更

    for model_n_dim in model_n_dims:
        ret = Euclidean_vs_Hyperbolic(
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
            lr_gamma=lr_gamma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            eps_1=eps_1,
            device=device,
            loader_workers=16,
            shuffle=True,
            sparse=False,
            calc_groundtruth=False
        )

        # torch.save(ret["model_hgg"],
        #            "temp/result_" + str(model_n_dim) + "_hgg.pth")
        torch.save(ret["model_wnd"],
                   "temp/result_" + str(model_n_dim) + "_wnd.pth")
        torch.save(ret["model_gaussian"],
                   "temp/result_" + str(model_n_dim) + "_gaussian.pth")

        ret.pop('model_wnd')
        ret.pop('model_gaussian')

        ret["data_type"] = data_type
        ret["model_n_dims"] = model_n_dim
        ret["n_nodes"] = params_dataset["n_nodes"]
        ret["n_dim"] = params_dataset["n_dim"]
        ret["R"] = params_dataset["R"]
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

        row = pd.DataFrame(ret.values(), index=ret.keys()).T

        row = row.reindex(columns=[
            "data_type",
            "model_n_dims",
            "n_nodes",
            "n_dim",
            "R",
            "sigma",
            "beta",
            "DNML_WND",
            "AIC_WND",
            "BIC_WND",
            "DNML_Gaussian",
            "AIC_Gaussian",
            "BIC_Gaussian",
            "AUC_WND",
            "AUC_Gaussian",
            "-log p_WND(y, z)",
            "-log p_WND(y|z)",
            "-log p_WND(z)",
            "-log p_Gaussian(y, z)",
            "-log p_Gaussian(y|z)",
            "-log p_Gaussian(z)",
            "pc_wnd_first",
            "pc_wnd_second",
            "pc_gaussian_first",
            "pc_gaussian_second",
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
            "eps_1"
        ]
        )

        filepath = RESULTS + "/" + data_type

        # if os.path.exists(filepath):
        #     result_previous = pd.read_csv(filepath)
        #     result = pd.concat([result_previous, row])
        #     result.to_csv(filepath, index=False)
        row.to_csv(filepath + "/result_" + str(n_nodes) + ".csv", index=False)


if __name__ == '__main__':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    import argparse

    parser = argparse.ArgumentParser(description='HGG')
    parser.add_argument('n_nodes', help='dataset')
    parser.add_argument('data_type', help='n_dim')
    parser.add_argument('device', help='device')
    args = parser.parse_args()
    print(args)

    if int(args.data_type) == 0:
        data_type = "WND"
    elif int(args.data_type) == 1:
        data_type = "Gaussian"

    os.makedirs(RESULTS + "/" + data_type, exist_ok=True)

    calc_hyperbolic_euclidean(n_nodes=int(args.n_nodes),
                              data_type=data_type, device_idx=int(args.device))
