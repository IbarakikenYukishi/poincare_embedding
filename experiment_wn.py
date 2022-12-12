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
from lorentz import arcosh, h_dist
import torch.multiprocessing as multi
from functools import partial
from scipy.io import mmread
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import os


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


def is_a_score(is_a, n_dim, lorentz_table, alpha=1000):

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
        # print(score)
        score_sum += score[0]
        pass

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

    # adj_mat = np.load(RESULTS + "/" + dataset_name + "/adj_mat.npy")
    # is_a = np.load(RESULTS + "/" + dataset_name + "/is_a.npy")

    print("n_nodes:", len(adj_mat))
    print("n_edges:", np.sum(adj_mat) / 2)
    n_nodes = len(adj_mat)

    params_dataset = {
        'n_nodes': n_nodes,
        'R': np.log(n_nodes),
    }

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
    sigma_max = 1.0
    sigma_min = 0.001
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
    # basescore_y_given_z_naive_list = []
    # basescore_z_list = []
    # DNML_codelength_list = []
    # pc_first_list = []
    # pc_second_list = []
    # AIC_naive_list = []
    # BIC_naive_list = []
    # AIC_naive_from_latent_list = []
    # BIC_naive_from_latent_list = []
    # CV_score_list = []
    # AUC_list = []
    # is_a_score_naive_list = []
    # is_a_score_latent_list = []

    # for model_n_dim in model_n_dims:
    ret = DNML_HGG(
        adj_mat=adj_mat,
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
    )
    torch.save(ret["model_hgg"].state_dict(),
               RESULTS + "/" + dataset_name + "/result_" + str(model_n_dim) + "_hgg.pth")
    torch.save(ret["model_wnd"].state_dict(),
               RESULTS + "/" + dataset_name + "/result_" + str(model_n_dim) + "_wnd.pth")
    torch.save(ret["model_naive"].state_dict(),
               RESULTS + "/" + dataset_name + "/result_" + str(model_n_dim) + "_naive.pth")

    # torch.save(ret["model_wnd"].state_dict(),
    #            "temp/result_" + str(model_n_dim) + "_wnd.pth")
    # torch.save(ret["model_naive"].state_dict(),
    #            "temp/result_" + str(model_n_dim) + "_naive.pth")

    lorentz_table_hgg = ret["model_hgg"].get_lorentz_table()
    lorentz_table_wnd = ret["model_wnd"].get_lorentz_table()
    lorentz_table_naive = ret["model_naive"].get_lorentz_table()

    # is_a_score_hgg_list.append(is_a_score(
    #     is_a, model_n_dim, lorentz_table_hgg))
    # is_a_score_wnd_list.append(is_a_score(
    #     is_a, model_n_dim, lorentz_table_wnd))
    # is_a_score_naive_list.append(is_a_score(
    #     is_a, model_n_dim, lorentz_table_naive))

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

    # basescore_y_and_z_list.append(ret["basescore_y_and_z"])
    # basescore_y_given_z_list.append(ret["basescore_y_given_z"])
    # basescore_y_given_z_naive_list.append(ret["basescore_y_given_z_naive"])
    # basescore_z_list.append(ret["basescore_z"])
    # DNML_codelength_list.append(ret["DNML_codelength"])
    # pc_first_list.append(ret["pc_first"])
    # pc_second_list.append(ret["pc_second"])
    # AIC_naive_list.append(ret["AIC_naive"])
    # BIC_naive_list.append(ret["BIC_naive"])
    # AIC_naive_from_latent_list.append(ret["AIC_naive_from_latent"])
    # BIC_naive_from_latent_list.append(ret["BIC_naive_from_latent"])

    # lorentz_table_hgg = ret["model_hgg"].get_lorentz_table()
    # lorentz_table_wnd = ret["model_wnd"].get_lorentz_table()
    # lorentz_table_naive = ret["model_naive"].get_lorentz_table()

    # is_a_score_hgg_list.append(is_a_score(
    #     is_a, model_n_dim, lorentz_table_hgg))
    # is_a_score_wnd_list.append(is_a_score(
    #     is_a, model_n_dim, lorentz_table_wnd))
    # is_a_score_naive_list.append(is_a_score(
    #     is_a, model_n_dim, lorentz_table_naive))

    # np.save(RESULTS + "/" + dataset_name + "/embedding_" +
    #         str(model_n_dim) + "_hgg.npy", lorentz_table_hgg)
    # np.save(RESULTS + "/" + dataset_name + "/embedding_" +
    #         str(model_n_dim) + "_wnd.npy", lorentz_table_wnd)
    # np.save(RESULTS + "/" + dataset_name + "/embedding_" +
    #         str(model_n_dim) + "_naive.npy", lorentz_table_naive)

    # result["model_n_dims"] = model_n_dims
    # result["n_nodes"] = params_dataset["n_nodes"]
    # result["R"] = params_dataset["R"]
    # result["DNML_codelength"] = DNML_codelength_list
    # result["AIC_naive"] = AIC_naive_list
    # result["BIC_naive"] = BIC_naive_list
    # result["AIC_naive_from_latent"] = AIC_naive_from_latent_list
    # result["BIC_naive_from_latent"] = BIC_naive_from_latent_list
    # result["basescore_y_and_z"] = basescore_y_and_z_list
    # result["basescore_y_given_z"] = basescore_y_given_z_list
    # result["basescore_y_given_naive_z"] = basescore_y_given_z_naive_list
    # result["basescore_z"] = basescore_z_list
    # result["pc_first"] = pc_first_list
    # result["pc_second"] = pc_second_list
    # result["burn_epochs"] = burn_epochs
    # result["burn_batch_size"] = burn_batch_size
    # result["n_max_positives"] = n_max_positives
    # result["n_max_negatives"] = n_max_negatives
    # result["lr_embeddings"] = lr_embeddings
    # result["lr_epoch_10"] = lr_epoch_10
    # result["lr_beta"] = lr_beta
    # result["lr_sigma"] = lr_sigma
    # result["sigma_max"] = sigma_max
    # result["sigma_min"] = sigma_min
    # result["beta_max"] = beta_max
    # result["beta_min"] = beta_min

    # result.to_csv(RESULTS + "/" + dataset_name + "/result.csv", index=False)


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
    # dataset_name = args.dataset_name

    if int(args.dataset) == 0:
        dataset_name = "animal"
    elif int(args.dataset) == 1:
        dataset_name = "mammal"
    elif int(args.dataset) == 2:
        dataset_name = "group"
    elif int(args.dataset) == 3:
        dataset_name = "solid"
    elif int(args.dataset) == 4:
        dataset_name = "tree"
    elif int(args.dataset) == 5:
        dataset_name = "worker"

    os.makedirs(RESULTS + "/" + dataset_name + "/", exist_ok=True)

    # model_n_dims = [2, 4, 8, 16, 32, 64]

    calc_metrics_realworld(device_idx=device_idx,
                           model_n_dim=model_n_dim, dataset_name=dataset_name)
