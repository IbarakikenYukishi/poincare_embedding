import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import os
import sys
import torch
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # this needs to come before other matplotlib imports
import matplotlib.pyplot as plt
import torch.multiprocessing as multi
import pandas as pd
import gc
import time
import math
from torch import nn, optim, Tensor
from tqdm import trange, tqdm
from collections import Counter
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from functools import partial
from scipy import integrate
from sklearn import metrics
from scipy import stats
from utils.utils import (
    arcosh,
    h_dist,
    e_dist,
    lorentz_scalar_product,
    tangent_norm,
    exp_map,
    log_map,
    set_dim0,
    calc_likelihood_list,
    calc_log_C_D,
    integral_sinh,
    calc_beta_hat,
    multigamma_ln,
    plot_figure_training
)
from utils.utils_dataset import (
    get_unobserved,
    Graph,
    NegGraph,
    create_test_for_link_prediction,
    create_dataset_for_basescore,
    create_dataset
)
from datasets import (
    hyperbolic_geometric_graph,
    connection_prob,
    wrapped_normal_distribution,
    euclidean_geometric_graph,
    init_HGG
)
from lorentz import RSGD_Lorentz, Lorentz, PseudoUniform, WrappedNormal


np.random.seed(0)
plt.style.use("ggplot")


def visualize_training(
    train_graph,
    positive_samples,
    negative_samples,
    lik_data,
    x_lorentz,
    params_dataset,
    model_n_dim,
    burn_epochs,
    burn_batch_size,
    n_max_positives,
    n_max_negatives,
    lr_embeddings,
    lr_epoch_10,
    lr_beta,
    lr_gamma,
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    gamma_min,
    gamma_max,
    eps_1,
    eps_2,
    init_range,
    device,
    calc_HGG=True,
    calc_WND=True,
    calc_naive=True,
    calc_othermetrics=True,
    calc_groundtruth=False,
    loader_workers=16,
    shuffle=True,
    sparse=False
):

    if model_n_dim != 2:
        raise Exception("model_n_dim must be 2.")

    print("model_n_dim:", model_n_dim)
    print("len data", len(lik_data))

    # burn-inでの処理
    dataloader = DataLoader(
        NegGraph(train_graph, n_max_positives, n_max_negatives),
        shuffle=shuffle,
        batch_size=burn_batch_size,
        num_workers=loader_workers,
        pin_memory=True
    )

    # model
    model_hgg = PseudoUniform(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta=1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_wnd = WrappedNormal(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        Sigma=torch.eye(model_n_dim) * 10,
        beta=1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_naive = PseudoUniform(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta=1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=False
    )

    # 最適化関数。
    rsgd_hgg = RSGD_Lorentz(
        model_hgg.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        device=device

    )
    rsgd_wnd = RSGD_Lorentz(
        model_wnd.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        device=device
    )
    rsgd_naive = RSGD_Lorentz(
        model_naive.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        device=device
    )

    model_hgg.to(device)
    model_wnd.to(device)
    model_naive.to(device)

    start = time.time()

    embedding_hgg = []
    embedding_wnd = []
    embedding_naive = []

    change_learning_rate = 100

    for epoch in range(burn_epochs):
        if epoch == change_learning_rate:
            rsgd_hgg.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_wnd.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_naive.param_groups[0]["lr_embeddings"] = lr_epoch_10

        losses_hgg = []
        losses_wnd = []
        losses_naive = []

        # MLE
        if calc_HGG:  # DNML-HGG
            print("HGG MLE")
            model_hgg.params_mle()
        if calc_WND:  # DNML-WND
            print("WND MLE")
            model_wnd.params_mle()
        if calc_naive:  # Naive model
            print("Naive MLE")
            model_naive.params_mle()

        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            if calc_HGG:  # DNML-HGG
                rsgd_hgg.zero_grad()
                if epoch < change_learning_rate:
                    loss_hgg = model_hgg.lik_y_given_z(pairs, labels).mean()
                else:
                    loss_hgg = model_hgg(pairs, labels).mean()
                loss_hgg.backward()
                rsgd_hgg.step()
                losses_hgg.append(loss_hgg)

            if calc_WND:  # DNML-WND
                rsgd_wnd.zero_grad()
                if epoch < change_learning_rate + 10:
                    loss_wnd = model_wnd.lik_y_given_z(pairs, labels).mean()
                else:
                    loss_wnd = model_wnd(pairs, labels).mean()
                loss_wnd.backward()
                rsgd_wnd.step()
                losses_wnd.append(loss_wnd)

            if calc_naive:  # Naive model
                rsgd_naive.zero_grad()
                loss_naive = model_naive(pairs, labels).mean()
                loss_naive.backward()
                rsgd_naive.step()
                losses_naive.append(loss_naive)

        print("epoch:", epoch)
        if calc_HGG:  # DNML-HGG
            print("loss_hgg:",
                  torch.Tensor(losses_hgg).mean().item())
        if calc_WND:  # DNML-WND
            print("loss_wnd:",
                  torch.Tensor(losses_wnd).mean().item())
        if calc_naive:  # Naive model
            print("loss_naive:",
                  torch.Tensor(losses_naive).mean().item())

        embedding_hgg.append(model_hgg.get_poincare_table())
        embedding_wnd.append(model_wnd.get_poincare_table())
        embedding_naive.append(model_naive.get_poincare_table())

    plot_figure_training(train_graph, embedding_hgg,
                         embedding_wnd, embedding_naive, suffix="01")

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


def visualize_training_(
    train_graph,
    positive_samples,
    negative_samples,
    lik_data,
    x_lorentz,
    params_dataset,
    model_n_dim,
    burn_epochs,
    burn_batch_size,
    n_max_positives,
    n_max_negatives,
    lr_embeddings,
    lr_epoch_10,
    lr_beta,
    lr_gamma,
    sigma_min,
    sigma_max,
    beta_min,
    beta_max,
    gamma_min,
    gamma_max,
    eps_1,
    eps_2,
    init_range,
    device,
    calc_HGG=True,
    calc_WND=True,
    calc_naive=True,
    calc_othermetrics=True,
    calc_groundtruth=False,
    loader_workers=16,
    shuffle=True,
    sparse=False
):

    if model_n_dim != 2:
        raise Exception("model_n_dim must be 2.")

    print("model_n_dim:", model_n_dim)
    print("len data", len(lik_data))

    # burn-inでの処理
    dataloader = DataLoader(
        NegGraph(train_graph, n_max_positives, n_max_negatives),
        shuffle=shuffle,
        batch_size=burn_batch_size,
        num_workers=loader_workers,
        pin_memory=True
    )

    # model
    model_hgg = PseudoUniform(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta=1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_wnd = WrappedNormal(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        Sigma=torch.eye(model_n_dim) * 10,
        beta=1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_naive = PseudoUniform(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,  # モデルの次元
        R=params_dataset['R'],
        sigma=1.0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        beta=1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=False
    )

    # 最適化関数。
    rsgd_hgg = RSGD_Lorentz(
        model_hgg.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        device=device

    )
    rsgd_wnd = RSGD_Lorentz(
        model_wnd.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        device=device
    )
    rsgd_naive = RSGD_Lorentz(
        model_naive.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        device=device
    )

    model_hgg.to(device)
    model_wnd.to(device)
    model_naive.to(device)

    start = time.time()

    embedding_hgg = []
    embedding_wnd = []
    embedding_naive = []

    change_learning_rate = 100

    for epoch in range(800):
        if epoch == change_learning_rate:
            rsgd_hgg.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_wnd.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_naive.param_groups[0]["lr_embeddings"] = lr_epoch_10

        losses_hgg = []
        losses_wnd = []
        losses_naive = []

        # MLE
        # if calc_HGG:  # DNML-HGG
        #     print("HGG MLE")
        #     model_hgg.params_mle()
        # if calc_WND:  # DNML-WND
        #     print("WND MLE")
        #     model_wnd.params_mle()
        if calc_naive:  # Naive model
            print("Naive MLE")
            model_naive.params_mle()

        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            # if calc_HGG:  # DNML-HGG
            #     rsgd_hgg.zero_grad()
            #     if epoch < change_learning_rate:
            #         loss_hgg = model_hgg.lik_y_given_z(pairs, labels).mean()
            #     else:
            #         loss_hgg = model_hgg(pairs, labels).mean()
            #     loss_hgg.backward()
            #     rsgd_hgg.step()
            #     losses_hgg.append(loss_hgg)

            # if calc_WND:  # DNML-WND
            #     rsgd_wnd.zero_grad()
            #     if epoch < change_learning_rate + 10:
            #         loss_wnd = model_wnd.lik_y_given_z(pairs, labels).mean()
            #     else:
            #         loss_wnd = model_wnd(pairs, labels).mean()
            #     loss_wnd.backward()
            #     rsgd_wnd.step()
            #     losses_wnd.append(loss_wnd)

            if calc_naive:  # Naive model
                rsgd_naive.zero_grad()
                loss_naive = model_naive(pairs, labels).mean()
                loss_naive.backward()
                rsgd_naive.step()
                losses_naive.append(loss_naive)

        print("epoch:", epoch)
        # if calc_HGG:  # DNML-HGG
        #     print("loss_hgg:",
        #           torch.Tensor(losses_hgg).mean().item())
        # if calc_WND:  # DNML-WND
        #     print("loss_wnd:",
        #           torch.Tensor(losses_wnd).mean().item())
        if calc_naive:  # Naive model
            print("loss_naive:",
                  torch.Tensor(losses_naive).mean().item())

        # embedding_hgg.append(model_hgg.get_poincare_table())
        # embedding_wnd.append(model_wnd.get_poincare_table())
        embedding_naive.append(model_naive.get_poincare_table())

    model_hgg.set_embedding(torch.Tensor(
        model_naive.get_lorentz_table()).to(device))
    model_wnd.set_embedding(torch.Tensor(
        model_naive.get_lorentz_table()).to(device))

    # Fine tuning
    for epoch in range(100):
        # if epoch == change_learning_rate:
        rsgd_hgg.param_groups[0]["lr_embeddings"] = lr_epoch_10
        rsgd_wnd.param_groups[0]["lr_embeddings"] = lr_epoch_10

        losses_hgg = []
        losses_wnd = []
        # losses_naive = []

        # MLE
        if calc_HGG:  # DNML-HGG
            print("HGG MLE")
            model_hgg.params_mle()
        if calc_WND:  # DNML-WND
            print("WND MLE")
            model_wnd.params_mle()
        # if calc_naive:  # Naive model
        #     print("Naive MLE")
        #     model_naive.params_mle()

        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            if calc_HGG:  # DNML-HGG
                rsgd_hgg.zero_grad()
                loss_hgg = model_hgg(pairs, labels).mean()
                loss_hgg.backward()
                rsgd_hgg.step()
                losses_hgg.append(loss_hgg)

            if calc_WND:  # DNML-WND
                rsgd_wnd.zero_grad()
                loss_wnd = model_wnd(pairs, labels).mean()
                loss_wnd.backward()
                rsgd_wnd.step()
                losses_wnd.append(loss_wnd)

            # if calc_naive:  # Naive model
            #     rsgd_naive.zero_grad()
            #     loss_naive = model_naive(pairs, labels).mean()
            #     loss_naive.backward()
            #     rsgd_naive.step()
            #     losses_naive.append(loss_naive)

        print("epoch:", epoch)
        if calc_HGG:  # DNML-HGG
            print("loss_hgg:",
                  torch.Tensor(losses_hgg).mean().item())
        if calc_WND:  # DNML-WND
            print("loss_wnd:",
                  torch.Tensor(losses_wnd).mean().item())
        # if calc_naive:  # Naive model
        #     print("loss_naive:",
        #           torch.Tensor(losses_naive).mean().item())

        embedding_hgg.append(model_hgg.get_poincare_table())
        embedding_wnd.append(model_wnd.get_poincare_table())

    plot_figure_training(train_graph, embedding_hgg,
                         embedding_wnd, embedding_naive, suffix="01_")

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == '__main__':
    # creating dataset
    # n_nodes = 6400
    n_nodes = 400

    print("R:", np.log(n_nodes))

    params_dataset = {
        'n_nodes': n_nodes,
        'n_dim': 4,
        'R': np.log(n_nodes),
        'sigma': 5,
        'beta': 0.8
    }

    # dataset_name=

    # data = np.load('dataset/' + dataset_name +
    #                '/data.npy', allow_pickle=True).item()
    # adj_mat = data["adj_mat"].toarray()
    # positive_samples = data["positive_samples"]
    # negative_samples = data["negative_samples"]
    # train_graph = data["train_graph"].toarray()
    # lik_data = data["lik_data"]

    # parameters
    burn_epochs = 800
    # burn_epochs = 200
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
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
    eps_2 = 1e3
    init_range = 0.001
    # others
    loader_workers = 16
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 隣接行列
    adj_mat, x_lorentz = hyperbolic_geometric_graph(
        n_nodes=params_dataset['n_nodes'],
        n_dim=params_dataset['n_dim'],
        R=params_dataset['R'],
        sigma=params_dataset['sigma'],
        beta=params_dataset['beta']
    )

    # 平均次数が少なくなるように手で調整する用
    print('average degree:', np.sum(adj_mat) / len(adj_mat))

    result = pd.DataFrame()

    # model_n_dims = [4, 8, 16, 32, 64]
    # model_n_dims = [16, 32, 64]
    # model_n_dims = [64]
    model_n_dims = [2]

    positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
        adj_mat=adj_mat,
        params_dataset=params_dataset
    )

    # negative samplingの比率を平均次数から決定
    pos_train_graph = len(np.where(train_graph == 1)[0])
    neg_train_graph = len(np.where(train_graph == 0)[0])
    ratio = neg_train_graph / pos_train_graph
    print("ratio:", ratio)

    n_max_negatives = int(n_max_positives * ratio)
    print("n_max_negatives:", n_max_negatives)
    lr_embeddings = 0.1
    lr_epoch_10 = 10.0 * \
        (burn_batch_size * (n_max_positives + n_max_negatives)) / \
        32 / 100  # batchサイズに対応して学習率変更
    # lr_epoch_10 = 1 * \
    #     (burn_batch_size * (n_max_positives + n_max_negatives)) / \
    #     32 / 100  # batchサイズに対応して学習率変更

    for model_n_dim in model_n_dims:
        visualize_training_(
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
            init_range=init_range,
            device=device,
            calc_HGG=True,
            calc_WND=True,
            calc_naive=True,
            calc_othermetrics=True,
            calc_groundtruth=True,
            loader_workers=16,
            shuffle=True,
            sparse=False
        )
