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
from space_selection import RSGD, Lorentz, Euclidean, Spherical


np.random.seed(0)
plt.style.use("ggplot")


def visualize_training(
    train_graph,
    positive_samples,
    negative_samples,
    lik_data,
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
    init_range,
    device,
    calc_lorentz=True,
    calc_euclidean=True,
    calc_spherical=True,
    perturbation=True,
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
    model_lorentz_latent = Lorentz(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,
        R=params_dataset['R'],
        k=-1,
        sigma=torch.ones(model_n_dim),
        beta=1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_euclidean_latent = Euclidean(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,
        sigma=torch.ones(model_n_dim),
        beta=1.0,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )
    model_spherical_latent = Spherical(
        n_nodes=params_dataset['n_nodes'],
        n_dim=model_n_dim,
        sigma=torch.ones(model_n_dim),
        beta=1.0,
        k=1,
        gamma=params_dataset['R'],
        init_range=init_range,
        sparse=sparse,
        device=device,
        calc_latent=True
    )

    # optimizer
    rsgd_lorentz_latent = RSGD(
        model_lorentz_latent.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=params_dataset['R'],
        k=-1,
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )
    rsgd_euclidean_latent = RSGD(
        model_euclidean_latent.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=0,  # dummy argument
        k=0,  # dummy argument
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )
    rsgd_spherical_latent = RSGD(
        model_spherical_latent.parameters(),
        lr_embeddings=lr_embeddings,
        lr_beta=lr_beta,
        lr_gamma=lr_gamma,
        R=0,  # dummy argument
        k=1,
        beta_max=beta_max,
        beta_min=beta_min,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        perturbation=perturbation,
        device=device
    )

    model_lorentz_latent.to(device)
    model_euclidean_latent.to(device)
    model_spherical_latent.to(device)

    start = time.time()

    embedding_lorentz = []
    embedding_euclidean = []
    embedding_spherical = []

    change_learning_rate = 100

    for epoch in range(burn_epochs):
        if epoch == change_learning_rate:
            rsgd_lorentz_latent.param_groups[0]["lr_embeddings"] = lr_epoch_10
            rsgd_euclidean_latent.param_groups[
                0]["lr_embeddings"] = lr_epoch_10
            rsgd_spherical_latent.param_groups[
                0]["lr_embeddings"] = lr_epoch_10

        losses_lorentz_latent = []
        losses_euclidean_latent = []
        losses_spherical_latent = []

        # MLE
        if calc_lorentz:  # Lorentz
            print("Lorentz MLE")
            model_lorentz_latent.params_mle(
                sigma_min,
                sigma_max
            )

        if calc_euclidean:  # Euclidean
            print("Euclidean MLE")
            model_euclidean_latent.params_mle(
                sigma_min,
                sigma_max
            )

        if calc_spherical:  # Euclidean
            print("Spherical MLE")
            model_spherical_latent.params_mle(
                sigma_min,
                sigma_max
            )

        for pairs, labels in dataloader:
            pairs = pairs.reshape((-1, 2))
            labels = labels.reshape(-1)

            pairs = pairs.to(device)
            labels = labels.to(device)

            if calc_lorentz:  # Lorentz
                rsgd_lorentz_latent.zero_grad()
                if epoch < change_learning_rate:
                    loss_lorentz_latent = model_lorentz_latent.lik_y_given_z(
                        pairs, labels).mean()
                else:
                    loss_lorentz_latent = model_lorentz_latent(
                        pairs, labels).mean()
                loss_lorentz_latent.backward()
                rsgd_lorentz_latent.step()
                losses_lorentz_latent.append(loss_lorentz_latent)

            if calc_euclidean:  # Euclidean
                rsgd_euclidean_latent.zero_grad()
                if epoch < change_learning_rate:
                    loss_euclidean_latent = model_euclidean_latent.lik_y_given_z(
                        pairs, labels).mean()
                else:
                    loss_euclidean_latent = model_euclidean_latent(
                        pairs, labels).mean()
                loss_euclidean_latent.backward()
                rsgd_euclidean_latent.step()
                losses_euclidean_latent.append(loss_euclidean_latent)

            if calc_spherical:  # Euclidean
                rsgd_spherical_latent.zero_grad()
                if epoch < change_learning_rate:
                    loss_spherical_latent = model_spherical_latent.lik_y_given_z(
                        pairs, labels).mean()
                else:
                    loss_spherical_latent = model_spherical_latent(
                        pairs, labels).mean()
                loss_spherical_latent.backward()
                rsgd_spherical_latent.step()
                losses_spherical_latent.append(loss_spherical_latent)

        print("epoch:", epoch)
        if calc_lorentz:  # Lorentz
            print("loss_lorentz:",
                  torch.Tensor(losses_lorentz_latent).mean().item())
        if calc_euclidean:  # Euclidean
            print("loss_euclidean:",
                  torch.Tensor(losses_euclidean_latent).mean().item())
        if calc_spherical:  # Spherical
            print("loss_spherical:",
                  torch.Tensor(losses_spherical_latent).mean().item())

        embedding_lorentz.append(model_lorentz_latent.get_poincare_table())
        embedding_euclidean.append(model_euclidean_latent.get_table())
        embedding_spherical.append(model_spherical_latent.get_table())

    plot_figure_training(train_graph, embedding_lorentz,
                         embedding_euclidean, embedding_spherical, suffix="01")

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
    # burn_epochs = 20
    burn_epochs = 800
    burn_batch_size = min(int(params_dataset["n_nodes"] * 0.2), 100)
    n_max_positives = min(int(params_dataset["n_nodes"] * 0.02), 10)
    lr_beta = 0.001
    lr_gamma = 0.001
    sigma_max = 100.0
    sigma_min = 0.001
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
        visualize_training(
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
            lr_beta=lr_beta,
            lr_gamma=lr_gamma,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            beta_min=beta_min,
            beta_max=beta_max,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            init_range=init_range,
            device=device,
            calc_lorentz=True,
            calc_euclidean=True,
            calc_spherical=True,
            perturbation=True,
            loader_workers=16,
            shuffle=True,
            sparse=False
        )
