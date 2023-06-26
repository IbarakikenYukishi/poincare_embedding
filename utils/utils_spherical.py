import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import os
import sys
import torch
import random
import numpy as np
from torch import nn, optim, Tensor
from tqdm import trange, tqdm
from collections import Counter
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch.multiprocessing as multi
from functools import partial, lru_cache
import pandas as pd
import gc
import time
from scipy import integrate
from sklearn import metrics
import math
from scipy import stats, special
from sklearn.linear_model import LogisticRegression
from matplotlib.animation import ArtistAnimation, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, current_process
from PIL import Image

from torch.distributions import MultivariateNormal


np.random.seed(0)

import matplotlib

matplotlib.use("Agg")  # this needs to come before other matplotlib imports
import matplotlib.pyplot as plt

plt.style.use("ggplot")


@lru_cache(maxsize=None)
def multigamma_ln(a, d):
    return special.multigammaln(a, d)


def approx_W_k(Sigma, k, sample_size):
    dim = Sigma.shape[0]
    points = np.random.multivariate_normal(np.zeros(dim), Sigma, sample_size)
    dists = np.linalg.norm(points, axis=1)
    ratio = len(np.where(dists <= np.pi / np.sqrt(k))[0]) / sample_size
    return ratio


def approx_log_W_k(sigma, k, sample_size):
    # assuming sigma is an 1-dimensional array
    dim = sigma.shape[0]
    # uniform with respect to angle
    points = np.random.multivariate_normal(
        np.zeros(dim), np.eye(dim), sample_size)
    norm = np.linalg.norm(points, axis=1).reshape((-1, 1))
    points = points / norm
    R = np.pi / np.sqrt(k)
    r = R * np.power(np.random.uniform(0, 1, size=sample_size),
                     1 / dim).reshape((sample_size, 1))
    points = points * r

    liks = - 0.5 * (points * points).dot((sigma**(-1)).reshape(dim, 1))
    liks += -0.5 * (dim * np.log(2 * np.pi) + np.sum(np.log(sigma))) + dim / \
        2 * np.log(np.pi) + dim * np.log(R) - multigamma_ln(dim / 2 + 1, 1)
    ret = -np.log(sample_size) + special.logsumexp(liks)
    return ret


def approx_log_W_k_gpu(sigma, k, sample_size, device):
    # assuming sigma is an 1-dimensional array
    dim = sigma.shape[0]
    # uniform with respect to angle
    points = MultivariateNormal(
        torch.zeros(dim).to(device), torch.eye(dim).to(device)).sample((sample_size,))
    # print(points)
    norm = torch.norm(points, dim=1).reshape((-1, 1))
    points = points / norm
    R = np.pi / np.sqrt(k)
    r = R * torch.pow(torch.rand(size=(sample_size,), device=device),
                     1 / dim).reshape((sample_size, 1))
    points = points * r
    # print(points)

    liks = - 0.5 * (points * points).mm((sigma**(-1)).reshape(dim, 1))
    liks += -0.5 * (dim * np.log(2 * np.pi) + torch.sum(torch.log(sigma))) + dim / \
        2 * np.log(np.pi) + dim * np.log(R) - multigamma_ln(dim / 2 + 1, 1)
    # print(liks)
    ret = -np.log(sample_size) + torch.logsumexp(liks, dim=0)
    return ret.detach().cpu().item()

def approx_log_v_jk_gpu(sigma, k, sample_size, device):
    # assuming sigma is an 1-dimensional array
    dim = sigma.shape[0]
    # uniform with respect to angle
    points = MultivariateNormal(
        torch.zeros(dim).to(device), torch.eye(dim).to(device)).sample((sample_size,))
    norm = torch.norm(points, dim=1).reshape((-1, 1))
    points = points / norm
    R = np.pi / np.sqrt(k)
    r = R * torch.pow(torch.rand(size=(sample_size,), device=device),
                     1 / dim).reshape((sample_size, 1))
    points = points * r
    v_jk = 2 * torch.log(torch.abs(points))

    liks_ = -0.5 * (points * points).mm((sigma**(-1)).reshape(dim, 1))
    liks_ += -0.5 * (dim * np.log(2 * np.pi) + torch.sum(torch.log(sigma))) + dim / \
        2 * np.log(np.pi) + dim * np.log(R) - multigamma_ln(dim / 2 + 1, 1)

    liks = liks_ + v_jk - 2 * torch.log(sigma)[None, :]
    ret = -np.log(sample_size) + torch.logsumexp(liks, dim=0)
    return ret



def approx_log_W_k(sigma, k, sample_size):
    # assuming sigma is an 1-dimensional array
    dim = sigma.shape[0]
    # uniform with respect to angle
    points = np.random.multivariate_normal(
        np.zeros(dim), np.eye(dim), sample_size)
    norm = np.linalg.norm(points, axis=1).reshape((-1, 1))
    points = points / norm
    R = np.pi / np.sqrt(k)
    r = R * np.power(np.random.uniform(0, 1, size=sample_size),
                     1 / dim).reshape((sample_size, 1))
    points = points * r

    liks = - 0.5 * (points * points).dot((sigma**(-1)).reshape(dim, 1))
    liks += -0.5 * (dim * np.log(2 * np.pi) + np.sum(np.log(sigma))) + dim / \
        2 * np.log(np.pi) + dim * np.log(R) - multigamma_ln(dim / 2 + 1, 1)
    ret = -np.log(sample_size) + special.logsumexp(liks)
    return ret


def approx_log_v_jk(sigma, k, sample_size):
    # assuming sigma is an 1-dimensional array
    dim = sigma.shape[0]
    # uniform with respect to angle
    points = np.random.multivariate_normal(
        np.zeros(dim), np.eye(dim), sample_size)
    norm = np.linalg.norm(points, axis=1).reshape((-1, 1))
    points = points / norm
    R = np.pi / np.sqrt(k)
    r = R * np.power(np.random.uniform(0, 1, size=sample_size),
                     1 / dim).reshape((sample_size, 1))
    points = points * r
    v_jk = 2 * np.log(np.abs(points))

    liks_ = -0.5 * (points * points).dot((sigma**(-1)).reshape(dim, 1))
    liks_ += -0.5 * (dim * np.log(2 * np.pi) + np.sum(np.log(sigma))) + dim / \
        2 * np.log(np.pi) + dim * np.log(R) - multigamma_ln(dim / 2 + 1, 1)

    liks = liks_ + v_jk - 2 * np.log(sigma)[np.newaxis, :]
    ret = -np.log(sample_size) + special.logsumexp(liks, axis=0)
    return ret

# start = time.time()
# print(approx_log_W_k_gpu(sigma=torch.ones(32).to("cuda:0"), k=1, sample_size=100000, device="cuda:0"))
# elapsed_time = time.time() - start
# print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
# start = time.time()
# print(approx_log_W_k_gpu(sigma=torch.ones(32).to("cuda:0"), k=1, sample_size=100000, device="cuda:0"))
# elapsed_time = time.time() - start
# print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
# start = time.time()
# print(approx_log_W_k(sigma=np.ones(32), k=1, sample_size=100000))
# elapsed_time = time.time() - start
# print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
# start = time.time()
# print(approx_log_v_jk_gpu(sigma=torch.ones(32).to("cuda:0"), k=1, sample_size=100000, device="cuda:0"))
# elapsed_time = time.time() - start
# print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
# start = time.time()
# print(approx_log_v_jk(sigma=np.ones(32), k=1, sample_size=100000))
# elapsed_time = time.time() - start
# print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


def approx_v_jk(Sigma, k, sample_size):
    dim = Sigma.shape[0]
    points = np.random.multivariate_normal(np.zeros(dim), Sigma, sample_size)
    dists = np.linalg.norm(points, axis=1)
    points_2 = points * points
    v_jk = np.sum(
        points_2[np.where(dists <= np.pi / np.sqrt(k))[0], :], axis=0)
    ret = v_jk / (sample_size * np.diag(Sigma)**2)
    return ret


def mle_truncated_normal(
    points,
    sigma_min,
    sigma_max,
    k,
    sample_size,
    n_iter,
    learning_rate,
    alpha,
    early_stopping=100,
    verbose=False
):
    n_points = points.shape[0]
    dim = points.shape[1]
    param_sigma = np.ones(dim) * sigma_min
    early_stopping_rounds = 0
    lik_best = 999999999999
    velocity = 0

    # iteration
    for t in range(n_iter):
        # already divided by n_points
        # momentum
        gradient = - np.sum(points * points / n_points,
                            axis=0) / (2 * param_sigma**2)
        gradient += 1 / (2 * approx_W_k(np.diag(param_sigma), k,
                                        sample_size)) * approx_v_jk(np.diag(param_sigma), k, sample_size)
        gradient /= max(np.linalg.norm(gradient), 1)
        velocity = alpha * velocity - learning_rate * gradient
        param_sigma = param_sigma + velocity
        param_sigma = np.where(param_sigma > sigma_max, sigma_max, param_sigma)
        param_sigma = np.where(param_sigma < sigma_min, sigma_min, param_sigma)

        lik = n_points / 2 * (dim * np.log(2 * np.pi) + np.sum(np.log(param_sigma))) + 0.5 * np.sum((points * points).dot((param_sigma**(-1)).reshape(dim, 1))) +\
            n_points * np.log(approx_W_k(np.diag(param_sigma), k, sample_size))

        if verbose:
            print("sigma:", param_sigma)
            print(gradient)
            print("lik:", lik)
            print("best:", lik_best)

        if lik_best <= lik:
            early_stopping_rounds += 1
        else:
            early_stopping_rounds = 0
            lik_best = lik
        if early_stopping_rounds >= early_stopping:
            if verbose:
                print("early_stopping")
            break

    lik = n_points / 2 * (dim * np.log(2 * np.pi) + np.sum(np.log(param_sigma))) + 0.5 * np.sum((points * points).dot((param_sigma**(-1)).reshape(dim, 1))) +\
        n_points * np.log(approx_W_k(np.diag(param_sigma), k, sample_size))

    return param_sigma, lik


def mle_truncated_normal_(
    points,
    sigma_min,
    sigma_max,
    k,
    sample_size,
    n_iter,
    learning_rate,
    alpha,
    early_stopping=100,
    verbose=False
):
    n_points = points.shape[0]
    dim = points.shape[1]
    param_sigma = np.ones(dim) * sigma_min
    early_stopping_rounds = 0
    lik_best = 999999999999
    velocity = 0

    # iteration
    for t in range(n_iter):
        # already divided by n_points
        # momentum
        gradient = - np.sum(points * points / n_points,
                            axis=0) / (2 * param_sigma**2)
        # gradient += 1 / (2 * approx_W_k(np.diag(param_sigma), k,
        # sample_size)) * approx_v_jk(np.diag(param_sigma), k, sample_size)
        gradient += 0.5 * np.exp(approx_log_v_jk(param_sigma, k,
                                                 sample_size) - approx_log_W_k(param_sigma, k, sample_size))

        gradient /= max(np.linalg.norm(gradient), 1)
        velocity = alpha * velocity - learning_rate * gradient
        param_sigma = param_sigma + velocity
        param_sigma = np.where(param_sigma > sigma_max, sigma_max, param_sigma)
        param_sigma = np.where(param_sigma < sigma_min, sigma_min, param_sigma)

        lik = n_points / 2 * (dim * np.log(2 * np.pi) + np.sum(np.log(param_sigma))) + 0.5 * np.sum((points * points).dot((param_sigma**(-1)).reshape(dim, 1))) +\
            n_points * approx_log_W_k(param_sigma, k, sample_size)

        if verbose:
            print("sigma:", param_sigma)
            print(gradient)
            print("lik:", lik)
            print("best:", lik_best)

        if lik_best <= lik:
            early_stopping_rounds += 1
        else:
            early_stopping_rounds = 0
            lik_best = lik
        if early_stopping_rounds >= early_stopping:
            if verbose:
                print("early_stopping")
            break

    # lik = n_points / 2 * (dim * np.log(2 * np.pi) + np.sum(np.log(param_sigma))) + 0.5 * np.sum((points * points).dot((param_sigma**(-1)).reshape(dim, 1))) +\
    #     n_points * np.log(approx_W_k(np.diag(param_sigma), k, sample_size))

    return param_sigma, lik_best


def mle_truncated_normal_gpu(
    points,
    sigma_min,
    sigma_max,
    k,
    sample_size,
    n_iter,
    learning_rate,
    alpha,
    sigma_init,
    device,
    early_stopping=100,
    verbose=False
):
    # assume that points are already on cuda:device
    n_points = points.shape[0]
    dim = points.shape[1]
    param_sigma = torch.clone(sigma_init)
    param_sigma_best = torch.clone(sigma_init)
    early_stopping_rounds = 0
    lik_best = 999999999999
    velocity = 0
    # print(points)

    # iteration
    for t in range(n_iter):
        # already divided by n_points
        # momentum
        param_sigma_ = param_sigma.detach().cpu().numpy()
        gradient = - torch.sum(points * points / n_points,
                               dim=0) / (2 * param_sigma**2)
        # gradient += 0.5 * torch.tensor(np.exp(approx_log_v_jk(
        #     param_sigma_, k, sample_size) - approx_log_W_k(param_sigma_, k, sample_size))).to(device)
        gradient += 0.5 * torch.exp(approx_log_v_jk_gpu(
            param_sigma, k, sample_size, device) - approx_log_W_k_gpu(param_sigma, k, sample_size, device))

        gradient /= max(torch.norm(gradient), 1)
        velocity = alpha * velocity - learning_rate * gradient
        update = param_sigma + velocity

        is_nan_inf = torch.isnan(update) | torch.isinf(update)
        param_sigma = torch.where(is_nan_inf, param_sigma, update)

        param_sigma = torch.where(
            param_sigma > sigma_max, torch.tensor(
                sigma_max).to(device), param_sigma)
        param_sigma = torch.where(
            param_sigma < sigma_min, torch.tensor(
                sigma_min).to(device), param_sigma)

        # lik = n_points / 2 * (dim * np.log(2 * np.pi) + torch.sum(torch.log(param_sigma))) + 0.5 * torch.sum((points * points).mm((param_sigma**(-1)).reshape(dim, 1))) +\
        #     n_points * \
        #     np.log(approx_W_k(
        #         np.diag(param_sigma.detach().cpu().numpy()), k, sample_size))
        # lik = n_points / 2 * (dim * np.log(2 * np.pi) + torch.sum(torch.log(param_sigma))) + 0.5 * torch.sum((points * points).mm((param_sigma**(-1)).reshape(dim, 1))) +\
        #     n_points * approx_log_W_k(param_sigma_, k, sample_size)
        lik = n_points / 2 * (dim * np.log(2 * np.pi) + torch.sum(torch.log(param_sigma))) + 0.5 * torch.sum((points * points).mm((param_sigma**(-1)).reshape(dim, 1))) +\
            n_points * approx_log_W_k_gpu(param_sigma, k, sample_size, device)


        if verbose:
            print("sigma:", param_sigma)
            print(gradient)
            print("lik:", lik)
            print("best:", lik_best)

        if lik_best <= lik:
            early_stopping_rounds += 1
        else:
            early_stopping_rounds = 0
            lik_best = lik
            param_sigma_best = torch.clone(param_sigma)
        if early_stopping_rounds >= early_stopping:
            if verbose:
                print("early_stopping")
            break

    # lik = n_points / 2 * (dim * np.log(2 * np.pi) + np.sum(np.log(param_sigma))) + 0.5 * np.sum((points * points).dot((param_sigma**(-1)).reshape(dim, 1))) +\
    #     n_points * np.log(approx_W_k(np.diag(param_sigma), k, sample_size))

    return param_sigma_best, lik_best


def mle_truncated_normal_gpu_(
    set_points,
    sigma_min,
    sigma_max,
    k,
    sample_size,
    n_iter,
    learning_rate,
    alpha,
    device,
    early_stopping=100,
    verbose=False
):
    # assume that points are already on cuda:device
    n_samples = set_points.shape[0]
    n_points = set_points.shape[1]
    dim = set_points.shape[2]
    param_sigma = torch.ones((n_samples, dim)).to(device) * sigma_min
    # early_stopping_rounds = 0
    # lik_best = 999999999999
    velocity = 0

    # iteration
    for t in range(n_iter):
        # already divided by n_points
        # momentum
        # print(param_sigma.detach().cpu().numpy())
        param_sigma_ = param_sigma.detach().cpu().numpy()
        # print(torch.sum(set_points * set_points / n_points,
        #                     axis=1).shape)
        gradient = - torch.sum(set_points * set_points / n_points,
                               axis=1) / (2 * param_sigma**2)

        vec = torch.zeros((n_samples, dim)).to(device)
        for j in range(n_samples):
            vec[j, :] = 1 / (2 * approx_W_k(np.diag(param_sigma_[j, :]), k,
                                            sample_size)) * torch.tensor(approx_v_jk(np.diag(param_sigma_[j, :]), k, sample_size)).to(device)
        gradient += vec
        # gradient += 1 / (2 * approx_W_k(np.diag(param_sigma_), k,
        #                                 sample_size)) * torch.tensor(approx_v_jk(np.diag(param_sigma_), k, sample_size)).to(device)
        # print(torch.renorm(gradient, p=2, dim=1, maxnorm=1).shape)
        gradient /= torch.renorm(gradient, p=2, dim=1, maxnorm=1)
        # gradient /= torch.max(torch.norm(gradient, dim=1, keepdim=True), 1)
        velocity = alpha * velocity - learning_rate * gradient
        param_sigma = param_sigma + velocity
        param_sigma = torch.where(
            param_sigma > sigma_max, sigma_max, param_sigma)
        param_sigma = torch.where(
            param_sigma < sigma_min, sigma_min, param_sigma)

        # lik = n_points / 2 * (dim * np.log(2 * np.pi) + torch.sum(torch.log(param_sigma))) + 0.5 * torch.sum((points * points).mm((param_sigma**(-1)).reshape(dim, 1))) +\
        #     n_points * np.log(approx_W_k(np.diag(param_sigma.detach().cpu().numpy()), k, sample_size))

        if verbose:
            print("sigma:", param_sigma)
            print(gradient)
            print("lik:", lik)
            print("best:", lik_best)

        # if lik_best <= lik:
        #     early_stopping_rounds += 1
        # else:
        #     early_stopping_rounds = 0
        #     lik_best = lik
        # if early_stopping_rounds >= early_stopping:
        #     if verbose:
        #         print("early_stopping")
        #     break

    param_sigma_ = param_sigma.detach().cpu().numpy()

    liks = []
    for i in range(n_samples):

        lik = n_points / 2 * (dim * np.log(2 * np.pi) + np.sum(np.log(param_sigma_[j, :]))) + 0.5 * np.sum((set_points * set_points).dot((param_sigma_[j, :]**(-1)).reshape(dim, 1))) +\
            n_points * \
            np.log(approx_W_k(np.diag(param_sigma_[j, :]), k, sample_size))
        liks.append(lik)

    return param_sigma, liks


def mle_normal(
    points,
    sigma_min,
    sigma_max,
    k,
    sample_size,
    n_iter,
    learning_rate,
    alpha,
    early_stopping=100,
    verbose=False,
):
    n_points = points.shape[0]
    dim = points.shape[1]
    param_sigma = np.ones(dim) * sigma_min
    early_stopping_rounds = 0
    lik_best = 999999999999
    velocity = 0

    for t in range(n_iter):
        # divided by n_points
        gradient = 1 / (2 * param_sigma) - np.sum(points *
                                                  points / n_points, axis=0) / (2 * param_sigma**2)
        gradient /= max(np.linalg.norm(gradient), 1)
        velocity = alpha * velocity - learning_rate * gradient
        param_sigma = param_sigma + velocity
        param_sigma = np.where(param_sigma > sigma_max, sigma_max, param_sigma)
        param_sigma = np.where(param_sigma < sigma_min, sigma_min, param_sigma)
        lik = n_points / 2 * (dim * np.log(2 * np.pi) + np.sum(np.log(param_sigma))) + \
            0.5 * \
            np.sum((points * points).dot((param_sigma**(-1)).reshape(dim, 1)))
        if verbose:
            print("sigma:", param_sigma)
            print(lik)
        if lik_best <= lik:
            early_stopping_rounds += 1
        else:
            early_stopping_rounds = 0
            lik_best = lik
        if early_stopping_rounds >= early_stopping:
            if verbose:
                print("early_stopping")
            break
    return param_sigma, lik_best


def calc_ml_with_importance(
    i,
    dim,  # 次元D
    sample_size,  # ノード数n
    # n_samples,  # parametric complexityを近似するサンプル数
    sigma_min,
    sigma_max,
    k,
    n_iter,
    learning_rate,
    alpha,
    sample_size_for_W,
    early_stopping=100
):
    np.random.seed(i)  # シード値を設定

    # uniform with respect to angle
    points = np.random.multivariate_normal(
        np.zeros(dim), np.eye(dim), sample_size)
    norm = np.linalg.norm(points, axis=1).reshape((-1, 1))
    points = points / norm
    R = np.pi / np.sqrt(k)
    r = R * np.power(np.random.uniform(0, 1, size=sample_size),
                     1 / dim).reshape((sample_size, 1))
    points = points * r

    _, lik = mle_truncated_normal(
        points,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        k=k,
        sample_size=sample_size_for_W,
        n_iter=n_iter,
        learning_rate=learning_rate,
        alpha=alpha,
        early_stopping=early_stopping,
        # verbose=True
    )
    # _, lik = mle_normal(
    #     points,
    #     sigma_min=sigma_min,
    #     sigma_max=sigma_max,
    #     k=k,
    #     sample_size=sample_size_for_W,
    #     n_iter=n_iter,
    #     learning_rate=learning_rate,
    #     alpha=alpha,
    #     early_stopping=early_stopping,
    #     # verbose=True
    # )
    return lik


def calc_spherical_complexity(
    dim,  # 次元D
    sample_size,  # ノード数n
    n_samples,  # parametric complexityを近似するサンプル数
    sigma_min,
    sigma_max,
    k,
    n_iter,
    learning_rate,
    alpha,
    sample_size_for_W=100,
    early_stopping=100
):
    calc_ml_with_importance_ = partial(
        calc_ml_with_importance,
        dim=dim,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        k=k,
        n_iter=n_iter,
        learning_rate=learning_rate,
        alpha=alpha,
        sample_size_for_W=sample_size_for_W,
        early_stopping=early_stopping
    )
    liks = []
    p = Pool(1)
    with tqdm(total=n_samples)as t:
        for lik in p.imap_unordered(calc_ml_with_importance_, range(n_samples)):
            t.update(1)
            liks.append(lik)
    liks = np.array(liks)
    R = np.pi / np.sqrt(k)
    parametric_complexity = - np.log(n_samples) + sample_size * (0.5 * (dim) * np.log(np.pi) +
                                                                 dim * np.log(R) - special.gammaln(dim / 2 + 1)) + special.logsumexp(-liks)
    print(liks)
    print(sample_size * (0.5 * (dim) * np.log(np.pi) +
                         dim * np.log(R) - special.gammaln(dim / 2 + 1)))
    print(special.logsumexp(-liks))
    print(parametric_complexity)

    hyperbolic_pc = dim * (sample_size / 2 * np.log(sample_size / (2 * np.e)) - multigamma_ln(
        sample_size / 2, 1)) + dim * np.log(np.log(sigma_max) - np.log(sigma_min))
    print(hyperbolic_pc)

    return parametric_complexity

if __name__ == "__main__":
    # start = time.time()
    # print(approx_W_k_(sigma=np.ones(64) * ((np.pi / 2)**2), k=1, sample_size=1000))
    # elapsed_time = time.time() - start
    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # start = time.time()
    # print(approx_W_k(Sigma=np.eye(64) * ((np.pi / 2)**2), k=1, sample_size=1000))
    # elapsed_time = time.time() - start
    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # print(approx_v_jk(Sigma=np.eye(32)*5, k=1, sample_size=10000))
    # print(np.exp(approx_log_v_jk(sigma=np.ones(32)*5, k=1, sample_size=10000)))

    # sigma = np.ones(64)
    # k=1
    # sample_size = 1000
    # print(approx_v_jk(np.diag(sigma), k, sample_size) / (2 * approx_W_k(np.diag(sigma), k, sample_size)))
    # print(0.5 * np.exp(approx_log_v_jk(sigma, k, sample_size) - approx_log_W_k(sigma, k, sample_size)))

    # start = time.time()
    # print(approx_W_k_(sigma=np.ones(32)*(0.01), k=1, sample_size=1000000))
    # elapsed_time = time.time() - start
    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # start = time.time()
    # print(approx_W_k(Sigma=np.eye(32)*(0.01), k=1, sample_size=1000000))
    # elapsed_time = time.time() - start
    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # # MLE truncated normal
    # dim = 32
    # Sigma = np.eye(dim) * 1
    # sample_size = 10000000
    # k = 1
    # alpha = 0.9
    # points = np.random.multivariate_normal(np.zeros(dim), Sigma, sample_size)
    # dists = np.linalg.norm(points, axis=1)
    # points = points[np.where(dists <= np.pi / np.sqrt(k))[0], :]
    # print(points.shape)

    # start = time.time()
    # param_sigma, lik = mle_truncated_normal_(
    #     points,
    #     sigma_min=0.001,
    #     sigma_max=100,
    #     k=1,
    #     sample_size=10000,
    #     n_iter=1000000,
    #     learning_rate=0.01,
    #     alpha=0.9,
    #     early_stopping=100,
    #     verbose=False
    # )
    # print(param_sigma, lik)
    # elapsed_time = time.time() - start
    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # # MLE truncated normal
    # dim = 3
    # Sigma = np.eye(dim) * 1
    # sample_size = 1000000
    # k = 1
    # alpha = 0.9
    # points = np.random.multivariate_normal(np.zeros(dim), Sigma, sample_size)
    # dists = np.linalg.norm(points, axis=1)
    # points = points[np.where(dists <= np.pi / np.sqrt(k))[0], :]
    # print(points.shape)
    # points= torch.tensor(points).reshape((1, points.shape[0], dim)).to("cuda:0")

    # start = time.time()
    # param_sigma, lik = mle_truncated_normal_gpu(
    #     points,
    #     sigma_min=0.001,
    #     sigma_max=100.0,
    #     k=1,
    #     sample_size=10000,
    #     n_iter=1000000,
    #     learning_rate=0.01,
    #     alpha=0.9,
    #     sigma_init=torch.tensor(np.ones(dim)*0.001).to("cuda:0").float(),
    #     device="cuda:0",
    #     early_stopping=100,
    #     verbose=True
    # )
    # print(param_sigma, lik)
    # elapsed_time = time.time() - start
    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # # MLE normal
    # dim = 3
    # Sigma = np.eye(dim) * 2
    # sample_size = 10000
    # k = 1
    # points = np.random.multivariate_normal(np.zeros(dim), Sigma, sample_size)
    # print(points.shape)

    # param_sigma, lik = mle_normal(
    #     points,
    #     sigma_min=0.001,
    #     sigma_max=100,
    #     k=1,
    #     sample_size=10000,
    #     n_iter=1000000,
    #     alpha=0.9,
    #     learning_rate=0.01
    # )
    # print(lik)

    # MLE truncated normal
    dim = 64
    Sigma = np.eye(dim) * 1
    sample_size = 100000
    k = 1
    alpha = 0.9
    set_points = []

    for i in range(1):
        points = np.random.multivariate_normal(
            np.zeros(dim), Sigma, sample_size)
        dists = np.linalg.norm(points, axis=1)
        points = points[np.where(dists <= np.pi / np.sqrt(k))[0], :]
        print(points.shape)
        set_points.append(points[:900000, :])

    set_points = np.array(set_points)[0]
    print("a")
    set_points = torch.tensor(set_points).to("cuda:0").float()
    print(set_points.shape)

    start = time.time()
    param_sigma, lik = mle_truncated_normal_gpu(
        set_points,
        sigma_min=0.001,
        sigma_max=100.0,
        k=1,
        sample_size=10000,
        n_iter=1000000,
        learning_rate=0.01,
        alpha=0.9,
        sigma_init=torch.tensor(np.ones(dim)).to("cuda:0").float()*0.1,
        device="cuda:0",
        early_stopping=100,
        verbose=False
    )
    print(param_sigma, lik)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # parametric_complexity = calc_spherical_complexity(
    #     dim=2,
    #     sample_size=100,
    #     n_samples=10,
    #     sigma_min=0.001,
    #     sigma_max=100,
    #     k=1,
    #     n_iter=1000000,
    #     learning_rate=0.0001,
    #     alpha=0.9,
    #     sample_size_for_W=100,
    #     early_stopping=1000000
    # )
