import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate
from copy import deepcopy
from utils.utils import integral_sinh, calc_likelihood_list

np.random.seed(0)

INTEGRAL_DIV = 10000


def connection_prob(d, R, beta):
    """
    接続確率
    """
    return 1 / (1 + beta * np.exp((d - R)))


def integral_sin(n, theta):
    if n == 0:
        return theta
    elif n == 1:
        return 1 - np.cos(theta)
    else:
        return -np.cos(theta) * (np.sin(theta)**(n - 1)) / n + ((n - 1) / n) * integral_sin(n - 2, theta)


def calc_dist_angle(n_dim, n, div=INTEGRAL_DIV):
    # nは1からn_dim-1となることが想定される。0次元目はr
    if n_dim - 1 == n:
        theta_array = 2 * np.pi * np.arange(0, div + 1) / div
        cum_dens = theta_array / (2 * np.pi)
    else:
        theta_array = np.pi * np.arange(0, div + 1) / div
        cum_dens = []
        for theta in theta_array:
            cum_dens.append(integral_sin(n_dim - 1 - n, theta))
        cum_dens = np.array(cum_dens) / np.max(cum_dens)
    return theta_array, cum_dens


def calc_dist_r(n_dim, sigma, R, div=INTEGRAL_DIV):
    # n_dimかRが大きくなると現状だと数値積分があまりうまくいかない。divを増やす必要がある。
    # 発散を防ぐために、exp(sigma*R*(n_dim-1))/(2**(n_dim-1))(分子の積分の支配項)で割ってある。

    def integral_sinh_(n, r):  # (exp(sigma*R)/2)^(D-1)で割った結果
        if n == 0:
            return r * (2 * np.exp(-sigma * R))**(n_dim - 1)
        elif n == 1:
            return 1 / sigma * (np.exp(sigma * (r - R)) + np.exp(- sigma * (r + R)) - 2 * np.exp(-sigma * R)) * (2 * np.exp(-sigma * r))**(n_dim - 2)
        else:
            ret = 1 / (sigma * n)
            ret = ret * (np.exp(sigma * (r - R)) - np.exp(-sigma * (r + R))
                         )**(n - 1) * (np.exp(sigma * (r - R)) + np.exp(-sigma * (r + R)))
            ret = ret * (2 * np.exp(-sigma * R)
                         )**(n_dim - 1 - n)
            return ret - (n - 1) / n * integral_sinh_(n - 2, r)
    r_array = R * np.arange(0, div + 1) / div
    cum_dens = []
    for r in r_array:
        cum_dens.append(integral_sinh_(n=n_dim - 1, r=r))
    cum_dens = np.array(cum_dens) / np.max(cum_dens)
    return r_array, cum_dens


def hyperbolic_geometric_graph(n_nodes, n_dim, R, sigma, beta):
    # TODO: プログラム前半部分も実行時間を短くする。
    # 現状は次元の2乗オーダーの計算量
    # n_dimは2以上で
    x_polar = np.random.uniform(0, 1, (n_nodes, n_dim))
    # 逆関数法で点を双曲空間からサンプリング
    # 双曲空間の意味での極座標で表示
    for i in range(n_dim):
        if i == 0:
            val_array, cum_dens = calc_dist_r(n_dim, sigma, R)
        else:
            val_array, cum_dens = calc_dist_angle(n_dim, i)
        for j in range(n_nodes):
            idx = np.max(np.where(cum_dens <= x_polar[j, i])[0])
            x_polar[j, i] = val_array[idx]
    # 直交座標に変換(Euclid)
    print('sampling ended')

    x_e = convert_euclid(x_polar)

    print('convert euclid ended')

    adj_mat = np.zeros((n_nodes, n_nodes))
    # サンプリング用の行列
    sampling_mat = np.random.uniform(0, 1, adj_mat.shape)
    sampling_mat = np.triu(
        sampling_mat) + np.triu(sampling_mat).T - np.diag(sampling_mat.diagonal())

    # lorentz scalar product
    first_term = - x_e[:, :1] * x_e[:, :1].T
    remaining = x_e[:, 1:].dot(x_e[:, 1:].T)
    adj_mat = - (first_term + remaining)

    for i in range(n_nodes):
        adj_mat[i, i] = 1
    # distance matrix
    adj_mat = np.arccosh(adj_mat)
    # probability matrix
    adj_mat = connection_prob(adj_mat, R, beta)

    for i in range(n_nodes):
        adj_mat[i, i] = 0

    adj_mat = np.where(sampling_mat < adj_mat, 1, 0)

    print(adj_mat)

    print("adj mat generated")

    return adj_mat, x_e


def convert_euclid(x_polar):
    n_nodes = x_polar.shape[0]
    n_dim = x_polar.shape[1]
    x_euclid = np.zeros((n_nodes, n_dim + 1))
    x_euclid[:, 0] = np.cosh(x_polar[:, 0])
    for i in range(n_dim):
        x_euclid[:, i + 1] = np.sinh(x_polar[:, 0])
        for j in range(0, i + 1):
            if j + 1 < n_dim:
                if j == i:
                    x_euclid[:, i + 1] *= np.cos(x_polar[:, j + 1])
                else:
                    x_euclid[:, i + 1] *= np.sin(x_polar[:, j + 1])
    return x_euclid


def create_dataset(
    adj_mat,
    n_max_positives=2,
    n_max_negatives=10,
    val_size=0.1
):
    # データセットを作成し、trainとvalidationに分ける
    _adj_mat = deepcopy(adj_mat)
    n_nodes = _adj_mat.shape[0]
    for i in range(n_nodes):
        _adj_mat[i, i] = -1
    # -1はサンプリング済みの箇所か対角要素

    data = []

    for i in range(n_nodes):
        # positiveをサンプリング
        idx_positives = np.where(_adj_mat[i, :] == 1)[0]
        idx_negatives = np.where(_adj_mat[i, :] == 0)[0]
        idx_positives = np.random.permutation(idx_positives)
        idx_negatives = np.random.permutation(idx_negatives)
        n_positives = min(len(idx_positives), n_max_positives)
        n_negatives = min(len(idx_negatives), n_max_negatives)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_positives[0:n_positives]:
            data.append((i, j, 1))  # positive sample

        # 隣接行列から既にサンプリングしたものを取り除く
        _adj_mat[i, idx_positives[0:n_positives]] = -1
        _adj_mat[idx_positives[0:n_positives], i] = -1

        for j in idx_negatives[0:n_negatives]:
            data.append((i, j, 0))  # positive sample

        # 隣接行列から既にサンプリングしたものを取り除く
        _adj_mat[i, idx_negatives[0:n_negatives]] = -1
        _adj_mat[idx_negatives[0:n_negatives], i] = -1

    data = np.random.permutation(data)

    train = data[0:int(len(data) * (1 - val_size))]
    val = data[int(len(data) * (1 - val_size)):]
    return train, val


def create_dataset_for_basescore(
    adj_mat,
    n_max_samples,
):
    # データセットを作成し、trainとvalidationに分ける
    _adj_mat = deepcopy(adj_mat)
    n_nodes = _adj_mat.shape[0]
    for i in range(n_nodes):
        _adj_mat[i, i] = -1
    # -1はサンプリング済みの箇所か対角要素

    data = []
    # print(np.sum(_adj_mat))

    for i in range(n_nodes):
        idx_samples = np.where(_adj_mat[i, :] != -1)[0]
        idx_samples = np.random.permutation(idx_samples)
        n_samples = min(len(idx_samples), n_max_samples)

        # node iを固定した上で、positiveなnode jを対象とする。それに対し、
        for j in idx_samples[0:n_samples]:
            data.append((i, j, _adj_mat[i, j]))

        # 隣接行列から既にサンプリングしたものを取り除く
        # _adj_mat[i, idx_samples[0:n_samples]] = -1
        # _adj_mat[idx_samples[0:n_samples], i] = -1

    # data = np.random.permutation(data)

    # train = data[0:int(len(data) * (1 - val_size))]
    # val = data[int(len(data) * (1 - val_size)):]
    print(np.sum(_adj_mat))

    return data


def create_test_for_link_prediction(
    adj_mat,
    params_dataset
):
    # testデータとtrain_graphを作成する
    n_total_positives = np.sum(adj_mat) / 2
    n_samples_test = int(n_total_positives * 0.1)
    n_neg_samples_per_positive = 1  # positive1つに対してnegativeをいくつサンプリングするか

    # positive sampleのサンプリング
    train_graph = np.copy(adj_mat)
    # 対角要素からはサンプリングしない
    for i in range(params_dataset["n_nodes"]):
        train_graph[i, i] = -1

    positive_samples = np.array(np.where(train_graph == 1)).T
    # 実質的に重複している要素を削除
    positive_samples_ = []
    for p in positive_samples:
        if p[0] > p[1]:
            positive_samples_.append([p[0], p[1]])
    positive_samples = np.array(positive_samples_)

    positive_samples = np.random.permutation(positive_samples)[:n_samples_test]

    # サンプリングしたデータをtrain_graphから削除
    for t in positive_samples:
        train_graph[t[0], t[1]] = -1
        train_graph[t[1], t[0]] = -1

    # negative sampleのサンプリング
    # permutationが遅くなるので直接サンプリングする
    negative_samples = []
    while len(negative_samples) < n_samples_test * n_neg_samples_per_positive:
        u = np.random.randint(0, params_dataset["n_nodes"])
        v = np.random.randint(0, params_dataset["n_nodes"])
        if train_graph[u, v] != 0:
            continue
        else:
            negative_samples.append([u, v])
            train_graph[u, v] = -1
            train_graph[v, u] = -1

    negative_samples = np.array(negative_samples)

    # これは重複を許す
    lik_data = create_dataset_for_basescore(
        adj_mat=train_graph,
        n_max_samples=int((params_dataset["n_nodes"] - 1) * 0.1)
    )

    return positive_samples, negative_samples, train_graph, lik_data


# def integral_sinh(n, n_dim, R, sigma):  # (exp(sigma*R)/2)^(D-1)で割った結果
#     if n == 0:
#         return R * (2 * np.exp(-sigma * R))**(n_dim - 1)
#     elif n == 1:
#         return 1 / sigma * (1 + np.exp(-2 * sigma * R) - 2 * np.exp(-sigma * R)) * (2 * np.exp(-sigma * R))**(n_dim - 2)
#     else:
#         ret = 1 / (sigma * n)
#         ret = ret * (1 - np.exp(-2 * sigma * R)
#                      )**(n - 1) * (1 + np.exp(-2 * sigma * R))
#         ret = ret * (2 * np.exp(-sigma * R)
#                      )**(n_dim - 1 - n)
# return ret - (n - 1) / n * integral_sinh(n=n - 2, n_dim=n_dim, R=R,
# sigma=sigma)


def calc_likelihood(n_nodes, n_dim, sigma, R, sigma_min, sigma_max):

    x_polar = np.random.uniform(0, 1, (n_nodes))
    # 逆関数法で点を双曲空間からサンプリング
    # 双曲空間の意味での極座標で表示
    val_array, cum_dens = calc_dist_r(n_dim, sigma, R)
    for j in range(n_nodes):
        idx = np.max(np.where(cum_dens <= x_polar[j])[0])
        x_polar[j] = val_array[idx]

    r = x_polar

    print(r)

    sigma_list, lik, sigma_hat = calc_likelihood_list(
        r, n_dim, R, sigma_min, sigma_max)

    print(sigma_hat)

    plt.plot(sigma_list, lik)
    plt.savefig("test.png")


if __name__ == '__main__':

    # calc_likelihood(n_nodes=1000, n_dim=3, sigma=1,
    #                 R=10, sigma_min=0.5, sigma_max=10)

    n_dim_true_list = [8, 16]
    n_nodes_list = [400, 800, 1600, 3200, 6400]
    n_graphs = 10

    for n_dim_true in n_dim_true_list:
        for n_nodes in n_nodes_list:
            for n_graph in range(n_graphs):

                params_adj_mat = {
                    'n_nodes': n_nodes,
                    'n_dim': n_dim_true,
                    'R': np.log(n_nodes) - 0.5,
                    'sigma': 0.1,
                    'beta': 0.3
                }
                adj_mat, x_e = hyperbolic_geometric_graph(
                    n_nodes=params_adj_mat["n_nodes"],
                    n_dim=params_adj_mat["n_dim"],
                    R=params_adj_mat["R"],
                    sigma=params_adj_mat["sigma"],
                    beta=params_adj_mat["beta"]
                )

                positive_samples, negative_samples, train_graph, lik_data = create_test_for_link_prediction(
                    adj_mat, params_adj_mat)

                graph_dict = {
                    "params_adj_mat": params_adj_mat,
                    "adj_mat": adj_mat,
                    "positive_samples": positive_samples,
                    "negative_samples": negative_samples,
                    "train_graph": train_graph,
                    "lik_data": lik_data,
                    "x_e": x_e
                }

                # 平均次数が少なくなるように手で調整する用
                print('average degree:', np.sum(adj_mat) / len(adj_mat))

                os.makedirs('dataset/dim_' +
                            str(params_adj_mat['n_dim']), exist_ok=True)
                np.save('dataset/dim_' + str(params_adj_mat['n_dim']) + '/graph_' + str(
    params_adj_mat['n_nodes']) + '_' + str(n_graph) + '.npy', graph_dict)
