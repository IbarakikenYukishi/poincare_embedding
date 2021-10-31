import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

INTEGRAL_DIV = 10000


def e_dist(u, v):
    return np.sqrt(np.sum((u - v)**2))


def h_dist(u_e, v_e):
    ret = 1
    ret += (2 * e_dist(u_e, v_e)**2) / \
        ((1 - e_dist(0, u_e)**2) * (1 - e_dist(0, v_e)**2))
    return np.arccosh(ret)


def connection_prob(d, R, T):
    """
    接続確率
    """
    return 1 / (1 + np.exp((d - R) / T))


def calc_dist_angle(n_dim, n, div=INTEGRAL_DIV):
    # nは1からn_dim-1となることが想定される。0次元目はr
    if n_dim - 1 == n:
        theta_array = 2 * np.pi * np.arange(0, div + 1) / div
        cum_dens = theta_array / (2 * np.pi)
    else:
        theta_array = np.pi * np.arange(0, div + 1) / div
        numerator = lambda theta: np.sin(theta)**(n_dim - 1 - n)
        cum_dens = []
        for theta in theta_array:
            cum_dens.append(integrate.quad(numerator, 0, theta)[0])
        cum_dens = np.array(cum_dens) / np.max(cum_dens)
    return theta_array, cum_dens


def calc_dist_r(n_dim, sigma, R, div=INTEGRAL_DIV):
    # n_dimかRが大きくなると現状だと数値積分があまりうまくいかない。divを増やす必要がある。
    # 発散を防ぐために、exp(sigma*R*(n_dim-1))/(2**(n_dim-1))(分子の積分の支配項)で割ってある。
    numerator = lambda r: (
        (np.exp(sigma * (r - R)) - np.exp(-sigma * (r + R))))**(n_dim - 1)
    r_array = R * np.arange(0, div + 1) / div
    cum_dens = []
    for r in r_array:
        cum_dens.append(integrate.quad(numerator, 0, r)[0])
    cum_dens = np.array(cum_dens) / np.max(cum_dens)
    return r_array, cum_dens


def hyperbolic_geometric_graph(n_nodes, n_dim, R, sigma, T):
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
    # 対角
    sampling_mat = np.random.uniform(0, 1, adj_mat.shape)
    sampling_mat = np.triu(
        sampling_mat) + np.triu(sampling_mat).T - np.diag(sampling_mat.diagonal())

    # できる限り行列演算を駆使して計算時間を短くする。
    norm_x_e_2 = np.sum(x_e**2, axis=1).reshape((-1, 1))
    denominator_mat = (1 - norm_x_e_2) * (1 - norm_x_e_2.T)
    numerator_mat = norm_x_e_2 + norm_x_e_2.T
    numerator_mat -= 2 * x_e.dot(x_e.T)
    # arccoshのエラー対策
    for i in range(n_nodes):
        numerator_mat[i, i] = 0
    adj_mat = np.arccosh(1 + 2 * numerator_mat / denominator_mat)
    adj_mat = connection_prob(adj_mat, R, T)
    # 対角成分は必要ない
    for i in range(n_nodes):
        adj_mat[i, i] = 0
    adj_mat = np.where(sampling_mat < adj_mat, 1, 0)

    print("adj mat generated")

    return adj_mat


def convert_euclid(x_polar):
    n_nodes = x_polar.shape[0]
    n_dim = x_polar.shape[1]
    x_euclid = np.zeros((n_nodes, n_dim))
    radius = np.sqrt(
        (np.cosh(x_polar[:, 0]) - 1) / (np.cosh(x_polar[:, 0]) + 1))
    for i in range(n_dim):
        x_euclid[:, i] = radius
        for j in range(0, i + 1):
            if j + 1 < n_dim:
                if j == i:
                    x_euclid[:, i] *= np.cos(x_polar[:, j + 1])
                else:
                    x_euclid[:, i] *= np.sin(x_polar[:, j + 1])
    return x_euclid

if __name__ == '__main__':
    adj_mat = hyperbolic_geometric_graph(
        n_nodes=10000, n_dim=10, R=10, sigma=1, T=10)
    print(adj_mat)
