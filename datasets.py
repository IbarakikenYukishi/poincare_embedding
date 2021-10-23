import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


INTEGRAL_DIV = 10000


def e_dist(u, v):
    return np.sqrt(np.sum((u - v)**2))


def h_dist(u_e, v_e):
    ret=1
    ret+=(2*e_dist(u_e, v_e)**2)/((1 - e_dist(0, u_e)**2)*(1 - e_dist(0, v_e)**2))
    # print('u_e', u_e)
    # print('v_e', v_e)
    # print('u_e_dist:', e_dist(0, u_e)**2)
    # print('v_e_dist:', e_dist(0, v_e)**2)
    # print('u_v_e_dist:', e_dist(u_e, v_e)**2)
    # print('ret:', ret)
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


def create_dataset(n_nodes, n_dim, R, sigma, T):
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
    # print(x_polar)
    # 直交座標に変換(Euclid)
    x_e = convert_euclid(x_polar)
    # print(x_e)
    # print(np.sum(x_e**2, axis=1))

    adj_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            distance=h_dist(x_e[i], x_e[j])
            # distance=h_dist(np.array([0.9999999,0,0]), np.array([-0.9999999,0,0]))
            prob=connection_prob(distance, R, T)
            # print(distance)
            # print(prob)
            if np.random.uniform(0,1)<prob:
                adj_mat[i,j]=1
                adj_mat[j,i]=1

    # print(adj_mat)
    # print(np.sum(adj_mat, axis=1))
    hist=np.sum(adj_mat, axis=1)
    # plt.hist(hist)
    # plt.show()
    # plt.pause(1)


def convert_euclid(x_polar):
    n_nodes = x_polar.shape[0]
    n_dim = x_polar.shape[1]
    x_euclid = np.zeros((n_nodes, n_dim))
    radius = np.sqrt((np.cosh(x_polar[:, 0]) - 1) / (np.cosh(x_polar[:, 0]) + 1))
    for i in range(n_dim):
        x_euclid[:, i] = radius
        for j in range(0, i + 1):
            if j + 1 < n_dim:
                if j == i:
                    x_euclid[:, i] *= np.cos(x_polar[:, j + 1])
                else:
                    x_euclid[:, i] *= np.sin(x_polar[:, j + 1])
    return x_euclid

if __name__=='__main__':
    adj_mat=create_dataset(n_nodes=1000, n_dim=10, R=10, sigma=1, T=2)

# print(calc_dist_r(2, 1, 30))
# print(calc_dist_angle(3, 1))
