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
from datasets import hyperbolic_geometric_graph
from embed import create_dataset, get_unobserved, Graph, SamplingGraph, RSGD, Poincare, calc_lik_pc_cpu, calc_lik_pc_gpu

if __name__ == '__main__':

    # パラメータ
    burn_epochs = 10
    burn_batch_size = 4096
    learning_rate = 10.0 * burn_batch_size / 32  # batchサイズに対応して学習率変更
    # それ以外
    loader_workers = 16
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_n_dim_list = [2, 3, 4, 5, 6, 7, 8]
    n_nodes_list = [1000, 2000, 4000, 8000, 16000, 32000, 64000]

    for n_nodes in n_nodes_list:

        # データ読み込み
        # dataset = np.load('dataset/graph_'+str(n_nodes)+'.npy', allow_pickle='TRUE').item() # object型なので、itemを付けないと辞書として使えない。
        # adj_mat = dataset["adj_mat"]
        # params_adj_mat = dataset["params_adj_mat"]
        # train = dataset["train"]
        params_adj_mat = {
            'n_nodes': n_nodes,
            'n_dim': 5,
            'R': 10,
            'sigma': 1,
            'T': 2
        }
        adj_mat, _ = hyperbolic_geometric_graph(
            n_nodes=params_adj_mat["n_nodes"],
            n_dim=params_adj_mat["n_dim"],
            R=params_adj_mat["R"],
            sigma=params_adj_mat["sigma"],
            T=params_adj_mat["T"]
        )
        params_dataset = {
            "n_max_positives": params_adj_mat["n_nodes"],  # 全てサンプリング
            "n_max_negatives": params_adj_mat["n_nodes"],  # 全てサンプリング
            "val_size": 0
        }
        train, val = create_dataset(
            adj_mat,
            params_dataset["n_max_positives"],
            params_dataset["n_max_negatives"],
            params_dataset["val_size"]
        )

        print('# of data in train:', len(train))
        # 平均次数が少なくなるように手で調整する用
        print('average degree:', np.sum(adj_mat) / len(adj_mat))

        result = pd.DataFrame()

        for model_n_dim in model_n_dim_list:
            aic_history = []
            bic_history = []
            basescore_history = []

            print("model_n_dim:", model_n_dim)

            # train_ = deepcopy(train)

            # burn-inでの処理
            start = time.time()

            dataloader = DataLoader(
                Graph(train),
                shuffle=shuffle,
                batch_size=burn_batch_size,
                num_workers=loader_workers,
                pin_memory=True
            )

            # Rは決め打ちするとして、Tは後々平均次数とRから推定する必要がある。
            # 平均次数とかから逆算できる気がする。
            model = Poincare(
                n_nodes=params_adj_mat['n_nodes'],
                n_dim=model_n_dim,  # モデルの次元
                R=params_adj_mat['R'],
                T=params_adj_mat['T'],
                sparse=True,
                init_range=0.001
            )

            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            model.to(device)  # GPUならGPUにおくっておく。

            # 最適化関数。
            rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
                        R=params_adj_mat['R'])

            loss_history = []

            for epoch in range(burn_epochs):
                if epoch != 0 and epoch % 2 == 0:  # 5 epochごとに学習率を減少
                    rsgd.param_groups[0]["learning_rate"] /= 5
                losses = []
                for pairs, labels in dataloader:
                    # pairs = pairs.to(device)
                    # labels = labels.to(device)
                    rsgd.zero_grad()
                    loss = model(pairs, labels).mean()
                    loss.backward()
                    rsgd.step()
                    losses.append(loss)

                loss_history.append(torch.Tensor(losses).mean().item())
                print("epoch:", epoch, ", loss:",
                      torch.Tensor(losses).mean().item())

            with torch.no_grad():
                # -2*log(p)の計算
                basescore = 0
                for pairs, labels in dataloader:
                    # print(model(pairs, labels))
                    # pairs = pairs.to(device)
                    # labels = labels.to(device)
                    loss = model(pairs, labels).sum().item()
                    basescore += 2 * loss
            print('basescore:', basescore)

            elapsed_time = time.time() - start
            print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            basescore_history.append(basescore)
            aic_history.append(
                basescore + 2 * (model_n_dim * params_adj_mat["n_nodes"]))
            bic_history.append(
                basescore + (model_n_dim * params_adj_mat["n_nodes"]) * np.log(len(train)))

            print('basescore :', basescore_history[-1])
            print('AIC :', aic_history[-1])
            print('BIC :', bic_history[-1])

            # del u_adj_mat_
            # del train_
            del dataloader
            del model
            # del dataloader_snml
            gc.collect()

            result["basescore_dim_" + str(model_n_dim)] = basescore_history
            result["aic_dim_" + str(model_n_dim)] = aic_history
            result["bic_dim_" + str(model_n_dim)] = bic_history
            result.to_csv("results/N_dcy/result_aic_bic_"+str(n_nodes)+".csv", index=False)

