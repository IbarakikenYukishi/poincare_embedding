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
from embed import create_dataset, get_unobserved, Graph, SamplingGraph, RSGD, Poincare, calc_lik_pc_cpu

if __name__ == '__main__':
    # データセット作成
    params_dataset = {
        'n_nodes': 1000,
        'n_dim': 32,
        'R': 10,
        'sigma': 1,
        'T': 2
    }

    # パラメータ
    burn_epochs = 10
    burn_batch_size = 1024
    learning_rate = 10.0*burn_batch_size/32  # batchサイズに対応して学習率変更
    # SNML用
    snml_n_iter = 10
    snml_learning_rate = 0.1
    snml_n_max_data = 500
    # それ以外
    loader_workers = 16
    print("loader_workers: ", loader_workers)
    shuffle = True
    sparse = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_graphs = 1


    model_n_dims = [2, 4, 8, 16, 32, 64, 128, 256]

    for n_graph in range(n_graphs):  # データ5本で性能比較
        result = pd.DataFrame()
        print("n_graph:", n_graph)
        # 隣接行列
        adj_mat = hyperbolic_geometric_graph(
            n_nodes=params_dataset['n_nodes'],
            n_dim=params_dataset['n_dim'],
            R=params_dataset['R'],
            sigma=params_dataset['sigma'],
            T=params_dataset['T']
        )
        train, val = create_dataset(
            adj_mat=adj_mat,
            n_max_positives=10,
            n_max_negatives=100,
            val_size=0.00002
        )

        # 平均次数が少なくなるように手で調整する用
        print('average degree:', np.sum(adj_mat) / len(adj_mat))

        u_adj_mat = get_unobserved(adj_mat, train)

        print('# of data in train:', len(train))
        print('# of data in val:', len(val))

        print("start burn-in")


        for model_n_dim in model_n_dims:
            train_ = deepcopy(train)
            u_adj_mat_ = deepcopy(u_adj_mat)

            print("model_n_dim:", model_n_dim)
            # burn-inでの処理
            start = time.time()

            dataloader = DataLoader(
                Graph(train_),
                shuffle=shuffle,
                batch_size=burn_batch_size,
                num_workers=loader_workers,
                pin_memory=True
            )

            # Rは決め打ちするとして、Tは後々平均次数とRから推定する必要がある。
            # 平均次数とかから逆算できる気がする。
            model = Poincare(
                n_nodes=params_dataset['n_nodes'],
                n_dim=model_n_dim,  # モデルの次元
                R=params_dataset['R'],
                T=params_dataset['T'],
                sparse=True,
                init_range=0.001
            )
            model.to(device)  # GPUならGPUにおくっておく。

            # 最適化関数。
            rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
                        R=params_dataset['R'])

            loss_history = []

            for epoch in range(burn_epochs):
                if epoch != 0 and epoch % 10 == 0:  # 10 epochごとに学習率を減少
                    rsgd.param_groups[0]["learning_rate"] /= 5
                losses = []
                for pairs, labels in dataloader:
                    pairs = pairs.to(device)
                    labels = labels.to(device)
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
                    pairs = pairs.to(device)
                    labels = labels.to(device)
                    loss = model(pairs, labels).sum().item()
                    basescore += 2 * loss
            print('basescore:', basescore)

            elapsed_time = time.time() - start
            print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

            model.to("cpu")

            # 以下ではmodelのみを流用する。
            # snmlの計算処理
            # こっちは1サンプルずつやる
            dataloader_snml = DataLoader(
                Graph(val),
                shuffle=shuffle,
                batch_size=1,
                num_workers=0,
            )

            snml_codelength = 0
            snml_codelength_history = []
            aic_history = []
            bic_history = []

            for data_idx, data in enumerate(dataloader_snml):
                # validationのデータ
                val_pair, val_label = data
                # print(val_pair)
                # print(val_label)

                # parametric complexityのサンプリングによる計算
                sampling_data = SamplingGraph(
                    adj_mat=u_adj_mat_,
                    n_max_data=snml_n_max_data,
                    positive_size=1 / 2  # 一様サンプリング以外は実装の修正が必要。
                )
                pairs, labels, n_possibles = sampling_data.get_all_data()
                print("n_possibles:", n_possibles)

                lik, snml_pc = calc_lik_pc_cpu(model, val_pair, val_label, pairs, labels, n_possibles,
                                  snml_n_iter, snml_learning_rate, params_dataset['R'], train_)

                basescore += 2 * lik
                snml_codelength += lik + snml_pc
                aic_history.append(
                    basescore + 2 * (model_n_dim * params_dataset["n_nodes"]))
                bic_history.append(
                    basescore + (model_n_dim * params_dataset["n_nodes"]) * np.log(len(train_) + 1))

                print('snml_codelength ', data_idx, ':', snml_codelength)
                # print('-log p:', loss.item())
                # print('snml_pc:', snml_pc)
                print('AIC ', data_idx, ':', aic_history[-1])
                print('BIC ', data_idx, ':', bic_history[-1])

                # valで使用したデータの削除
                u_adj_mat_[val_pair[0, 0], val_pair[0, 1]] = -1
                u_adj_mat_[val_pair[0, 1], val_pair[0, 0]] = -1

                # データセットを更新する。
                val_ = np.array(
                    [val_pair[0, 0], val_pair[0, 1], val_label[0]]).reshape((1, -1))
                train_ = np.concatenate([train_, val_], axis=0)

                snml_codelength_history.append(snml_codelength)

            del u_adj_mat_
            del train_
            del dataloader
            del dataloader_snml
            gc.collect()

            result["snml_codelength_dim" +
                   str(model_n_dim)] = snml_codelength_history
            result["aic_dim" + str(model_n_dim)] = aic_history
            result["bic_dim" + str(model_n_dim)] = bic_history
            result.to_csv("result_" + str(n_graph) + ".csv", index=False)

        del train
        del val
        del adj_mat
        del u_adj_mat

        gc.collect()

        result.to_csv("result_" + str(n_graph) + ".csv", index=False)
