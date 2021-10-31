import warnings
warnings.simplefilter('ignore')
# ParallelNativeやpytorchから要求されるwarningを無視する。
import torch
import numpy as np
import pandas as pd
import gc
from torch.utils.data import DataLoader
from datasets import hyperbolic_geometric_graph
from embed import create_dataset, get_unobserved, Graph, SamplingGraph, RSGD, Poincare, calc_pc

if __name__ == '__main__':
    # データセット作成
    params_dataset = {
        'n_nodes': 10000,
        'n_dim': 5,
        'R': 10,
        'sigma': 1,
        'T': 2
    }

    # パラメータ
    burn_epochs = 3
    learning_rate = 10
    burn_batch_size = 32
    # SNML用
    snml_n_iter = 10
    snml_learning_rate = 0.1
    snml_n_max_data = 500
    # それ以外
    loader_workers = 0
    shuffle = True
    sparse = True

    model_n_dims = [2, 3, 4, 5, 6, 7, 8]

    for n_graph in range(1):  # データ5本で性能比較
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
            n_max_positives=3,
            n_max_negatives=30,
            val_size=0.00001
        )

        print(len(val))

        # 平均次数が少なくなるように手で調整する用
        print('average degree:', np.sum(adj_mat) / len(adj_mat))

        u_adj_mat = get_unobserved(adj_mat, train)

        print("start burn-in")

        for model_n_dim in model_n_dims:
            print("model_n_dim:", model_n_dim)
            # burn-inでの処理
            dataloader = DataLoader(
                Graph(train),
                shuffle=shuffle,
                batch_size=burn_batch_size,
                num_workers=loader_workers,
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
            # 最適化関数。
            rsgd = RSGD(model.parameters(), learning_rate=learning_rate,
                        R=params_dataset['R'])

            loss_history = []

            for epoch in range(burn_epochs):
                if epoch != 0 and epoch % 10 == 0:  # 10 epochごとに学習率を減少
                    rsgd.param_groups[0]["learning_rate"] /= 5
                losses = []
                for pairs, labels in dataloader:
                    rsgd.zero_grad()
                    # print(model(pairs, labels))
                    loss = model(pairs, labels).mean()
                    loss.backward()
                    rsgd.step()
                    losses.append(loss)

                loss_history.append(torch.Tensor(losses).mean().item())
                print("epoch:", epoch, ", loss:",
                      torch.Tensor(losses).mean().item())

            # -2*log(p)の計算
            basescore = 0
            for pairs, labels in dataloader:
                # print(model(pairs, labels))
                loss = model(pairs, labels).sum().item()
                basescore += 2 * loss
            print('basescore:', basescore)

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
                pair, label = data

                # parametric complexityのサンプリングによる計算
                sampling_data = SamplingGraph(
                    adj_mat=u_adj_mat,
                    n_max_data=snml_n_max_data,
                    positive_size=1 / 2  # 一様サンプリング以外は実装の修正が必要。
                )
                pairs, labels, n_possibles = sampling_data.get_all_data()

                snml_pc = calc_pc(model, pairs, labels, n_possibles,
                                  snml_n_iter, snml_learning_rate, params_dataset['R'], train)

                # valのデータでの尤度
                # uとvに関わるデータを5個ずつサンプリングする。
                u = pair[0, 0].item()
                v = pair[0, 1].item()
                u_indice = np.union1d(np.where(train[:, 0] == u)[
                                      0], np.where(train[:, 1] == u)[0])
                u_indice = np.random.permutation(u_indice)[0:5]
                v_indice = np.union1d(np.where(train[:, 0] == v)[
                                      0], np.where(train[:, 1] == v)[0])
                v_indice = np.random.permutation(v_indice)[0:5]

                pair_ = torch.cat((pair, torch.Tensor(
                    train[u_indice, 0:2]), torch.Tensor(train[v_indice, 0:2])), dim=0).long()
                label_ = torch.cat((label.reshape(-1, 1), torch.Tensor(train[u_indice, 2]).reshape(
                    -1, 1), torch.Tensor(train[v_indice, 2]).reshape(-1, 1)), dim=0).long()

                rsgd = RSGD(model.parameters(), learning_rate=snml_learning_rate,
                            R=params_dataset['R'])
                for _ in range(snml_n_iter):
                    rsgd.zero_grad()
                    loss = model(pair_, label_).mean()
                    loss.backward()
                    rsgd.step()

                with torch.no_grad():
                    lik = model(pair, label).item()
                    basescore += 2 * lik
                    snml_codelength += lik + snml_pc
                    aic_history.append(
                        basescore + 2 * (model_n_dim * params_dataset["n_nodes"]))
                    bic_history.append(
                        basescore + (model_n_dim * params_dataset["n_nodes"]) * np.log(len(train) + 1))

                print('snml_codelength ', data_idx, ':', snml_codelength)
                # print('-log p:', loss.item())
                # print('snml_pc:', snml_pc)
                print(aic_history[-1])
                print(bic_history[-1])

                # valで使用したデータの削除
                u_adj_mat[pair[0, 0], pair[0, 1]] = -1
                u_adj_mat[pair[0, 1], pair[0, 0]] = -1

                # データセットを更新する。
                val_ = np.array(
                    [pair[0, 0], pair[0, 1], label[0]]).reshape((1, -1))
                train = np.concatenate([train, val_], axis=0)

                snml_codelength_history.append(snml_codelength)

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
