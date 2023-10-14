import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate, stats
# from embed import create_dataset
from copy import deepcopy
import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score, average_precision_score
import matplotlib.pyplot as plt
import matplotlib
# from experiment_wn import is_a_score
import torch
from multiprocessing import Pool
from functools import partial


RESULTS = "results"
loosely_dotted = (0, (1, 10))


def artificial(dataset):
    # D_true_list = [8, 16]
    D_true_list = [8]
    # n_nodes_list = [400, 800, 1600, 3200, 6400]
    n_nodes_list = [400, 800, 1600, 3200]
    # n_nodes_list = [400, 800, 3200]
    n_graphs = 12
    T_gap = 2

    for D_true in D_true_list:
        # if D_true == 8:
        #     label = [0, 0, 1, 0, 0, 0]
        # elif D_true == 16:
        #     label = [0, 0, 0, 1, 0, 0]
        if D_true == 8:
            label = [0, 0, 1, 0, 0]
        elif D_true == 16:
            label = [0, 0, 0, 1, 0]

        for n_nodes in n_nodes_list:
            correct_space_DNML_latent = 0
            correct_space_AIC_latent = 0
            correct_space_BIC_latent = 0
            correct_space_AIC_naive = 0
            correct_space_BIC_naive = 0

            bene_DNML_lorentz_latent = []
            bene_DNML_euclidean_latent = []
            bene_DNML_spherical_latent = []
            bene_AIC_lorentz_latent = []
            bene_AIC_euclidean_latent = []
            bene_AIC_spherical_latent = []
            bene_BIC_lorentz_latent = []
            bene_BIC_euclidean_latent = []
            bene_BIC_spherical_latent = []
            bene_AIC_lorentz_naive = []
            bene_AIC_euclidean_naive = []
            bene_AIC_spherical_naive = []
            bene_BIC_lorentz_naive = []
            bene_BIC_euclidean_naive = []
            bene_BIC_spherical_naive = []
            estimate_DNML_lorentz_latent = []
            estimate_DNML_euclidean_latent = []
            estimate_DNML_spherical_latent = []
            estimate_AIC_lorentz_latent = []
            estimate_AIC_euclidean_latent = []
            estimate_AIC_spherical_latent = []
            estimate_BIC_lorentz_latent = []
            estimate_BIC_euclidean_latent = []
            estimate_BIC_spherical_latent = []
            estimate_AIC_lorentz_naive = []
            estimate_AIC_euclidean_naive = []
            estimate_AIC_spherical_naive = []
            estimate_BIC_lorentz_naive = []
            estimate_BIC_euclidean_naive = []
            estimate_BIC_spherical_naive = []

            df_curvature = pd.DataFrame({})

            for n_graph in range(n_graphs):
                result = pd.read_csv(RESULTS + "/" + dataset + "/dim_" + str(D_true) +
                                     "/result_" + str(n_nodes) + "_" + str(n_graph) + ".csv")
                result = result.fillna(9999999999999)

                # result = result[result["model_n_dims"].isin(
                #     [2, 4, 8, 16, 32, 64])]
                result = result[result["model_n_dims"].isin(
                    [2, 4, 8, 16, 32])]

                result = result.sort_values("model_n_dims")

                D_DNML_lorentz_latent = result["model_n_dims"].values[
                    np.argmin(result["DNML_lorentz_latent"].values)]
                C_DNML_lorentz_latent = np.min(
                    result["DNML_lorentz_latent"].values)
                D_AIC_lorentz_latent = result["model_n_dims"].values[
                    np.argmin(result["AIC_lorentz_latent"].values)]
                C_AIC_lorentz_latent = np.min(
                    result["AIC_lorentz_latent"].values)
                D_BIC_lorentz_latent = result["model_n_dims"].values[
                    np.argmin(result["BIC_lorentz_latent"].values)]
                C_BIC_lorentz_latent = np.min(
                    result["BIC_lorentz_latent"].values)
                D_AIC_lorentz_naive = result["model_n_dims"].values[
                    np.argmin(result["AIC_lorentz_naive"].values)]
                C_AIC_lorentz_naive = np.min(
                    result["AIC_lorentz_naive"].values)
                D_BIC_lorentz_naive = result["model_n_dims"].values[
                    np.argmin(result["BIC_lorentz_naive"].values)]
                C_BIC_lorentz_naive = np.min(
                    result["BIC_lorentz_naive"].values)
                D_DNML_euclidean_latent = result["model_n_dims"].values[
                    np.argmin(result["DNML_euclidean_latent"].values)]
                C_DNML_euclidean_latent = np.min(
                    result["DNML_euclidean_latent"].values)
                D_AIC_euclidean_latent = result["model_n_dims"].values[
                    np.argmin(result["AIC_euclidean_latent"].values)]
                C_AIC_euclidean_latent = np.min(
                    result["AIC_euclidean_latent"].values)
                D_BIC_euclidean_latent = result["model_n_dims"].values[
                    np.argmin(result["BIC_euclidean_latent"].values)]
                C_BIC_euclidean_latent = np.min(
                    result["BIC_euclidean_latent"].values)
                D_AIC_euclidean_naive = result["model_n_dims"].values[
                    np.argmin(result["AIC_euclidean_naive"].values)]
                C_AIC_euclidean_naive = np.min(
                    result["AIC_euclidean_naive"].values)
                D_BIC_euclidean_naive = result["model_n_dims"].values[
                    np.argmin(result["BIC_euclidean_naive"].values)]
                C_BIC_euclidean_naive = np.min(
                    result["BIC_euclidean_naive"].values)
                D_DNML_spherical_latent = result["model_n_dims"].values[
                    np.argmin(result["DNML_spherical_latent"].values)]
                C_DNML_spherical_latent = np.min(
                    result["DNML_spherical_latent"].values)
                D_AIC_spherical_latent = result["model_n_dims"].values[
                    np.argmin(result["AIC_spherical_latent"].values)]
                C_AIC_spherical_latent = np.min(
                    result["AIC_spherical_latent"].values)
                D_BIC_spherical_latent = result["model_n_dims"].values[
                    np.argmin(result["BIC_spherical_latent"].values)]
                C_BIC_spherical_latent = np.min(
                    result["BIC_spherical_latent"].values)
                D_AIC_spherical_naive = result["model_n_dims"].values[
                    np.argmin(result["AIC_spherical_naive"].values)]
                C_AIC_spherical_naive = np.min(
                    result["AIC_spherical_naive"].values)
                D_BIC_spherical_naive = result["model_n_dims"].values[
                    np.argmin(result["BIC_spherical_naive"].values)]
                C_BIC_spherical_naive = np.min(
                    result["BIC_spherical_naive"].values)

                if dataset == "hyperbolic":
                    model_lorentz_latent = torch.load(
                        RESULTS + "/" + dataset + "/" + "dim_" + str(D_true) + "/result_" + str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) + "_lorentz_latent.pth")
                    K_lorentz_latent = 1 / model_lorentz_latent.kappa.detach().cpu().item()
                    model_lorentz_naive = torch.load(
                        RESULTS + "/" + dataset + "/" + "dim_" + str(D_true) + "/result_" + str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) + "_lorentz_naive.pth")
                    K_lorentz_naive = 1 / model_lorentz_naive.kappa.detach().cpu().item()

                    data = np.load('dataset/' + dataset + '/dim_' + str(D_true) + '/graph_' + str(n_nodes) + '_' + str(n_graph) +
                                      '.npy', allow_pickle='TRUE').item()  # object型なので、itemを付けないと辞書として使えない。
                    true_curvature = data["params_adj_mat"]["k"]

                    row = pd.DataFrame({"true_curvature": [true_curvature], "K_lorentz_latent": K_lorentz_latent, "K_lorentz_naive": K_lorentz_naive})
                    df_curvature = pd.concat([df_curvature, row])
                    # print(K_DNML_lorentz_latent)
                    # print(true_curvature)

                if dataset == "spherical":
                    model_spherical_latent = torch.load(
                        RESULTS + "/" + dataset + "/" + "dim_" + str(D_true) + "/result_" + str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) + "_spherical_latent.pth")
                    K_spherical_latent = 1 / model_spherical_latent.kappa.detach().cpu().item()
                    model_spherical_naive = torch.load(
                        RESULTS + "/" + dataset + "/" + "dim_" + str(D_true) + "/result_" + str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) + "_spherical_naive.pth")
                    K_spherical_naive = 1 / model_spherical_naive.kappa.detach().cpu().item()

                    data = np.load('dataset/' + dataset + '/dim_' + str(D_true) + '/graph_' + str(n_nodes) + '_' + str(n_graph) +
                                      '.npy', allow_pickle='TRUE').item()  # object型なので、itemを付けないと辞書として使えない。
                    true_curvature = data["params_adj_mat"]["k"]

                    row = pd.DataFrame({"true_curvature": [true_curvature], "K_spherical_latent": K_spherical_latent, "K_spherical_naive": K_spherical_naive})
                    df_curvature = pd.concat([df_curvature, row])

                    # print(K_DNML_spherical_latent)
                    # print(true_curvature)


                # model_spherical_latent = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
                #     int(D_DNML_spherical_latent)) + "_spherical_latent.pth")
                # K_DNML_spherical_latent = 1 / model_spherical_latent.kappa.detach().cpu().item()
                # model_spherical_latent = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
                #     int(D_DNML_spherical_latent)) + "_spherical_latent.pth")
                # K_AIC_spherical_latent = 1 / model_spherical_latent.kappa.detach().cpu().item()
                # model_spherical_latent = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
                #     int(D_BIC_spherical_latent)) + "_spherical_latent.pth")
                # K_BIC_spherical_latent = 1 / model_spherical_latent.kappa.detach().cpu().item()
                # model_spherical_naive = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
                #     int(D_AIC_spherical_naive)) + "_spherical_naive.pth")
                # K_AIC_spherical_naive = 1 / model_spherical_naive.kappa.detach().cpu().item()
                # model_spherical_naive = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
                #     int(D_BIC_spherical_naive)) + "_spherical_naive.pth")
                # K_BIC_spherical_naive = 1 / model_spherical_naive.kappa.detach().cpu().item()

                if dataset == "hyperbolic":
                    if C_DNML_lorentz_latent == min(C_DNML_lorentz_latent, C_DNML_euclidean_latent, C_DNML_spherical_latent):
                        correct_space_DNML_latent += 1
                    if C_AIC_lorentz_latent == min(C_AIC_lorentz_latent, C_AIC_euclidean_latent, C_AIC_spherical_latent):
                        correct_space_AIC_latent += 1
                    if C_BIC_lorentz_latent == min(C_BIC_lorentz_latent, C_BIC_euclidean_latent, C_BIC_spherical_latent):
                        correct_space_BIC_latent += 1
                    if C_AIC_lorentz_naive == min(C_AIC_lorentz_naive, C_AIC_euclidean_naive, C_AIC_spherical_naive):
                        correct_space_AIC_naive += 1
                    if C_BIC_lorentz_naive == min(C_BIC_lorentz_naive, C_BIC_euclidean_naive, C_BIC_spherical_naive):
                        correct_space_BIC_naive += 1
                elif dataset == "euclidean":
                    if C_DNML_euclidean_latent == min(C_DNML_lorentz_latent, C_DNML_euclidean_latent, C_DNML_spherical_latent):
                        correct_space_DNML_latent += 1
                    if C_AIC_euclidean_latent == min(C_AIC_lorentz_latent, C_AIC_euclidean_latent, C_AIC_spherical_latent):
                        correct_space_AIC_latent += 1
                    if C_BIC_euclidean_latent == min(C_BIC_lorentz_latent, C_BIC_euclidean_latent, C_BIC_spherical_latent):
                        correct_space_BIC_latent += 1
                    if C_AIC_euclidean_naive == min(C_AIC_lorentz_naive, C_AIC_euclidean_naive, C_AIC_spherical_naive):
                        correct_space_AIC_naive += 1
                    if C_BIC_euclidean_naive == min(C_BIC_lorentz_naive, C_BIC_euclidean_naive, C_BIC_spherical_naive):
                        correct_space_BIC_naive += 1
                elif dataset == "spherical":
                    if C_DNML_spherical_latent == min(C_DNML_lorentz_latent, C_DNML_euclidean_latent, C_DNML_spherical_latent):
                        correct_space_DNML_latent += 1
                    if C_AIC_spherical_latent == min(C_AIC_lorentz_latent, C_AIC_euclidean_latent, C_AIC_spherical_latent):
                        correct_space_AIC_latent += 1
                    if C_BIC_spherical_latent == min(C_BIC_lorentz_latent, C_BIC_euclidean_latent, C_BIC_spherical_latent):
                        correct_space_BIC_latent += 1
                    if C_AIC_spherical_naive == min(C_AIC_lorentz_naive, C_AIC_euclidean_naive, C_AIC_spherical_naive):
                        correct_space_AIC_naive += 1
                    if C_BIC_spherical_naive == min(C_BIC_lorentz_naive, C_BIC_euclidean_naive, C_BIC_spherical_naive):
                        correct_space_BIC_naive += 1

                estimate_DNML_lorentz_latent.append(D_DNML_lorentz_latent)
                estimate_DNML_euclidean_latent.append(D_DNML_euclidean_latent)
                estimate_DNML_spherical_latent.append(D_DNML_spherical_latent)
                estimate_AIC_lorentz_latent.append(D_AIC_lorentz_latent)
                estimate_AIC_euclidean_latent.append(D_AIC_euclidean_latent)
                estimate_AIC_spherical_latent.append(D_AIC_spherical_latent)
                estimate_BIC_lorentz_latent.append(D_BIC_lorentz_latent)
                estimate_BIC_euclidean_latent.append(D_BIC_euclidean_latent)
                estimate_BIC_spherical_latent.append(D_BIC_spherical_latent)
                estimate_AIC_lorentz_naive.append(D_AIC_lorentz_naive)
                estimate_AIC_euclidean_naive.append(D_AIC_euclidean_naive)
                estimate_AIC_spherical_naive.append(D_AIC_spherical_naive)
                estimate_BIC_lorentz_naive.append(D_BIC_lorentz_naive)
                estimate_BIC_euclidean_naive.append(D_BIC_euclidean_naive)
                estimate_BIC_spherical_naive.append(D_BIC_spherical_naive)

                bene_DNML_lorentz_latent.append(
                    label_ranking_average_precision_score([label], [-result["DNML_lorentz_latent"].values]))
                bene_DNML_euclidean_latent.append(
                    label_ranking_average_precision_score([label], [-result["DNML_euclidean_latent"].values]))
                bene_DNML_spherical_latent.append(
                    label_ranking_average_precision_score([label], [-result["DNML_spherical_latent"].values]))
                bene_AIC_lorentz_latent.append(
                    label_ranking_average_precision_score([label], [-result["AIC_lorentz_latent"].values]))
                bene_AIC_euclidean_latent.append(
                    label_ranking_average_precision_score([label], [-result["AIC_euclidean_latent"].values]))
                bene_AIC_spherical_latent.append(
                    label_ranking_average_precision_score([label], [-result["AIC_spherical_latent"].values]))
                bene_BIC_lorentz_latent.append(
                    label_ranking_average_precision_score([label], [-result["BIC_lorentz_latent"].values]))
                bene_BIC_euclidean_latent.append(
                    label_ranking_average_precision_score([label], [-result["BIC_euclidean_latent"].values]))
                bene_BIC_spherical_latent.append(
                    label_ranking_average_precision_score([label], [-result["BIC_spherical_latent"].values]))
                bene_AIC_lorentz_naive.append(
                    label_ranking_average_precision_score([label], [-result["AIC_lorentz_naive"].values]))
                bene_AIC_euclidean_naive.append(
                    label_ranking_average_precision_score([label], [-result["AIC_euclidean_naive"].values]))
                bene_AIC_spherical_naive.append(
                    label_ranking_average_precision_score([label], [-result["AIC_spherical_naive"].values]))
                bene_BIC_lorentz_naive.append(
                    label_ranking_average_precision_score([label], [-result["BIC_lorentz_naive"].values]))
                bene_BIC_euclidean_naive.append(
                    label_ranking_average_precision_score([label], [-result["BIC_euclidean_naive"].values]))
                bene_BIC_spherical_naive.append(
                    label_ranking_average_precision_score([label], [-result["BIC_spherical_naive"].values]))

                # bene_AIC_HGG.append(
                #     label_ranking_average_precision_score([label], [-result["AIC_HGG"].values]))
                # bene_BIC_HGG.append(
                #     label_ranking_average_precision_score([label], [-result["BIC_HGG"].values]))
                # bene_DNML_WND.append(
                #     label_ranking_average_precision_score([label], [-result["DNML_WND"].values]))
                # bene_AIC_WND.append(
                #     label_ranking_average_precision_score([label], [-result["AIC_WND"].values]))
                # bene_BIC_WND.append(
                #     label_ranking_average_precision_score([label], [-result["BIC_WND"].values]))
                # bene_AIC_naive.append(
                #     label_ranking_average_precision_score([label], [-result["AIC_naive"].values]))
                # bene_BIC_naive.append(
                #     label_ranking_average_precision_score([label], [-result["BIC_naive"].values]))
                # bene_MinGE.append(
                # label_ranking_average_precision_score([label],
                # [-result_MinGE["MinGE"].values]))

                # bene_DNML.append(
                #     max(0, 1 - abs(np.log2(D_DNML) - np.log2(D_true)) / T_gap))
                # bene_AIC_latent.append(
                #     max(0, 1 - abs(np.log2(D_AIC_latent) - np.log2(D_true)) / T_gap))
                # bene_BIC_latent.append(
                #     max(0, 1 - abs(np.log2(D_BIC_latent) - np.log2(D_true)) / T_gap))
                # bene_AIC_naive.append(
                #     max(0, 1 - abs(np.log2(D_AIC_naive) - np.log2(D_true)) / T_gap))
                # bene_BIC_naive.append(
                #     max(0, 1 - abs(np.log2(D_BIC_naive) - np.log2(D_true)) / T_gap))
                # bene_MinGE.append(
                # max(0, 1 - abs(np.log2(D_MinGE) - np.log2(D_true)) / T_gap))

                # plt.clf()
                # fig = plt.figure(figsize=(8, 5))
                # ax = fig.add_subplot(111)

                # def normalize(x):
                # return (x - np.min(x.values)) / (np.max(x.values) -
                # np.min(x.values))

                # result["DNML_HGG"] = normalize(
                #     result["DNML_HGG"])
                # result["AIC_HGG"] = normalize(result["AIC_HGG"])
                # result["BIC_HGG"] = normalize(result["BIC_HGG"])
                # result["DNML_WND"] = normalize(
                #     result["DNML_WND"])
                # result["AIC_WND"] = normalize(result["AIC_WND"])
                # result["BIC_WND"] = normalize(result["BIC_WND"])
                # result["AIC_naive"] = normalize(result["AIC_naive"])
                # result["BIC_naive"] = normalize(result["BIC_naive"])
                # result["MinGE"] = normalize(result_MinGE["MinGE"])

                # if dataset == "HGG":
                #     ax.plot(result["model_n_dims"], result[
                #             "DNML_" + dataset], label="DNML-PUD", linestyle="solid", color="black")
                # else:
                #     ax.plot(result["model_n_dims"], result[
                #             "DNML_" + dataset], label="DNML-" + dataset, linestyle="solid", color="black")
                # ax.plot(result["model_n_dims"], result[
                #         "DNML_HGG"], label="DNML-PUD", linestyle="solid", color="black")
                # ax.plot(result["model_n_dims"], result[
                #         "DNML_WND"], label="DNML-WND", linestyle=loosely_dotted, color="black")

                # # ax.plot(result["model_n_dims"], result["AIC_"+dataset],
                # #         label="AIC_"+dataset, color="blue")
                # # ax.plot(result["model_n_dims"], result["BIC_"+dataset],
                # #         label="BIC_"+dataset, color="green")
                # ax.plot(result["model_n_dims"], result["AIC_naive"],
                #         label="AIC", linestyle="dotted", color="black")
                # ax.plot(result["model_n_dims"], result["BIC_naive"],
                #         label="BIC", linestyle="dashed", color="black")
                # ax.plot(result["model_n_dims"], result[
                #     "MinGE"], label="MinGE", linestyle="dashdot", color="black")
                # plt.xscale('log')

                # # plt.xticks(result["model_n_dims"], fontsize=8)
                # plt.xticks([2, 4, 8, 16, 32, 64], fontsize=20)
                # plt.yticks(fontsize=20)
                # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                # plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
                #            borderaxespad=0, fontsize=15)
                # ax.set_xlabel("Dimensionality", fontsize=20)
                # ax.set_ylabel("Normalized Criterion", fontsize=20)
                # plt.tight_layout()
                # os.makedirs(RESULTS + "/" + dataset + "_fig/", exist_ok=True)

                # plt.savefig(RESULTS + "/" + dataset + "_fig/result_" +
                # str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) +
                # ".png")

            if dataset != "euclidean":
                print(df_curvature.groupby("true_curvature").mean())

            bene_DNML_lorentz_latent = np.array(bene_DNML_lorentz_latent)
            bene_DNML_euclidean_latent = np.array(bene_DNML_euclidean_latent)
            bene_DNML_spherical_latent = np.array(bene_DNML_spherical_latent)
            bene_AIC_lorentz_latent = np.array(bene_AIC_lorentz_latent)
            bene_AIC_euclidean_latent = np.array(bene_AIC_euclidean_latent)
            bene_AIC_spherical_latent = np.array(bene_AIC_spherical_latent)
            bene_BIC_lorentz_latent = np.array(bene_BIC_lorentz_latent)
            bene_BIC_euclidean_latent = np.array(bene_BIC_euclidean_latent)
            bene_BIC_spherical_latent = np.array(bene_BIC_spherical_latent)
            bene_AIC_lorentz_naive = np.array(bene_AIC_lorentz_naive)
            bene_AIC_euclidean_naive = np.array(bene_AIC_euclidean_naive)
            bene_AIC_spherical_naive = np.array(bene_AIC_spherical_naive)
            bene_BIC_lorentz_naive = np.array(bene_BIC_lorentz_naive)
            bene_BIC_euclidean_naive = np.array(bene_BIC_euclidean_naive)
            bene_BIC_spherical_naive = np.array(bene_BIC_spherical_naive)

            estimate_DNML_lorentz_latent = np.array(
                estimate_DNML_lorentz_latent)
            estimate_DNML_euclidean_latent = np.array(
                estimate_DNML_euclidean_latent)
            estimate_DNML_spherical_latent = np.array(
                estimate_DNML_spherical_latent)
            estimate_AIC_lorentz_latent = np.array(estimate_AIC_lorentz_latent)
            estimate_AIC_euclidean_latent = np.array(
                estimate_AIC_euclidean_latent)
            estimate_AIC_spherical_latent = np.array(
                estimate_AIC_spherical_latent)
            estimate_BIC_lorentz_latent = np.array(estimate_BIC_lorentz_latent)
            estimate_BIC_euclidean_latent = np.array(
                estimate_BIC_euclidean_latent)
            estimate_BIC_spherical_latent = np.array(
                estimate_BIC_spherical_latent)
            estimate_AIC_lorentz_naive = np.array(estimate_AIC_lorentz_naive)
            estimate_AIC_euclidean_naive = np.array(
                estimate_AIC_euclidean_naive)
            estimate_AIC_spherical_naive = np.array(
                estimate_AIC_spherical_naive)
            estimate_BIC_lorentz_naive = np.array(estimate_BIC_lorentz_naive)
            estimate_BIC_euclidean_naive = np.array(
                estimate_BIC_euclidean_naive)
            estimate_BIC_spherical_naive = np.array(
                estimate_BIC_spherical_naive)

            print("n_nodes:", n_nodes)
            print("dimensionality:", D_true)
            print("correct_space_DNML_latent:", correct_space_DNML_latent / 12)
            # print("correct_space_AIC_latent:", correct_space_AIC_latent)
            # print("correct_space_BIC_latent:", correct_space_BIC_latent)
            print("correct_space_AIC_naive:", correct_space_AIC_naive / 12)
            print("correct_space_BIC_naive:", correct_space_BIC_naive / 12)



            if dataset == "hyperbolic":
                print("DNML_lorentz_latent:",
                      np.mean(bene_DNML_lorentz_latent), "±", np.std(bene_DNML_lorentz_latent))
                # print("AIC_lorentz_latent:",
                #       np.mean(bene_AIC_lorentz_latent), "±", np.std(bene_AIC_lorentz_latent))
                # print("BIC_lorentz_latent:",
                # np.mean(bene_BIC_lorentz_latent), "±",
                # np.std(bene_BIC_lorentz_latent))
                print("AIC_lorentz_naive:",
                      np.mean(bene_AIC_lorentz_naive), "±", np.std(bene_AIC_lorentz_naive))
                print("BIC_lorentz_naive:",
                      np.mean(bene_BIC_lorentz_naive), "±", np.std(bene_BIC_lorentz_naive))
                print("DNML_lorentz_latent:", np.mean(estimate_DNML_lorentz_latent),
                      "±", np.std(estimate_DNML_lorentz_latent))
                # print("AIC_lorentz_latent:", np.mean(estimate_AIC_lorentz_latent),
                #       "±", np.std(estimate_AIC_lorentz_latent))
                # print("BIC_lorentz_latent:", np.mean(estimate_BIC_lorentz_latent),
                #       "±", np.std(estimate_BIC_lorentz_latent))
                print("AIC_lorentz_naive:", np.mean(estimate_AIC_lorentz_naive),
                      "±", np.std(estimate_AIC_lorentz_naive))
                print("BIC_lorentz_naive:", np.mean(estimate_BIC_lorentz_naive),
                      "±", np.std(estimate_BIC_lorentz_naive))
            elif dataset == "euclidean":
                print("DNML_euclidean_latent:",
                      np.mean(bene_DNML_euclidean_latent), "±", np.std(bene_DNML_euclidean_latent))
                # print("AIC_euclidean_latent:",
                #       np.mean(bene_AIC_euclidean_latent), "±", np.std(bene_AIC_euclidean_latent))
                # print("BIC_euclidean_latent:",
                # np.mean(bene_BIC_euclidean_latent), "±",
                # np.std(bene_BIC_euclidean_latent))
                print("AIC_euclidean_naive:",
                      np.mean(bene_AIC_euclidean_naive), "±", np.std(bene_AIC_euclidean_naive))
                print("BIC_euclidean_naive:",
                      np.mean(bene_BIC_euclidean_naive), "±", np.std(bene_BIC_euclidean_naive))
                print("DNML_euclidean_latent:", np.mean(estimate_DNML_euclidean_latent),
                      "±", np.std(estimate_DNML_euclidean_latent))
                # print("AIC_euclidean_latent:", np.mean(estimate_AIC_euclidean_latent),
                #       "±", np.std(estimate_AIC_euclidean_latent))
                # print("BIC_euclidean_latent:", np.mean(estimate_BIC_euclidean_latent),
                #       "±", np.std(estimate_BIC_euclidean_latent))
                print("AIC_euclidean_naive:", np.mean(estimate_AIC_euclidean_naive),
                      "±", np.std(estimate_AIC_euclidean_naive))
                print("BIC_euclidean_naive:", np.mean(estimate_BIC_euclidean_naive),
                      "±", np.std(estimate_BIC_euclidean_naive))
            elif dataset == "spherical":
                print("DNML_spherical_latent:",
                      np.mean(bene_DNML_spherical_latent), "±", np.std(bene_DNML_spherical_latent))
                # print("AIC_spherical_latent:",
                #       np.mean(bene_AIC_spherical_latent), "±", np.std(bene_AIC_spherical_latent))
                # print("BIC_spherical_latent:",
                # np.mean(bene_BIC_spherical_latent), "±",
                # np.std(bene_BIC_spherical_latent))
                print("AIC_spherical_naive:",
                      np.mean(bene_AIC_spherical_naive), "±", np.std(bene_AIC_spherical_naive))
                print("BIC_spherical_naive:",
                      np.mean(bene_BIC_spherical_naive), "±", np.std(bene_BIC_spherical_naive))
                print("DNML_spherical_latent:", np.mean(estimate_DNML_spherical_latent),
                      "±", np.std(estimate_DNML_spherical_latent))
                # print("AIC_spherical_latent:", np.mean(estimate_AIC_spherical_latent),
                #       "±", np.std(estimate_AIC_spherical_latent))
                # print("BIC_spherical_latent:", np.mean(estimate_BIC_spherical_latent),
                #       "±", np.std(estimate_BIC_spherical_latent))
                print("AIC_spherical_naive:", np.mean(estimate_AIC_spherical_naive),
                      "±", np.std(estimate_AIC_spherical_naive))
                print("BIC_spherical_naive:", np.mean(estimate_BIC_spherical_naive),
                      "±", np.std(estimate_BIC_spherical_naive))

            # print("DNML_euclidean_latent:",
            #       np.mean(bene_DNML_euclidean_latent), "±", np.std(bene_DNML_lorentz_latent))
            # print("DNML_spherical_latent:",
            # np.mean(bene_DNML_spherical_latent), "±",
            # np.std(bene_DNML_spherical_latent))

            # print("DNML_" + dataset + ":",
            #       np.mean(bene_DNML), "±", np.std(bene_DNML))
            # print("AIC_" + dataset + ":", np.mean(bene_AIC_latent),
            #       "±", np.std(bene_AIC_latent))
            # print("BIC_" + dataset + ":", np.mean(bene_BIC_latent),
            #       "±", np.std(bene_BIC_latent))
            # print("DNML_HGG:",
            #       np.mean(bene_DNML_HGG), "±", np.std(bene_DNML_HGG))
            # print("AIC_HGG:", np.mean(bene_AIC_HGG),
            #       "±", np.std(bene_AIC_HGG))
            # print("BIC_HGG:", np.mean(bene_BIC_HGG),
            #       "±", np.std(bene_BIC_HGG))
            # print("DNML_WND:",
            #       np.mean(bene_DNML_WND), "±", np.std(bene_DNML_WND))
            # print("AIC_WND:", np.mean(bene_AIC_WND),
            #       "±", np.std(bene_AIC_WND))
            # print("BIC_WND:", np.mean(bene_BIC_WND),
            #       "±", np.std(bene_BIC_WND))
            # print("AIC_naive:", np.mean(bene_AIC_naive),
            #       "±", np.std(bene_AIC_naive))
            # print("BIC_naive:", np.mean(bene_BIC_naive),
            #       "±", np.std(bene_BIC_naive))
            # print("MinGE:", np.mean(bene_MinGE), "±", np.std(bene_MinGE))

            # print("DNML_" + dataset + ":", np.mean(estimate_DNML),
            #       "±", np.std(estimate_DNML))
            # print("AIC_" + dataset + ":", np.mean(estimate_AIC_latent),
            #       "±", np.std(estimate_AIC_latent))
            # print("BIC_" + dataset + ":", np.mean(estimate_BIC_latent),
            #       "±", np.std(estimate_BIC_latent))
            # print("DNML_HGG:", np.mean(estimate_DNML_HGG),
            #       "±", np.std(estimate_DNML_HGG))
            # print("AIC_HGG:", np.mean(estimate_AIC_HGG),
            #       "±", np.std(estimate_AIC_HGG))
            # print("BIC_HGG:", np.mean(estimate_BIC_HGG),
            #       "±", np.std(estimate_BIC_HGG))
            # print("DNML_WND:", np.mean(estimate_DNML_WND),
            #       "±", np.std(estimate_DNML_WND))
            # print("AIC_WND:", np.mean(estimate_AIC_WND),
            #       "±", np.std(estimate_AIC_WND))
            # print("BIC_WND:", np.mean(estimate_BIC_WND),
            #       "±", np.std(estimate_BIC_WND))
            # print("AIC_naive:", np.mean(estimate_AIC_naive),
            #       "±", np.std(estimate_AIC_naive))
            # print("BIC_naive:", np.mean(estimate_BIC_naive),
            #       "±", np.std(estimate_BIC_naive))
            # print("MinGE:", np.mean(estimate_MinGE),
            #       "±", np.std(estimate_MinGE))


def plot_figure(dataset_name, n_graph):

    result = pd.read_csv(RESULTS + "/" + dataset_name +
                         "/dim_8/result_6400_" + str(n_graph) + ".csv")

    dataset = np.load('dataset/' + dataset_name + '/dim_8/graph_6400_' + str(n_graph) +
                      '.npy', allow_pickle='TRUE').item()  # object型なので、itemを付けないと辞書として使えない。
    params = dataset["params_adj_mat"]
    print(params)

    result = result.fillna(9999999999999)
    result = result[result["model_n_dims"].isin(
        [2, 4, 8, 16, 32, 64])]

    # result = result.drop(result.index[[5]])
    D_DNML = result["model_n_dims"].values[
        np.argmin(result["DNML_" + dataset_name].values)]
    # print(result[["beta", "sigma"]])
    # print(result[["beta", "R", "sigma"]])

    plt.clf()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    # def normalize(x):
    #     return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

    # result["DNML_codelength"] = normalize(
    #     result["DNML_codelength"])

    ax.plot(result["model_n_dims"], result[
            "DNML_" + dataset_name], label="L_DNML(y, z)", linestyle="solid", color="black")
    ax.plot(result["model_n_dims"], result[
            "-log p_" + dataset_name + "(y|z)"], label="L_NML(y|z)", linestyle="dotted", color="black")
    ax_2 = ax.twinx()
    ax_2.plot(result["model_n_dims"], result[
        "-log p_" + dataset_name + "(z)"], label="L_NML(z)", linestyle="dashdot", color="black")
    plt.xscale('log')
    # plt.yscale('log')

    # plt.xticks(result["model_n_dims"], fontsize=20)
    plt.xticks([2, 4, 8, 16, 32, 64], fontsize=20)

    ax.tick_params(labelsize=20)
    ax_2.tick_params(labelsize=20)

    plt.yticks(fontsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.legend(bbox_to_anchor=(1, 1), loc='upper right',
    #            borderaxespad=0, fontsize=15)
    # ax_2.legend(bbox_to_anchor=(1, 1), loc='upper right',
    #            borderaxespad=0, fontsize=15)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_2.get_legend_handles_labels()
    ax_2.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1, 1), loc='upper right',
                borderaxespad=0, fontsize=15)

    ax.set_xlabel("Dimensionality", fontsize=20)
    ax.set_ylabel("Code Length", fontsize=20)
    plt.tight_layout()

    plt.savefig("example_" + dataset_name + ".png")


# def calc_average_conciseness(result, eps, dataset_name):
#     D_max = 64

#     D_DNML_HGG = result["model_n_dims"].values[
#         np.argmin(result["DNML_HGG"].values)]
#     D_AIC_HGG = result["model_n_dims"].values[
#         np.argmin(result["AIC_HGG"].values)]
#     D_BIC_HGG = result["model_n_dims"].values[
#         np.argmin(result["BIC_HGG"].values)]
#     D_DNML_WND = result["model_n_dims"].values[
#         np.argmin(result["DNML_WND"].values)]
#     D_AIC_WND = result["model_n_dims"].values[
#         np.argmin(result["AIC_WND"].values)]
#     D_BIC_WND = result["model_n_dims"].values[
#         np.argmin(result["BIC_WND"].values)]
#     D_AIC_naive = result["model_n_dims"].values[
#         np.argmin(result["AIC_naive"].values)]
#     D_BIC_naive = result["model_n_dims"].values[
#         np.argmin(result["BIC_naive"].values)]
#     D_MinGE = result["model_n_dims"].values[
#         np.argmin(result["MinGE"].values)]

#     model_n_dims = np.array(result["model_n_dims"])
#     AUC_HGG = np.array(result["AUC_HGG"])
#     AUC_WND = np.array(result["AUC_WND"])
#     AUC_naive = np.array(result["AUC_naive"])

#     # AUC_HGG = np.array(result["AUC_HGG"])/np.max(result["AUC_HGG"])
#     # AUC_WND = np.array(result["AUC_WND"])/np.max(result["AUC_WND"])
#     # AUC_naive = np.array(result["AUC_naive"])/np.max(result["AUC_naive"])

#     # print(AUC_HGG)
#     # print(AUC_WND)
#     # print(AUC_naive)

#     def conceseness_with_fixed_eps(D_hat, D_eps_list):
#         if len(D_eps_list) == 0:
#             return 0
#         D_max = max(D_eps_list)
#         D_min = min(D_eps_list)
#         if D_hat in D_eps_list:
#             if D_max == D_min:
#                 return 0
#             else:
#                 return 1 - (np.log2(D_hat) - np.log2(D_min)) / (np.log2(D_max) - np.log2(D_min))
#                 # return 1 - (D_hat - D_min) / (D_max - D_min)

#         else:
#             return 0

#     def average_conciseness(AUCs, AUC_max, D_hat, eps_range, DIV):
#         criterion_list = []

#         # AUC_max = max(AUCs)
#         # AUC_min = min(AUCs)
#         eps_list = np.arange(DIV) * eps_range / DIV

#         # AUCs = AUCs

#         for eps in eps_list:
#             D_eps_list = model_n_dims[np.where(AUCs >= AUC_max - eps)[0]]
#             # D_eps_list = model_n_dims[np.where(AUCs >= 1 - eps)[0]]
#             # print(D_eps_list)
#             # D_min = min(D_eps_list)

#             criterion_list.append(
#                 conceseness_with_fixed_eps(D_hat, D_eps_list))

#         criterion_list = np.array(criterion_list)

#         return criterion_list

#     DIV = 1000
#     # eps_range = 0.02
#     AUC_max = max(np.max(result["AUC_HGG"]), np.max(
#         result["AUC_WND"]), np.max(result["AUC_naive"]))

#     criterion_DNML_HGG_list = average_conciseness(
#         AUC_HGG, AUC_max, D_DNML_HGG, eps, DIV)
#     criterion_AIC_HGG_list = average_conciseness(
#         AUC_HGG, AUC_max, D_AIC_HGG, eps, DIV)
#     criterion_BIC_HGG_list = average_conciseness(
#         AUC_HGG, AUC_max, D_BIC_HGG, eps, DIV)
#     criterion_DNML_WND_list = average_conciseness(
#         AUC_WND, AUC_max, D_DNML_WND, eps, DIV)
#     criterion_AIC_WND_list = average_conciseness(
#         AUC_WND, AUC_max, D_AIC_WND, eps, DIV)
#     criterion_BIC_WND_list = average_conciseness(
#         AUC_WND, AUC_max, D_BIC_WND, eps, DIV)
#     criterion_AIC_naive_list = average_conciseness(
#         AUC_naive, AUC_max, D_AIC_naive, eps, DIV)
#     criterion_BIC_naive_list = average_conciseness(
#         AUC_naive, AUC_max, D_BIC_naive, eps, DIV)
#     criterion_MinGE_list = average_conciseness(
#         AUC_naive, AUC_max, D_MinGE, eps, DIV)

#     print("Average conciseness")
#     print("DNML_HGG:", np.average(criterion_DNML_HGG_list))
#     # print("AIC_HGG:", np.average(criterion_AIC_HGG_list))
#     # print("BIC_HGG:", np.average(criterion_BIC_HGG_list))
#     print("DNML_WND:", np.average(criterion_DNML_WND_list))
#     # print("AIC_WND:", np.average(criterion_AIC_WND_list))
#     # print("BIC_WND:", np.average(criterion_BIC_WND_list))
#     print("AIC_naive:", np.average(criterion_AIC_naive_list))
#     print("BIC_naive:", np.average(criterion_BIC_naive_list))
#     print("MinGE:", np.average(criterion_MinGE_list))

#     ret = pd.DataFrame()

#     ret["dataset_name"] = [dataset_name]
#     ret["eps"] = [eps]
#     ret["DNML_HGG"] = np.average(criterion_DNML_HGG_list)
#     # ret["AIC_HGG"] = np.average(criterion_AIC_HGG_list)
#     # ret["BIC_HGG"] = np.average(criterion_BIC_HGG_list)
#     ret["DNML_WND"] = np.average(criterion_DNML_WND_list)
#     # ret["AIC_WND"] = np.average(criterion_AIC_WND_list)
#     # ret["BIC_WND"] = np.average(criterion_BIC_WND_list)
#     ret["AIC_naive"] = np.average(criterion_AIC_naive_list)
#     ret["BIC_naive"] = np.average(criterion_BIC_naive_list)
#     ret["MinGE:"] = np.average(criterion_MinGE_list)

#     return ret


def calc_average_conciseness(result, eps, dataset_name):
    D_max = 64

    D_DNML_lorentz_latent = result["model_n_dims"].values[
        np.argmin(result["DNML_lorentz_latent"].values)]
    D_AIC_lorentz_latent = result["model_n_dims"].values[
        np.argmin(result["AIC_lorentz_latent"].values)]
    D_BIC_lorentz_latent = result["model_n_dims"].values[
        np.argmin(result["BIC_lorentz_latent"].values)]
    D_AIC_lorentz_naive = result["model_n_dims"].values[
        np.argmin(result["AIC_lorentz_naive"].values)]
    D_BIC_lorentz_naive = result["model_n_dims"].values[
        np.argmin(result["BIC_lorentz_naive"].values)]
    D_DNML_euclidean_latent = result["model_n_dims"].values[
        np.argmin(result["DNML_euclidean_latent"].values)]
    D_AIC_euclidean_latent = result["model_n_dims"].values[
        np.argmin(result["AIC_euclidean_latent"].values)]
    D_BIC_euclidean_latent = result["model_n_dims"].values[
        np.argmin(result["BIC_euclidean_latent"].values)]
    D_AIC_euclidean_naive = result["model_n_dims"].values[
        np.argmin(result["AIC_euclidean_naive"].values)]
    D_BIC_euclidean_naive = result["model_n_dims"].values[
        np.argmin(result["BIC_euclidean_naive"].values)]
    D_DNML_spherical_latent = result["model_n_dims"].values[
        np.argmin(result["DNML_spherical_latent"].values)]
    D_AIC_spherical_latent = result["model_n_dims"].values[
        np.argmin(result["AIC_spherical_latent"].values)]
    D_BIC_spherical_latent = result["model_n_dims"].values[
        np.argmin(result["BIC_spherical_latent"].values)]
    D_AIC_spherical_naive = result["model_n_dims"].values[
        np.argmin(result["AIC_spherical_naive"].values)]
    D_BIC_spherical_naive = result["model_n_dims"].values[
        np.argmin(result["BIC_spherical_naive"].values)]

    model_n_dims = np.array(result["model_n_dims"])
    AUC_lorentz_latent = np.array(result["AUC_lorentz_latent"])
    AUC_euclidean_latent = np.array(result["AUC_euclidean_latent"])
    AUC_spherical_latent = np.array(result["AUC_spherical_latent"])
    AUC_lorentz_naive = np.array(result["AUC_lorentz_naive"])
    AUC_euclidean_naive = np.array(result["AUC_euclidean_naive"])
    AUC_spherical_naive = np.array(result["AUC_spherical_naive"])

    # print(AUC_HGG)
    # print(AUC_WND)
    # print(AUC_naive)

    def conceseness_with_fixed_eps(D_hat, D_eps_list):
        if len(D_eps_list) == 0:
            return 0
        D_max = max(D_eps_list)
        D_min = min(D_eps_list)
        if D_hat in D_eps_list:
            if D_max == D_min:
                return 0
            else:
                return 1 - (np.log2(D_hat) - np.log2(D_min)) / (np.log2(D_max) - np.log2(D_min))
                # return 1 - (D_hat - D_min) / (D_max - D_min)

        else:
            return 0

    def average_conciseness(AUCs, AUC_max, D_hat, eps_range, DIV):
        criterion_list = []

        # AUC_max = max(AUCs)
        # AUC_min = min(AUCs)
        eps_list = np.arange(DIV) * eps_range / DIV

        # AUCs = AUCs

        for eps in eps_list:
            D_eps_list = model_n_dims[np.where(AUCs >= AUC_max - eps)[0]]
            # D_eps_list = model_n_dims[np.where(AUCs >= 1 - eps)[0]]
            # print(D_eps_list)
            # D_min = min(D_eps_list)

            criterion_list.append(
                conceseness_with_fixed_eps(D_hat, D_eps_list))

        criterion_list = np.array(criterion_list)

        return criterion_list

    DIV = 1000
    # eps_range = 0.02
    # AUC_lorentz_max = max(np.max(result["AUC_lorentz_latent"]), np.max(result["AUC_lorentz_naive"]))
    # AUC_euclidean_max = max(np.max(result["AUC_euclidean_latent"]), np.max(result["AUC_euclidean_naive"]))
    # AUC_spherical_max = max(np.max(result["AUC_spherical_latent"]), np.max(result["AUC_spherical_naive"]))
    AUC_lorentz_max = max(np.max(result["AUC_lorentz_latent"]), np.max(result["AUC_lorentz_naive"]), np.max(result["AUC_euclidean_latent"]), np.max(
        result["AUC_euclidean_naive"]), np.max(result["AUC_spherical_latent"]), np.max(result["AUC_spherical_naive"]))
    AUC_euclidean_max = max(np.max(result["AUC_lorentz_latent"]), np.max(result["AUC_lorentz_naive"]), np.max(result["AUC_euclidean_latent"]), np.max(
        result["AUC_euclidean_naive"]), np.max(result["AUC_spherical_latent"]), np.max(result["AUC_spherical_naive"]))
    AUC_spherical_max = max(np.max(result["AUC_lorentz_latent"]), np.max(result["AUC_lorentz_naive"]), np.max(result["AUC_euclidean_latent"]), np.max(
        result["AUC_euclidean_naive"]), np.max(result["AUC_spherical_latent"]), np.max(result["AUC_spherical_naive"]))

    criterion_DNML_lorentz_latent_list = average_conciseness(
        AUC_lorentz_latent, AUC_lorentz_max, D_DNML_lorentz_latent, eps, DIV)
    criterion_DNML_euclidean_latent_list = average_conciseness(
        AUC_euclidean_latent, AUC_euclidean_max, D_DNML_euclidean_latent, eps, DIV)
    criterion_DNML_spherical_latent_list = average_conciseness(
        AUC_spherical_latent, AUC_spherical_max, D_DNML_spherical_latent, eps, DIV)
    criterion_AIC_lorentz_latent_list = average_conciseness(
        AUC_lorentz_latent, AUC_lorentz_max, D_AIC_lorentz_latent, eps, DIV)
    criterion_AIC_euclidean_latent_list = average_conciseness(
        AUC_euclidean_latent, AUC_euclidean_max, D_AIC_euclidean_latent, eps, DIV)
    criterion_AIC_spherical_latent_list = average_conciseness(
        AUC_spherical_latent, AUC_spherical_max, D_AIC_spherical_latent, eps, DIV)
    criterion_BIC_lorentz_latent_list = average_conciseness(
        AUC_lorentz_latent, AUC_lorentz_max, D_BIC_lorentz_latent, eps, DIV)
    criterion_BIC_euclidean_latent_list = average_conciseness(
        AUC_euclidean_latent, AUC_euclidean_max, D_BIC_euclidean_latent, eps, DIV)
    criterion_BIC_spherical_latent_list = average_conciseness(
        AUC_spherical_latent, AUC_spherical_max, D_BIC_spherical_latent, eps, DIV)
    criterion_AIC_lorentz_naive_list = average_conciseness(
        AUC_lorentz_naive, AUC_lorentz_max, D_AIC_lorentz_naive, eps, DIV)
    criterion_AIC_euclidean_naive_list = average_conciseness(
        AUC_euclidean_naive, AUC_euclidean_max, D_AIC_euclidean_naive, eps, DIV)
    criterion_AIC_spherical_naive_list = average_conciseness(
        AUC_spherical_naive, AUC_spherical_max, D_AIC_spherical_naive, eps, DIV)
    criterion_BIC_lorentz_naive_list = average_conciseness(
        AUC_lorentz_naive, AUC_lorentz_max, D_BIC_lorentz_naive, eps, DIV)
    criterion_BIC_euclidean_naive_list = average_conciseness(
        AUC_euclidean_naive, AUC_euclidean_max, D_BIC_euclidean_naive, eps, DIV)
    criterion_BIC_spherical_naive_list = average_conciseness(
        AUC_spherical_naive, AUC_spherical_max, D_BIC_spherical_naive, eps, DIV)

    print("Average conciseness")
    print("DNML_lorentz_latent:", np.average(
        criterion_DNML_lorentz_latent_list))
    print("DNML_euclidean_latent:", np.average(
        criterion_DNML_euclidean_latent_list))
    print("DNML_spherical_latent:", np.average(
        criterion_DNML_spherical_latent_list))
    # print("AIC_lorentz_latent:", np.average(criterion_AIC_lorentz_latent_list))
    # print("AIC_euclidean_latent:", np.average(criterion_AIC_euclidean_latent_list))
    # print("AIC_spherical_latent:", np.average(criterion_AIC_spherical_latent_list))
    # print("BIC_lorentz_latent:", np.average(criterion_BIC_lorentz_latent_list))
    # print("BIC_euclidean_latent:", np.average(criterion_BIC_euclidean_latent_list))
    # print("BIC_spherical_latent:", np.average(criterion_BIC_spherical_latent_list))
    print("AIC_lorentz_naive:", np.average(criterion_AIC_lorentz_naive_list))
    print("AIC_euclidean_naive:", np.average(
        criterion_AIC_euclidean_naive_list))
    print("AIC_spherical_naive:", np.average(
        criterion_AIC_spherical_naive_list))
    print("BIC_lorentz_naive:", np.average(criterion_BIC_lorentz_naive_list))
    print("BIC_euclidean_naive:", np.average(
        criterion_BIC_euclidean_naive_list))
    print("BIC_spherical_naive:", np.average(
        criterion_BIC_spherical_naive_list))

    ret = pd.DataFrame()

    ret["dataset_name"] = [dataset_name]
    ret["eps"] = [eps]
    ret["DNML_lorentz_latent"] = np.average(criterion_DNML_lorentz_latent_list)
    ret["DNML_euclidean_latent"] = np.average(
        criterion_DNML_euclidean_latent_list)
    ret["DNML_spherical_latent"] = np.average(
        criterion_DNML_spherical_latent_list)
    ret["AIC_lorentz_latent"] = np.average(criterion_AIC_lorentz_latent_list)
    ret["AIC_euclidean_latent"] = np.average(
        criterion_AIC_euclidean_latent_list)
    ret["AIC_spherical_latent"] = np.average(
        criterion_AIC_spherical_latent_list)
    ret["BIC_lorentz_latent"] = np.average(criterion_BIC_lorentz_latent_list)
    ret["BIC_euclidean_latent"] = np.average(
        criterion_BIC_euclidean_latent_list)
    ret["BIC_spherical_latent"] = np.average(
        criterion_BIC_spherical_latent_list)
    ret["AIC_lorentz_naive"] = np.average(criterion_AIC_lorentz_naive_list)
    ret["AIC_euclidean_naive"] = np.average(criterion_AIC_euclidean_naive_list)
    ret["AIC_spherical_naive"] = np.average(criterion_AIC_spherical_naive_list)
    ret["BIC_lorentz_naive"] = np.average(criterion_BIC_lorentz_naive_list)
    ret["BIC_euclidean_naive"] = np.average(criterion_BIC_euclidean_naive_list)
    ret["BIC_spherical_naive"] = np.average(criterion_BIC_spherical_naive_list)

    # # ret["AIC_HGG"] = np.average(criterion_AIC_HGG_list)
    # # ret["BIC_HGG"] = np.average(criterion_BIC_HGG_list)
    # ret["DNML_WND"] = np.average(criterion_DNML_WND_list)
    # # ret["AIC_WND"] = np.average(criterion_AIC_WND_list)
    # # ret["BIC_WND"] = np.average(criterion_BIC_WND_list)
    # ret["AIC_naive"] = np.average(criterion_AIC_naive_list)
    # ret["BIC_naive"] = np.average(criterion_BIC_naive_list)

    return ret


def realworld():
    dataset_name_list = [
        "ca-AstroPh",
        # "ca-CondMat",
        # "ca-GrQc",
        "ca-HepPh",
        # "cora",
        # "pubmed",
        "airport",
        # "bio-yeast-protein-inter",
        "mammal",
        "solid",
        # "worker",
        # "adult",
        # "instrument",
        # "leader",
        # "implement",
        # "inf-euroroad",
        # "inf-power"
    ]
    # dataset_name_list = [
    #     # "ca-AstroPh",
    #     # "ca-CondMat",
    #     "ca-GrQc",
    #     # "ca-HepPh",
    #     "cora",
    #     # "pubmed",
    #     "airport",
    #     "bio-yeast-protein-inter"
    # ]
    n_dim_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64]
    # n_dim_list = [2, 3, 4, 5, 6, 7, 8, 9]

    result_conciseness = pd.DataFrame({})

    for dataset_name in dataset_name_list:
        print("-----------------------", dataset_name, "-----------------------")

        result = pd.DataFrame()

        for n_dim in n_dim_list:
            row = pd.read_csv(RESULTS + "/" + dataset_name +
                              "/result_" + str(n_dim) + ".csv")
            result = result.append(row)

        # result_MinGE = pd.read_csv(
        #     RESULTS + "/" + dataset_name + "/result_MinGE.csv")

        # result = pd.merge(result, result_MinGE, on="model_n_dims", how="left")

        D_DNML_lorentz_latent = result["model_n_dims"].values[
            np.argmin(result["DNML_lorentz_latent"].values)]
        C_DNML_lorentz_latent = np.min(result["DNML_lorentz_latent"].values)
        D_AIC_lorentz_latent = result["model_n_dims"].values[
            np.argmin(result["AIC_lorentz_latent"].values)]
        C_AIC_lorentz_latent = np.min(result["AIC_lorentz_latent"].values)
        D_BIC_lorentz_latent = result["model_n_dims"].values[
            np.argmin(result["BIC_lorentz_latent"].values)]
        C_BIC_lorentz_latent = np.min(result["BIC_lorentz_latent"].values)
        D_AIC_lorentz_naive = result["model_n_dims"].values[
            np.argmin(result["AIC_lorentz_naive"].values)]
        C_AIC_lorentz_naive = np.min(result["AIC_lorentz_naive"].values)
        D_BIC_lorentz_naive = result["model_n_dims"].values[
            np.argmin(result["BIC_lorentz_naive"].values)]
        C_BIC_lorentz_naive = np.min(result["BIC_lorentz_naive"].values)
        D_DNML_euclidean_latent = result["model_n_dims"].values[
            np.argmin(result["DNML_euclidean_latent"].values)]
        C_DNML_euclidean_latent = np.min(
            result["DNML_euclidean_latent"].values)
        D_AIC_euclidean_latent = result["model_n_dims"].values[
            np.argmin(result["AIC_euclidean_latent"].values)]
        C_AIC_euclidean_latent = np.min(result["AIC_euclidean_latent"].values)
        D_BIC_euclidean_latent = result["model_n_dims"].values[
            np.argmin(result["BIC_euclidean_latent"].values)]
        C_BIC_euclidean_latent = np.min(result["BIC_euclidean_latent"].values)
        D_AIC_euclidean_naive = result["model_n_dims"].values[
            np.argmin(result["AIC_euclidean_naive"].values)]
        C_AIC_euclidean_naive = np.min(result["AIC_euclidean_naive"].values)
        D_BIC_euclidean_naive = result["model_n_dims"].values[
            np.argmin(result["BIC_euclidean_naive"].values)]
        C_BIC_euclidean_naive = np.min(result["BIC_euclidean_naive"].values)
        D_DNML_spherical_latent = result["model_n_dims"].values[
            np.argmin(result["DNML_spherical_latent"].values)]
        C_DNML_spherical_latent = np.min(
            result["DNML_spherical_latent"].values)
        D_AIC_spherical_latent = result["model_n_dims"].values[
            np.argmin(result["AIC_spherical_latent"].values)]
        C_AIC_spherical_latent = np.min(result["AIC_spherical_latent"].values)
        D_BIC_spherical_latent = result["model_n_dims"].values[
            np.argmin(result["BIC_spherical_latent"].values)]
        C_BIC_spherical_latent = np.min(result["BIC_spherical_latent"].values)
        D_AIC_spherical_naive = result["model_n_dims"].values[
            np.argmin(result["AIC_spherical_naive"].values)]
        C_AIC_spherical_naive = np.min(result["AIC_spherical_naive"].values)
        D_BIC_spherical_naive = result["model_n_dims"].values[
            np.argmin(result["BIC_spherical_naive"].values)]
        C_BIC_spherical_naive = np.min(result["BIC_spherical_naive"].values)

        model_lorentz_latent = torch.load(
            RESULTS + "/" + dataset_name + "/result_" + str(int(D_DNML_lorentz_latent)) + "_lorentz_latent.pth")
        K_DNML_lorentz_latent = 1 / model_lorentz_latent.kappa.detach().cpu().item()
        model_spherical_latent = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
            int(D_DNML_spherical_latent)) + "_spherical_latent.pth")
        K_DNML_spherical_latent = 1 / model_spherical_latent.kappa.detach().cpu().item()

        model_lorentz_latent = torch.load(
            RESULTS + "/" + dataset_name + "/result_" + str(int(D_AIC_lorentz_latent)) + "_lorentz_latent.pth")
        K_AIC_lorentz_latent = 1 / model_lorentz_latent.kappa.detach().cpu().item()
        model_spherical_latent = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
            int(D_DNML_spherical_latent)) + "_spherical_latent.pth")
        K_AIC_spherical_latent = 1 / model_spherical_latent.kappa.detach().cpu().item()

        model_lorentz_latent = torch.load(
            RESULTS + "/" + dataset_name + "/result_" + str(int(D_BIC_lorentz_latent)) + "_lorentz_latent.pth")
        K_BIC_lorentz_latent = 1 / model_lorentz_latent.kappa.detach().cpu().item()
        model_spherical_latent = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
            int(D_BIC_spherical_latent)) + "_spherical_latent.pth")
        K_BIC_spherical_latent = 1 / model_spherical_latent.kappa.detach().cpu().item()

        model_lorentz_naive = torch.load(
            RESULTS + "/" + dataset_name + "/result_" + str(int(D_AIC_lorentz_naive)) + "_lorentz_naive.pth")
        K_AIC_lorentz_naive = 1 / model_lorentz_naive.kappa.detach().cpu().item()
        model_spherical_naive = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
            int(D_AIC_spherical_naive)) + "_spherical_naive.pth")
        K_AIC_spherical_naive = 1 / model_spherical_naive.kappa.detach().cpu().item()

        model_lorentz_naive = torch.load(
            RESULTS + "/" + dataset_name + "/result_" + str(int(D_BIC_lorentz_naive)) + "_lorentz_naive.pth")
        K_BIC_lorentz_naive = 1 / model_lorentz_naive.kappa.detach().cpu().item()
        model_spherical_naive = torch.load(RESULTS + "/" + dataset_name + "/result_" + str(
            int(D_BIC_spherical_naive)) + "_spherical_naive.pth")
        K_BIC_spherical_naive = 1 / model_spherical_naive.kappa.detach().cpu().item()

        # print("K:", 1/model_lorentz_latent.kappa.detach().cpu().item())

        print("Selected Dimensionality of ", dataset_name)
        print("DNML_lorentz_latent:", D_DNML_lorentz_latent, K_DNML_lorentz_latent,
              ", ", C_DNML_lorentz_latent,  ",", result[result["model_n_dims"] == D_DNML_lorentz_latent]["AUC_lorentz_latent"].values[0])
        print("DNML_euclidean_latent:", D_DNML_euclidean_latent,
              ", ", C_DNML_euclidean_latent,  ",", result[result["model_n_dims"] == D_DNML_euclidean_latent]["AUC_euclidean_latent"].values[0])
        print("DNML_spherical_latent:", D_DNML_spherical_latent, K_DNML_spherical_latent,
              ", ", C_DNML_spherical_latent,  ",", result[result["model_n_dims"] == D_DNML_spherical_latent]["AUC_spherical_latent"].values[0])
        print("AIC_lorentz_naive:", D_AIC_lorentz_naive, K_AIC_lorentz_naive, ", ", C_AIC_lorentz_naive, ",", result[
              result["model_n_dims"] == D_AIC_lorentz_naive]["AUC_lorentz_naive"].values[0])
        print("AIC_euclidean:_naive", D_AIC_euclidean_naive, ", ", C_AIC_euclidean_naive,  ",", result[
              result["model_n_dims"] == D_AIC_euclidean_naive]["AUC_euclidean_naive"].values[0])
        print("AIC_spherical_naive:", D_AIC_spherical_naive, K_AIC_spherical_naive, ", ", C_AIC_spherical_naive,  ",", result[
              result["model_n_dims"] == D_AIC_spherical_naive]["AUC_spherical_naive"].values[0])
        print("BIC_lorentz_naive:", D_BIC_lorentz_naive, K_BIC_lorentz_naive, ", ", C_BIC_lorentz_naive,  ",", result[
              result["model_n_dims"] == D_BIC_lorentz_naive]["AUC_lorentz_naive"].values[0])
        print("BIC_euclidean:_naive", D_BIC_euclidean_naive, ", ", C_BIC_euclidean_naive,  ",", result[
              result["model_n_dims"] == D_BIC_euclidean_naive]["AUC_euclidean_naive"].values[0])
        print("BIC_spherical_naive:", D_BIC_spherical_naive, K_BIC_spherical_naive, ", ", C_BIC_spherical_naive,  ",", result[
              result["model_n_dims"] == D_BIC_spherical_naive]["AUC_spherical_naive"].values[0])
        # print("AIC_lorentz_latent:", D_AIC_lorentz_latent, ", ", C_AIC_lorentz_latent)
        # print("AIC_euclidean:_latent", D_AIC_euclidean_latent, ", ", C_AIC_euclidean_latent)
        # print("AIC_spherical_latent:", D_AIC_spherical_latent, ", ", C_AIC_spherical_latent)
        # print("BIC_lorentz_latent:", D_BIC_lorentz_latent, ", ", C_BIC_lorentz_latent)
        # print("BIC_euclidean:_latent", D_BIC_euclidean_latent, ", ", C_BIC_euclidean_latent)
        # print("BIC_spherical_latent:", D_BIC_spherical_latent, ", ", C_BIC_spherical_latent)

        # print(result[["model_n_dims", "AUC_lorentz_latent"]])
        # print(result[["model_n_dims", "AUC_euclidean_latent"]])
        # print(result[["model_n_dims", "AUC_spherical_latent"]])
        # print(result[["model_n_dims", "AUC_lorentz_naive"]])
        # print(result[["model_n_dims", "AUC_euclidean_naive"]])
        # print(result[["model_n_dims", "AUC_spherical_naive"]])

        # cor_DNML_HGG, _ = stats.spearmanr(
        #     result["AUC_HGG"], -result["DNML_HGG"].values)
        # cor_AIC_HGG, _ = stats.spearmanr(
        #     result["AUC_HGG"], -result["AIC_HGG"].values)
        # cor_BIC_HGG, _ = stats.spearmanr(
        #     result["AUC_HGG"], -result["BIC_HGG"].values)
        # cor_DNML_WND, _ = stats.spearmanr(
        #     result["AUC_WND"], -result["DNML_WND"].values)
        # cor_AIC_WND, _ = stats.spearmanr(
        #     result["AUC_WND"], -result["AIC_WND"].values)
        # cor_BIC_WND, _ = stats.spearmanr(
        #     result["AUC_WND"], -result["BIC_WND"].values)
        # cor_AIC, _ = stats.spearmanr(
        #     result["AUC_naive"], -result["AIC_naive"].values)
        # cor_BIC, _ = stats.spearmanr(
        #     result["AUC_naive"], -result["BIC_naive"].values)
        # cor_MinGE, _ = stats.spearmanr(
        #     result["AUC_naive"], -result["MinGE"].values)

        # print("cor_DNML_HGG:", cor_DNML_HGG)
        # print("cor_AIC_HGG:", cor_AIC_HGG)
        # print("cor_BIC_HGG:", cor_BIC_HGG)
        # print("cor_DNML_WND:", cor_DNML_WND)
        # print("cor_AIC_WND:", cor_AIC_WND)
        # print("cor_BIC_WND:", cor_BIC_WND)
        # print("cor_AIC_naive:", cor_AIC)
        # print("cor_BIC_naive:", cor_BIC)
        # print("cor_MinGE:", cor_MinGE)

        # result_conciseness = pd.DataFrame()
        # conciseness
        eps = 0.05
        print("eps =", eps)
        ret = calc_average_conciseness(result, eps, dataset_name)
        result_conciseness = pd.concat([result_conciseness, ret])
        eps = 0.100
        print("eps =", eps)
        ret = calc_average_conciseness(result, eps, dataset_name)
        result_conciseness = pd.concat([result_conciseness, ret])

        # row = pd.DataFrame(ret.values(), index=ret.keys()).T
        # result_conciseness = pd.concat([result_conciseness, row])

        # # 各criterionの値
        # plt.clf()

        # fig = plt.figure(figsize=(8, 5))
        # ax = fig.add_subplot(111)

        # ax.plot(result["model_n_dims"], result["AUC_HGG"],
        #         label="-log p(y, z; β, γ, σ)", linestyle="solid", color="black")
        # ax.plot(result["model_n_dims"], result["AUC_WND"],
        #         label="-log p(y, z; β, γ, Σ)", linestyle=loosely_dotted, color="black")
        # ax.plot(result["model_n_dims"], result["AUC_naive"],
        #         label="-log p(y|z; β, γ)", linestyle="dashed", color="black")
        # plt.xscale('log')
        # plt.ylim(0.6, 1.00)
        # # plt.ylim(0.4, 0.80)
        # # plt.xticks(result["model_n_dims"], fontsize=8)

        # plt.xticks([2, 4, 8, 16, 32, 64], fontsize=20)

        # plt.yticks(fontsize=20)
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # plt.legend(bbox_to_anchor=(1, 0), loc='lower right',
        #            borderaxespad=0, fontsize=15)
        # ax.set_xlabel("Dimensionality", fontsize=20)
        # ax.set_ylabel("AUC", fontsize=20)
        # plt.tight_layout()

        # plt.savefig(RESULTS + "/" + dataset_name +
        #             "/result_AUC_" + dataset_name + ".png")

        # # # 各criterionの値
        # # plt.clf()

        # fig = plt.figure(figsize=(8, 5))
        # ax = fig.add_subplot(111)

        # def normalize(x):
        #     # x_ = np.log(x)
        # return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

        # result["DNML_HGG"] = normalize(result["DNML_HGG"])
        # result["DNML_WND"] = normalize(result["DNML_WND"])
        # result["AIC_naive"] = normalize(result["AIC_naive"])
        # result["BIC_naive"] = normalize(result["BIC_naive"])
        # result["MinGE"] = normalize(result["MinGE"])

        # ax.plot(result["model_n_dims"], result[
        #         "DNML_HGG"], label="DNML-PUD", linestyle="solid", color="black")
        # ax.plot(result["model_n_dims"], result[
        #         "DNML_WND"], label="DNML-WND", linestyle=loosely_dotted, color="black")
        # ax.plot(result["model_n_dims"], result["AIC_naive"],
        #         label="AIC", linestyle="dotted", color="black")
        # ax.plot(result["model_n_dims"], result["BIC_naive"],
        #         label="BIC", linestyle="dashed", color="black")
        # ax.plot(result["model_n_dims"], result[
        #         "MinGE"], label="MinGE", linestyle="dashdot", color="black")
        # plt.xscale('log')
        # # plt.xticks(result["model_n_dims"], fontsize=8)
        # plt.xticks([2, 4, 8, 16, 32, 64], fontsize=20)

        # plt.yticks(fontsize=20)
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
        #            borderaxespad=0, fontsize=15)
        # ax.set_xlabel("Dimensionality", fontsize=20)
        # ax.set_ylabel("Normalized Criterion", fontsize=20)
        # plt.tight_layout()

        # plt.savefig(RESULTS + "/" + dataset_name +
        #             "/result_" + dataset_name + ".png")

    # print(result_conciseness)
    # result_conciseness.to_csv(RESULTS + "/result_realworld.csv", index=False)
    # print(result_conciseness.groupby("dataset_name").mean().drop("eps", axis=1))
    # result_conciseness.groupby("dataset_name").mean().drop(
    #     "eps", axis=1).to_csv(RESULTS + "/result_realworld_dataset.csv")
    # print(result_conciseness.groupby("eps").mean())
    # result_conciseness.groupby("eps").mean().to_csv(
    #     RESULTS + "/result_realworld_eps.csv", index=False)


def realworld_stats():
    dataset_name_list = [
        "ca-AstroPh",
        "ca-CondMat",
        "ca-GrQc",
        "ca-HepPh",
        "cora",
        "pubmed",
        "airport",
        "bio-yeast-protein-inter"
    ]

    for dataset_name in dataset_name_list:
        data = np.load('dataset/' + dataset_name +
                       '/data.npy', allow_pickle=True).item()
        adj_mat = data["adj_mat"].toarray()
        n_nodes = adj_mat.shape[0]
        n_edges = np.sum(adj_mat) / 2

        print(dataset_name)
        print("# nodes:", n_nodes)
        print("# edges:", n_edges)


if __name__ == "__main__":

    # print("Plot Example Figure")
    # plot_figure("HGG", 5)
    # plot_figure("WND", 3)
    # print("Results of Artificial Datasets")
    artificial("hyperbolic")
    artificial("euclidean")
    artificial("spherical")

    # artificial("HGG")
    # artificial("WND")
    # print("Results of Scientific Collaboration Networks")
    # realworld_stats()
    realworld()
    # print("Results of WN dataset")
    # wn_dataset()
