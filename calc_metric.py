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
from experiment_wn import is_a_score
import torch
from multiprocessing import Pool
from functools import partial


RESULTS = "results"


def artificial(dataset):
    D_true_list = [8, 16]
    # n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
    n_nodes_list = [400, 800, 1600, 3200, 6400]
    n_graphs = 12
    T_gap = 2

    for D_true in D_true_list:

        for n_nodes in n_nodes_list:
            bene_DNML = []
            bene_AIC_latent = []
            bene_BIC_latent = []
            bene_AIC_naive = []
            bene_BIC_naive = []
            bene_MinGE = []
            estimate_DNML = []
            estimate_AIC_latent = []
            estimate_BIC_latent = []
            estimate_AIC_naive = []
            estimate_BIC_naive = []
            estimate_MinGE = []

            for n_graph in range(n_graphs):
                result = pd.read_csv(RESULTS + "/" + dataset + "/dim_" + str(D_true) +
                                     "/result_" + str(n_nodes) + "_" + str(n_graph) + ".csv")
                result = result.fillna(9999999999999)
                result_MinGE = pd.read_csv(
                    RESULTS + "/" + dataset + "/dim_" + str(D_true) + "/result_" + str(n_nodes) +
                    "_" + str(n_graph) + "_MinGE.csv")

                D_DNML = result["model_n_dims"].values[
                    np.argmin(result["DNML_" + dataset].values)]
                D_AIC_latent = result["model_n_dims"].values[
                    np.argmin(result["AIC_" + dataset].values)]
                D_BIC_latent = result["model_n_dims"].values[
                    np.argmin(result["BIC_" + dataset].values)]
                D_AIC_naive = result["model_n_dims"].values[
                    np.argmin(result["AIC_naive"].values)]
                D_BIC_naive = result["model_n_dims"].values[
                    np.argmin(result["BIC_naive"].values)]
                D_MinGE = result_MinGE["model_n_dims"].values[
                    np.argmin(result_MinGE["MinGE"].values)]

                estimate_DNML.append(D_DNML)
                estimate_AIC_latent.append(D_AIC_latent)
                estimate_BIC_latent.append(D_BIC_latent)
                estimate_AIC_naive.append(D_AIC_naive)
                estimate_BIC_naive.append(D_BIC_naive)
                estimate_MinGE.append(D_MinGE)

                # bene_DNML.append(
                #     label_ranking_average_precision_score([label], [-result["DNML_codelength"].values]))
                # bene_AIC.append(
                #     label_ranking_average_precision_score([label], [-result["AIC_naive"].values]))
                # bene_BIC.append(
                #     label_ranking_average_precision_score([label], [-result["BIC_naive"].values]))
                # bene_MinGE.append(
                # label_ranking_average_precision_score([label],
                # [-result_MinGE["MinGE"].values]))

                bene_DNML.append(
                    max(0, 1 - abs(np.log2(D_DNML) - np.log2(D_true)) / T_gap))
                bene_AIC_latent.append(
                    max(0, 1 - abs(np.log2(D_AIC_latent) - np.log2(D_true)) / T_gap))
                bene_BIC_latent.append(
                    max(0, 1 - abs(np.log2(D_BIC_latent) - np.log2(D_true)) / T_gap))
                bene_AIC_naive.append(
                    max(0, 1 - abs(np.log2(D_AIC_naive) - np.log2(D_true)) / T_gap))
                bene_BIC_naive.append(
                    max(0, 1 - abs(np.log2(D_BIC_naive) - np.log2(D_true)) / T_gap))
                bene_MinGE.append(
                    max(0, 1 - abs(np.log2(D_MinGE) - np.log2(D_true)) / T_gap))

                plt.clf()
                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)

                def normalize(x):
                    return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

                result["DNML_" + dataset] = normalize(
                    result["DNML_" + dataset])
                result["AIC_" + dataset] = normalize(result["AIC_" + dataset])
                result["BIC_" + dataset] = normalize(result["BIC_" + dataset])
                result["AIC_naive"] = normalize(result["AIC_naive"])
                result["BIC_naive"] = normalize(result["BIC_naive"])
                result["MinGE"] = normalize(result_MinGE["MinGE"])

                ax.plot(result["model_n_dims"], result[
                        "DNML_" + dataset], label="DNML-" + dataset, color="red")
                # ax.plot(result["model_n_dims"], result["AIC_"+dataset],
                #         label="AIC_"+dataset, color="blue")
                # ax.plot(result["model_n_dims"], result["BIC_"+dataset],
                #         label="BIC_"+dataset, color="green")
                ax.plot(result["model_n_dims"], result["AIC_naive"],
                        label="AIC_naive", color="blue")
                ax.plot(result["model_n_dims"], result["BIC_naive"],
                        label="BIC_naive", color="green")
                ax.plot(result["model_n_dims"], result[
                    "MinGE"], label="MinGE", color="orange")
                plt.xscale('log')

                plt.xticks(result["model_n_dims"], fontsize=20)
                plt.yticks(fontsize=20)
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
                           borderaxespad=0, fontsize=15)
                ax.set_xlabel("Dimensionality", fontsize=20)
                ax.set_ylabel("Normalized Criterion", fontsize=20)
                plt.tight_layout()
                os.makedirs(RESULTS + "/" + dataset + "_fig/", exist_ok=True)

                plt.savefig(RESULTS + "/" + dataset + "_fig/result_" +
                            str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) + ".png")

            bene_DNML = np.array(bene_DNML)
            bene_AIC_latent = np.array(bene_AIC_latent)
            bene_BIC_latent = np.array(bene_BIC_latent)
            bene_AIC_naive = np.array(bene_AIC_naive)
            bene_BIC_naive = np.array(bene_BIC_naive)
            bene_MinGE = np.array(bene_MinGE)

            estimate_DNML = np.array(estimate_DNML)
            estimate_AIC_latent = np.array(estimate_AIC_latent)
            estimate_BIC_latent = np.array(estimate_BIC_latent)
            estimate_AIC_naive = np.array(estimate_AIC_naive)
            estimate_BIC_naive = np.array(estimate_BIC_naive)
            estimate_MinGE = np.array(estimate_MinGE)

            print("n_nodes:", n_nodes)
            print("dimensionality:", D_true)
            print("DNML_" + dataset + ":",
                  np.mean(bene_DNML), "±", np.std(bene_DNML))
            print("AIC_" + dataset + ":", np.mean(bene_AIC_latent),
                  "±", np.std(bene_AIC_latent))
            print("BIC_" + dataset + ":", np.mean(bene_BIC_latent),
                  "±", np.std(bene_BIC_latent))
            print("AIC_naive:", np.mean(bene_AIC_naive),
                  "±", np.std(bene_AIC_naive))
            print("BIC_naive:", np.mean(bene_BIC_naive),
                  "±", np.std(bene_BIC_naive))
            print("MinGE:", np.mean(bene_MinGE), "±", np.std(bene_MinGE))

            print("DNML_" + dataset + ":", np.mean(estimate_DNML),
                  "±", np.std(estimate_DNML))
            print("AIC_" + dataset + ":", np.mean(estimate_AIC_latent),
                  "±", np.std(estimate_AIC_latent))
            print("BIC_" + dataset + ":", np.mean(estimate_BIC_latent),
                  "±", np.std(estimate_BIC_latent))
            print("AIC_naive:", np.mean(estimate_AIC_naive),
                  "±", np.std(estimate_AIC_naive))
            print("BIC_naive:", np.mean(estimate_BIC_naive),
                  "±", np.std(estimate_BIC_naive))
            print("MinGE:", np.mean(estimate_MinGE),
                  "±", np.std(estimate_MinGE))


def plot_figure():

    result = pd.read_csv(RESULTS + "/dim_16/result_6400_3.csv")
    result = result.fillna(9999999999999)
    # result = result.drop(result.index[[5]])
    D_DNML = result["model_n_dims"].values[
        np.argmin(result["DNML_codelength"].values)]

    plt.clf()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    # def normalize(x):
    #     return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

    # result["DNML_codelength"] = normalize(
    #     result["DNML_codelength"])

    ax.plot(result["model_n_dims"], result[
            "DNML_codelength"], label="L_DNML(y, z)", color="green")
    ax.plot(result["model_n_dims"], result[
            "basescore_y_given_z"], label="L_NML(y|z)", color="blue")
    ax_2 = ax.twinx()
    ax_2.plot(result["model_n_dims"], result[
        "basescore_z"], label="L_NML(z)", color="orange")
    plt.xscale('log')
    # plt.yscale('log')

    plt.xticks(result["model_n_dims"], fontsize=20)
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

    plt.savefig("example.png")


def calc_new_metrics(result):
    D_max = 64

    D_DNML_HGG = result["model_n_dims"].values[
        np.argmin(result["DNML_HGG"].values)]
    D_AIC_HGG = result["model_n_dims"].values[
        np.argmin(result["AIC_HGG"].values)]
    D_BIC_HGG = result["model_n_dims"].values[
        np.argmin(result["BIC_HGG"].values)]
    D_DNML_WND = result["model_n_dims"].values[
        np.argmin(result["DNML_WND"].values)]
    D_AIC_WND = result["model_n_dims"].values[
        np.argmin(result["AIC_WND"].values)]
    D_BIC_WND = result["model_n_dims"].values[
        np.argmin(result["BIC_WND"].values)]
    D_AIC_naive = result["model_n_dims"].values[
        np.argmin(result["AIC_naive"].values)]
    D_BIC_naive = result["model_n_dims"].values[
        np.argmin(result["BIC_naive"].values)]
    D_MinGE = result["model_n_dims"].values[
        np.argmin(result["MinGE"].values)]

    model_n_dims = np.array(result["model_n_dims"])
    AUC_HGG = np.array(result["AUC_HGG"])
    AUC_WND = np.array(result["AUC_WND"])
    AUC_naive = np.array(result["AUC_naive"])

    def calc_criterion(D_hat, D_eps_list):
        D_max = max(D_eps_list)
        D_min = min(D_eps_list)
        if D_hat in D_eps_list:
            if D_max == D_min:
                return 0
            else:
                return 1 - (np.log2(D_hat) - np.log2(D_min)) / (np.log2(D_max) - np.log2(D_min))
        else:
            return 0

    def calc_conciseness(AUCs, D_hat, eps_range, DIV):
        criterion_list = []

        AUC_max = max(AUCs)
        AUC_min = min(AUCs)
        eps_list = np.arange(DIV) * eps_range / DIV

        for eps in eps_list:
            D_eps_list = model_n_dims[np.where(AUCs >= AUC_max - eps)[0]]
            D_min = min(D_eps_list)

            criterion_list.append(calc_criterion(D_hat, D_eps_list))

        criterion_list = np.array(criterion_list)

        return criterion_list

    DIV = 1000
    eps_range = 0.1

    criterion_DNML_HGG_list = calc_conciseness(
        AUC_HGG, D_DNML_HGG, eps_range, DIV)
    criterion_AIC_HGG_list = calc_conciseness(
        AUC_HGG, D_AIC_HGG, eps_range, DIV)
    criterion_BIC_HGG_list = calc_conciseness(
        AUC_HGG, D_BIC_HGG, eps_range, DIV)
    criterion_DNML_WND_list = calc_conciseness(
        AUC_WND, D_DNML_WND, eps_range, DIV)
    criterion_AIC_WND_list = calc_conciseness(
        AUC_WND, D_AIC_WND, eps_range, DIV)
    criterion_BIC_WND_list = calc_conciseness(
        AUC_WND, D_BIC_WND, eps_range, DIV)
    criterion_AIC_naive_list = calc_conciseness(
        AUC_naive, D_AIC_naive, eps_range, DIV)
    criterion_BIC_naive_list = calc_conciseness(
        AUC_naive, D_BIC_naive, eps_range, DIV)
    criterion_MinGE_list = calc_conciseness(AUC_naive, D_MinGE, eps_range, DIV)

    print("Average conciseness")
    print("DNML_HGG:", np.average(criterion_DNML_HGG_list))
    print("AIC_HGG:", np.average(criterion_AIC_HGG_list))
    print("BIC_HGG:", np.average(criterion_BIC_HGG_list))
    print("DNML_WND:", np.average(criterion_DNML_WND_list))
    print("AIC_WND:", np.average(criterion_AIC_WND_list))
    print("BIC_WND:", np.average(criterion_BIC_WND_list))
    print("AIC_naive:", np.average(criterion_AIC_naive_list))
    print("BIC_naive:", np.average(criterion_BIC_naive_list))
    print("MinGE:", np.average(criterion_MinGE_list))

    ret = {}

    ret["DNML_HGG"] = np.average(criterion_DNML_HGG_list)
    ret["AIC_HGG"] = np.average(criterion_AIC_HGG_list)
    ret["BIC_HGG"] = np.average(criterion_BIC_HGG_list)
    ret["DNML_WND"] = np.average(criterion_DNML_WND_list)
    ret["AIC_WND"] = np.average(criterion_AIC_WND_list)
    ret["BIC_WND"] = np.average(criterion_BIC_WND_list)
    ret["AIC_naive"] = np.average(criterion_AIC_naive_list)
    ret["BIC_naive"] = np.average(criterion_BIC_naive_list)
    ret["MinGE:"] = np.average(criterion_MinGE_list)

    return ret


def realworld():
    dataset_name_list = ["ca-AstroPh", "ca-CondMat", "ca-GrQc", "ca-HepPh"]
    # dataset_name_list = ["ca-AstroPh", "ca-HepPh"]

    # dataset_name_list = ["ca-GrQc",  "ca-HepPh"]

    # n_dim_list = [2, 4, 8, 16, 32, 64]
    # n_dim_list = [2, 3, 4, 5, 6, 7, 8, 16, 32, 64]
    n_dim_list = [2, 3, 5, 6, 8, 16, 32, 64]

    result_conciseness = pd.DataFrame({})

    for dataset_name in dataset_name_list:
        print("-----------------------", dataset_name, "-----------------------")

        result = pd.DataFrame()

        for n_dim in n_dim_list:
            row = pd.read_csv(RESULTS + "/" + dataset_name +
                              "/result_" + str(n_dim) + ".csv")
            result = result.append(row)

        result_MinGE = pd.read_csv(
            RESULTS + "/" + dataset_name + "/result_MinGE.csv")

        result = pd.merge(result, result_MinGE, on="model_n_dims", how="left")

        D_DNML_HGG = result["model_n_dims"].values[
            np.argmin(result["DNML_HGG"].values)]
        D_AIC_HGG = result["model_n_dims"].values[
            np.argmin(result["AIC_HGG"].values)]
        D_BIC_HGG = result["model_n_dims"].values[
            np.argmin(result["BIC_HGG"].values)]
        D_DNML_WND = result["model_n_dims"].values[
            np.argmin(result["DNML_WND"].values)]
        D_AIC_WND = result["model_n_dims"].values[
            np.argmin(result["AIC_WND"].values)]
        D_BIC_WND = result["model_n_dims"].values[
            np.argmin(result["BIC_WND"].values)]
        D_AIC_naive = result["model_n_dims"].values[
            np.argmin(result["AIC_naive"].values)]
        D_BIC_naive = result["model_n_dims"].values[
            np.argmin(result["BIC_naive"].values)]
        D_MinGE = result["model_n_dims"].values[
            np.argmin(result["MinGE"].values)]

        print("Selected Dimensionality of ", dataset_name)
        print("DNML_HGG:", D_DNML_HGG)
        print("AIC_HGG:", D_AIC_HGG)
        print("BIC_HGG:", D_BIC_HGG)
        print("DNML_WND:", D_DNML_WND)
        print("AIC_WND:", D_AIC_WND)
        print("BIC_WND:", D_BIC_WND)
        print("AIC_naive:", D_AIC_naive)
        print("BIC_naive:", D_BIC_naive)
        print("MinGE:", D_MinGE)

        print(result[["model_n_dims", "AUC_HGG"]])
        print(result[["model_n_dims", "AUC_WND"]])
        print(result[["model_n_dims", "AUC_naive"]])

        cor_DNML_HGG, _ = stats.spearmanr(
            result["AUC_HGG"], -result["DNML_HGG"].values)
        cor_AIC_HGG, _ = stats.spearmanr(
            result["AUC_HGG"], -result["AIC_HGG"].values)
        cor_BIC_HGG, _ = stats.spearmanr(
            result["AUC_HGG"], -result["BIC_HGG"].values)
        cor_DNML_WND, _ = stats.spearmanr(
            result["AUC_WND"], -result["DNML_WND"].values)
        cor_AIC_WND, _ = stats.spearmanr(
            result["AUC_WND"], -result["AIC_WND"].values)
        cor_BIC_WND, _ = stats.spearmanr(
            result["AUC_WND"], -result["BIC_WND"].values)
        cor_AIC, _ = stats.spearmanr(
            result["AUC_naive"], -result["AIC_naive"].values)
        cor_BIC, _ = stats.spearmanr(
            result["AUC_naive"], -result["BIC_naive"].values)
        cor_MinGE, _ = stats.spearmanr(
            result["AUC_naive"], -result["MinGE"].values)

        print("cor_DNML_HGG:", cor_DNML_HGG)
        print("cor_AIC_HGG:", cor_AIC_HGG)
        print("cor_BIC_HGG:", cor_BIC_HGG)
        print("cor_DNML_WND:", cor_DNML_WND)
        print("cor_AIC_WND:", cor_AIC_WND)
        print("cor_BIC_WND:", cor_BIC_WND)
        print("cor_AIC_naive:", cor_AIC)
        print("cor_BIC_naive:", cor_BIC)
        print("cor_MinGE:", cor_MinGE)

        ret = calc_new_metrics(result)

        row = pd.DataFrame(ret.values(), index=ret.keys()).T
        result_conciseness = pd.concat([result_conciseness, row])

        # 各criterionの値
        plt.clf()

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.plot(result["model_n_dims"], result["AUC_HGG"],
                label="AUC of -log p_HGG(y, z)", color="red")
        ax.plot(result["model_n_dims"], result["AUC_WND"],
                label="AUC of -log p_WND(y, z)", color="orange")
        ax.plot(result["model_n_dims"], result["AUC_naive"],
                label="AUC of -log p(y|z)", color="blue")
        plt.xscale('log')
        plt.ylim(0.4, 1.00)
        plt.xticks(result["model_n_dims"], fontsize=20)
        plt.yticks(fontsize=20)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right',
                   borderaxespad=0, fontsize=15)
        ax.set_xlabel("Dimensionality", fontsize=20)
        ax.set_ylabel("AUC", fontsize=20)
        plt.tight_layout()

        plt.savefig(RESULTS + "/" + dataset_name +
                    "/result_AUC_" + dataset_name + ".png")

        # 各criterionの値
        plt.clf()

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        def normalize(x):
            # x_ = np.log(x)
            return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

        result["DNML_HGG"] = normalize(result["DNML_HGG"])
        result["DNML_WND"] = normalize(result["DNML_WND"])
        result["AIC_naive"] = normalize(result["AIC_naive"])
        result["BIC_naive"] = normalize(result["BIC_naive"])
        result["MinGE"] = normalize(result["MinGE"])

        ax.plot(result["model_n_dims"], result[
                "DNML_HGG"], label="DNML-HGG", color="red")
        ax.plot(result["model_n_dims"], result[
                "DNML_WND"], label="DNML-WND", color="black")
        ax.plot(result["model_n_dims"], result["AIC_naive"],
                label="AIC_naive", color="blue")
        ax.plot(result["model_n_dims"], result["BIC_naive"],
                label="BIC_naive", color="green")
        ax.plot(result["model_n_dims"], result[
                "MinGE"], label="MinGE", color="orange")
        plt.xscale('log')
        plt.xticks(result["model_n_dims"], fontsize=20)
        plt.yticks(fontsize=20)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
                   borderaxespad=0, fontsize=15)
        ax.set_xlabel("Dimensionality", fontsize=20)
        ax.set_ylabel("Normalized Criterion", fontsize=20)
        plt.tight_layout()

        plt.savefig(RESULTS + "/" + dataset_name +
                    "/result_" + dataset_name + ".png")

    print(result_conciseness.mean().T)


def calc_is_a_scores(model_n_dim, dataset_name, is_a):
    lorentz_table_hgg = torch.load(
        RESULTS + "/" + dataset_name + '/result_' + str(model_n_dim) + '_hgg.pth').get_lorentz_table()
    lorentz_table_wnd = torch.load(
        RESULTS + "/" + dataset_name + '/result_' + str(model_n_dim) + '_wnd.pth').get_lorentz_table()
    lorentz_table_naive = torch.load(
        RESULTS + "/" + dataset_name + '/result_' + str(model_n_dim) + '_naive.pth').get_lorentz_table()
    is_a_score_hgg = is_a_score(
        is_a, model_n_dim, lorentz_table_hgg, 100).item()
    is_a_score_wnd = is_a_score(
        is_a, model_n_dim, lorentz_table_wnd, 100).item()
    is_a_score_naive = is_a_score(
        is_a, model_n_dim, lorentz_table_naive, 100).item()

    return is_a_score_hgg, is_a_score_wnd, is_a_score_naive


def result_wn(model_n_dims, dataset_name):
    print(dataset_name)
    is_a_score_hgg_list = []
    is_a_score_wnd_list = []
    is_a_score_naive_list = []

    data = np.load("dataset/wn_dataset/" + dataset_name +
                   "_data.npy", allow_pickle=True).item()
    is_a = data["is_a"]

    calc_is_a_scores_ = partial(
        calc_is_a_scores, dataset_name=dataset_name, is_a=is_a)

    # multiprocessing
    p = Pool(12)

    results = p.map(calc_is_a_scores_, model_n_dims)
    results = np.array(results)

    is_a_score_hgg_list = results[:, 0]
    is_a_score_wnd_list = results[:, 1]
    is_a_score_naive_list = results[:, 2]

    # for model_n_dim in model_n_dims:
    #     lorentz_table_hgg = torch.load(
    #         RESULTS + "/" + dataset_name + '/result_' + str(model_n_dim) + '_hgg.pth').get_lorentz_table()
    #     lorentz_table_wnd = torch.load(
    #         RESULTS + "/" + dataset_name + '/result_' + str(model_n_dim) + '_wnd.pth').get_lorentz_table()
    #     lorentz_table_naive = torch.load(
    # RESULTS + "/" + dataset_name + '/result_' + str(model_n_dim) +
    # '_naive.pth').get_lorentz_table()

    #     print("HGG D=", model_n_dim)
    #     is_a_score_hgg_list.append(is_a_score(
    #         is_a, model_n_dim, lorentz_table_hgg, 100))
    #     print("WND D=", model_n_dim)
    #     is_a_score_wnd_list.append(is_a_score(
    #         is_a, model_n_dim, lorentz_table_wnd, 100))
    #     print("Naive D=", model_n_dim)
    #     is_a_score_naive_list.append(is_a_score(
    #         is_a, model_n_dim, lorentz_table_naive, 100))

    # is_a_score_hgg_list = np.array(is_a_score_hgg_list)
    # is_a_score_wnd_list = np.array(is_a_score_wnd_list)
    # is_a_score_naive_list = np.array(is_a_score_naive_list)

    result = pd.DataFrame()

    for n_dim in model_n_dims:
        row = pd.read_csv(RESULTS + "/" + dataset_name +
                          "/result_" + str(n_dim) + ".csv")
        result = result.append(row)

    # result = pd.read_csv(RESULTS + "/" + dataset_name + "/result.csv")

    result_MinGE = pd.read_csv(
        RESULTS + "/" + dataset_name + "/result_MinGE.csv")

    result = pd.merge(result, result_MinGE, on="model_n_dims", how="left")

    # result = result.iloc[1:, :]

    D_DNML_HGG = result["model_n_dims"].values[
        np.argmin(result["DNML_HGG"].values)]
    D_AIC_HGG = result["model_n_dims"].values[
        np.argmin(result["AIC_HGG"].values)]
    D_BIC_HGG = result["model_n_dims"].values[
        np.argmin(result["BIC_HGG"].values)]
    D_DNML_WND = result["model_n_dims"].values[
        np.argmin(result["DNML_WND"].values)]
    D_AIC_WND = result["model_n_dims"].values[
        np.argmin(result["AIC_WND"].values)]
    D_BIC_WND = result["model_n_dims"].values[
        np.argmin(result["BIC_WND"].values)]
    D_AIC_naive = result["model_n_dims"].values[
        np.argmin(result["AIC_naive"].values)]
    D_BIC_naive = result["model_n_dims"].values[
        np.argmin(result["BIC_naive"].values)]
    D_MinGE = result["model_n_dims"].values[
        np.argmin(result["MinGE"].values)]

    best_D_HGG = result["model_n_dims"].values[
        np.argmax(is_a_score_hgg_list)]
    best_D_WND = result["model_n_dims"].values[
        np.argmax(is_a_score_wnd_list)]
    best_D_naive = result["model_n_dims"].values[
        np.argmax(is_a_score_naive_list)]

    print("best HGG:", best_D_HGG)
    print("best WND:", best_D_WND)
    print("best naive:", best_D_naive)
    print("DNML-HGG:", D_DNML_HGG)
    print("AIC-HGG:", D_AIC_HGG)
    print("BIC-HGG:", D_BIC_HGG)
    print("DNML-WND:", D_DNML_WND)
    print("AIC-WND:", D_AIC_WND)
    print("BIC-WND:", D_BIC_WND)
    print("AIC_naive:", D_AIC_naive)
    print("BIC_naive:", D_BIC_naive)
    print("MinGE:", D_MinGE)

    T_gap = 2
    ret = {
        "bene_DNML_HGG": max(0, 1 - abs(np.log2(D_DNML_HGG) - np.log2(best_D_HGG)) / T_gap),
        "bene_AIC_HGG": max(0, 1 - abs(np.log2(D_AIC_HGG) - np.log2(best_D_HGG)) / T_gap),
        "bene_BIC_HGG": max(0, 1 - abs(np.log2(D_BIC_HGG) - np.log2(best_D_HGG)) / T_gap),
        "bene_DNML_WND": max(0, 1 - abs(np.log2(D_DNML_WND) - np.log2(best_D_WND)) / T_gap),
        "bene_AIC_WND": max(0, 1 - abs(np.log2(D_AIC_WND) - np.log2(best_D_WND)) / T_gap),
        "bene_BIC_WND": max(0, 1 - abs(np.log2(D_BIC_WND) - np.log2(best_D_WND)) / T_gap),
        "bene_AIC_naive": max(0, 1 - abs(np.log2(D_AIC_naive) - np.log2(best_D_naive)) / T_gap),
        "bene_BIC_naive": max(0, 1 - abs(np.log2(D_BIC_naive) - np.log2(best_D_naive)) / T_gap),
        "bene_MinGE": max(0, 1 - abs(np.log2(D_MinGE) - np.log2(best_D_naive)) / T_gap)
    }

    cor_DNML_HGG, _ = stats.spearmanr(
        is_a_score_hgg_list, -result["DNML_HGG"].values)
    cor_AIC_HGG, _ = stats.spearmanr(
        is_a_score_hgg_list, -result["AIC_HGG"].values)
    cor_BIC_HGG, _ = stats.spearmanr(
        is_a_score_hgg_list, -result["BIC_HGG"].values)
    cor_DNML_WND, _ = stats.spearmanr(
        is_a_score_wnd_list, -result["DNML_WND"].values)
    cor_AIC_WND, _ = stats.spearmanr(
        is_a_score_wnd_list, -result["AIC_WND"].values)
    cor_BIC_WND, _ = stats.spearmanr(
        is_a_score_wnd_list, -result["BIC_WND"].values)
    cor_AIC, _ = stats.spearmanr(
        is_a_score_naive_list, -result["AIC_naive"].values)
    cor_BIC, _ = stats.spearmanr(
        is_a_score_naive_list, -result["BIC_naive"].values)
    cor_MinGE, _ = stats.spearmanr(
        is_a_score_naive_list, -result["MinGE"].values)

    ret["cor_DNML_HGG"] = cor_DNML_HGG
    ret["cor_DNML_WND"] = cor_DNML_WND
    ret["cor_AIC_naive"] = cor_AIC
    ret["cor_BIC_naive"] = cor_BIC
    ret["cor_MinGE"] = cor_MinGE

    print("cor_DNML_HGG:", cor_DNML_HGG)
    print("cor_AIC_HGG:", cor_AIC_HGG)
    print("cor_BIC_HGG:", cor_BIC_HGG)
    print("cor_DNML_WND:", cor_DNML_WND)
    print("cor_AIC_WND:", cor_AIC_WND)
    print("cor_BIC_WND:", cor_BIC_WND)
    print("cor_AIC_naive:", cor_AIC)
    print("cor_BIC_naive:", cor_BIC)
    print("cor_MinGE:", cor_MinGE)

    plt.clf()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    def normalize(x):
        return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

    result["DNML_HGG"] = normalize(result["DNML_HGG"])
    result["DNML_WND"] = normalize(result["DNML_WND"])
    result["AIC_naive"] = normalize(result["AIC_naive"])
    result["BIC_naive"] = normalize(result["BIC_naive"])
    result["MinGE"] = normalize(result["MinGE"])

    ax.plot(result["model_n_dims"], result[
            "DNML_HGG"], label="DNML-HGG", color="red")
    ax.plot(result["model_n_dims"], result[
            "DNML_WND"], label="DNML-WND", color="black")
    ax.plot(result["model_n_dims"], result["AIC_naive"],
            label="AIC_naive", color="blue")
    ax.plot(result["model_n_dims"], result["BIC_naive"],
            label="BIC_naive", color="green")
    ax.plot(result["model_n_dims"], result[
            "MinGE"], label="MinGE", color="orange")
    plt.xscale('log')
    plt.xticks(result["model_n_dims"], fontsize=20)
    plt.yticks(fontsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=15)
    ax.set_xlabel("Dimensionality", fontsize=20)
    ax.set_ylabel("Normalized Criterion", fontsize=20)
    plt.tight_layout()

    plt.savefig(RESULTS + "/" + dataset_name +
                "/result_" + dataset_name + ".png")

    plt.clf()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.plot(result["model_n_dims"], is_a_score_hgg_list,
            label="Is-a scores of -log p_HGG(y, z)", color="red")
    ax.plot(result["model_n_dims"], is_a_score_wnd_list,
            label="Is-a scores of -log p_WND(y, z)", color="orange")
    ax.plot(result["model_n_dims"], is_a_score_naive_list,
            label="Is-a scores of -log p(y|z)", color="blue")
    plt.xscale('log')
    plt.xticks(result["model_n_dims"], fontsize=20)
    plt.yticks(fontsize=20)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=15)
    ax.set_xlabel("Dimensionality", fontsize=20)
    ax.set_ylabel("is a score", fontsize=20)
    plt.tight_layout()

    plt.savefig(RESULTS + "/" + dataset_name +
                "/result_is_a_score_" + dataset_name + ".png")

    return ret


def wn_dataset():
    dataset_name_list = [
        "animal",
        "group",
        "mammal",
        "solid",
        "tree",
        # "verb",
        "worker"
    ]
    bene_DNML_HGG = []
    bene_DNML_WND = []
    bene_AIC_naive = []
    bene_BIC_naive = []
    bene_MinGE = []
    cor_DNML_HGG = []
    cor_DNML_WND = []
    cor_AIC_naive = []
    cor_BIC_naive = []
    cor_MinGE = []

    for dataset_name in dataset_name_list:
        ret = result_wn(
            model_n_dims=[2, 3, 4, 5, 6, 7, 8, 16, 32, 64], dataset_name=dataset_name)
        bene_DNML_HGG.append(ret["bene_DNML_HGG"])
        bene_DNML_WND.append(ret["bene_DNML_WND"])
        bene_AIC_naive.append(ret["bene_AIC_naive"])
        bene_BIC_naive.append(ret["bene_BIC_naive"])
        bene_MinGE.append(ret["bene_MinGE"])
        cor_DNML_HGG.append(ret["cor_DNML_HGG"])
        cor_DNML_WND.append(ret["cor_DNML_WND"])
        cor_AIC_naive.append(ret["cor_AIC_naive"])
        cor_BIC_naive.append(ret["cor_BIC_naive"])
        cor_MinGE.append(ret["cor_MinGE"])

    bene_DNML_HGG = np.array(bene_DNML_HGG)
    bene_DNML_WND = np.array(bene_DNML_WND)
    bene_AIC_naive = np.array(bene_AIC_naive)
    bene_BIC_naive = np.array(bene_BIC_naive)
    bene_MinGE = np.array(bene_MinGE)
    cor_DNML_HGG = np.array(cor_DNML_HGG)
    cor_DNML_WND = np.array(cor_DNML_WND)
    cor_AIC_naive = np.array(cor_AIC_naive)
    cor_BIC_naive = np.array(cor_BIC_naive)
    cor_MinGE = np.array(cor_MinGE)

    print("benefit")
    print("DNML-HGG:", np.mean(bene_DNML_HGG), "±", np.std(bene_DNML_HGG))
    print("DNML-WND:", np.mean(bene_DNML_WND), "±", np.std(bene_DNML_WND))
    print("AIC_naive:", np.mean(bene_AIC_naive),
          "±", np.std(bene_AIC_naive))
    print("BIC_naive:", np.mean(bene_BIC_naive),
          "±", np.std(bene_BIC_naive))
    print("MinGE:", np.mean(bene_MinGE), "±", np.std(bene_MinGE))
    print("cor")
    print("DNML-HGG:", np.mean(cor_DNML_HGG), "±", np.std(cor_DNML_HGG))
    print("DNML-WND:", np.mean(cor_DNML_WND), "±", np.std(cor_DNML_WND))
    print("AIC_naive:", np.mean(cor_AIC_naive),
          "±", np.std(cor_AIC_naive))
    print("BIC_naive:", np.mean(cor_BIC_naive),
          "±", np.std(cor_BIC_naive))
    print("MinGE:", np.mean(cor_MinGE), "±", np.std(cor_MinGE))


if __name__ == "__main__":

    # print("Plot Example Figure")
    # plot_figure()
    # print("Results of Artificial Datasets")
    # artificial("HGG")
    # artificial("WND")
    print("Results of Scientific Collaboration Networks")
    realworld()
    # print("Results of WN dataset")
    # wn_dataset()
