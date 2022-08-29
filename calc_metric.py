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


RESULTS = "results"


def artificial():

    # D_true_list = [4, 8, 16]
    D_true_list = [16]

    n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
    n_graphs = 12
    T_gap = 2

    for D_true in D_true_list:
        # if D_true == 4:
        #     label = [0, 1, 0, 0, 0, 0]
        # elif D_true == 8:
        #     label = [0, 0, 1, 0, 0, 0]
        # elif D_true == 16:
        #     label = [0, 0, 0, 1, 0, 0]

        # if D_true == 4:
        #     label = [0, 1, 0, 0, 0]
        # elif D_true == 8:
        #     label = [0, 0, 1, 0, 0]
        # elif D_true == 16:
        #     label = [0, 0, 0, 1, 0]

        for n_nodes in n_nodes_list:
            bene_DNML = []
            bene_AIC_naive = []
            bene_BIC_naive = []
            bene_AIC_naive_from_latent = []
            bene_BIC_naive_from_latent = []
            bene_MinGE = []
            estimate_DNML = []
            estimate_AIC_naive = []
            estimate_BIC_naive = []
            estimate_AIC_naive_from_latent = []
            estimate_BIC_naive_from_latent = []
            estimate_MinGE = []

            for n_graph in range(n_graphs):
                result = pd.read_csv(RESULTS + "/dim_" + str(D_true) +
                                     "/result_" + str(n_nodes) + "_" + str(n_graph) + ".csv")
                result = result.fillna(9999999999999)
                result_MinGE = pd.read_csv(
                    RESULTS + "/dim_" + str(D_true) + "/result_" + str(n_nodes) +
                    "_" + str(n_graph) + "_MinGE.csv")

                # if D_true<=16:
                #     result = result.drop(result.index[[5]])
                #     result_MinGE = result_MinGE.drop(result_MinGE.index[[5]])
                # if D_true<=8:
                #     result = result.drop(result.index[[4]])
                #     result_MinGE = result_MinGE.drop(result_MinGE.index[[4]])
                # if D_true<=4:
                #     result = result.drop(result.index[[3]])
                #     result_MinGE = result_MinGE.drop(result_MinGE.index[[3]])

                # print(label_ranking_average_precision_score([label], [-result["DNML_codelength"].values]))

                D_DNML = result["model_n_dims"].values[
                    np.argmin(result["DNML_codelength"].values)]
                D_AIC_naive = result["model_n_dims"].values[
                    np.argmin(result["AIC_naive"].values)]
                D_BIC_naive = result["model_n_dims"].values[
                    np.argmin(result["BIC_naive"].values)]
                D_AIC_naive_from_latent = result["model_n_dims"].values[
                    np.argmin(result["AIC_naive_from_latent"].values)]
                D_BIC_naive_from_latent = result["model_n_dims"].values[
                    np.argmin(result["BIC_naive_from_latent"].values)]
                D_MinGE = result_MinGE["model_n_dims"].values[
                    np.argmin(result_MinGE["MinGE"].values)]

                estimate_DNML.append(D_DNML)
                estimate_AIC_naive.append(D_AIC_naive)
                estimate_BIC_naive.append(D_BIC_naive)
                estimate_AIC_naive_from_latent.append(D_AIC_naive_from_latent)
                estimate_BIC_naive_from_latent.append(D_BIC_naive_from_latent)
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
                bene_AIC_naive.append(
                    max(0, 1 - abs(np.log2(D_AIC_naive) - np.log2(D_true)) / T_gap))
                bene_BIC_naive.append(
                    max(0, 1 - abs(np.log2(D_BIC_naive) - np.log2(D_true)) / T_gap))
                bene_AIC_naive_from_latent.append(
                    max(0, 1 - abs(np.log2(D_AIC_naive_from_latent) - np.log2(D_true)) / T_gap))
                bene_BIC_naive_from_latent.append(
                    max(0, 1 - abs(np.log2(D_BIC_naive_from_latent) - np.log2(D_true)) / T_gap))
                bene_MinGE.append(
                    max(0, 1 - abs(np.log2(D_MinGE) - np.log2(D_true)) / T_gap))

                plt.clf()
                fig = plt.figure(figsize=(8, 5))
                ax = fig.add_subplot(111)

                def normalize(x):
                    return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

                result["DNML_codelength"] = normalize(
                    result["DNML_codelength"])
                result["AIC_naive"] = normalize(result["AIC_naive"])
                result["BIC_naive"] = normalize(result["BIC_naive"])
                result["MinGE"] = normalize(result_MinGE["MinGE"])

                ax.plot(result["model_n_dims"], result[
                        "DNML_codelength"], label="DNML-HGG", color="red")
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

                plt.savefig(RESULTS + "/dim_" + str(D_true) + "/result_" +
                            str(D_true) + "_" + str(n_nodes) + "_" + str(n_graph) + ".png")

            bene_DNML = np.array(bene_DNML)
            bene_AIC_naive = np.array(bene_AIC_naive)
            bene_BIC_naive = np.array(bene_BIC_naive)
            bene_AIC_naive_from_latent = np.array(bene_AIC_naive_from_latent)
            bene_BIC_naive_from_latent = np.array(bene_BIC_naive_from_latent)
            bene_MinGE = np.array(bene_MinGE)

            estimate_DNML = np.array(estimate_DNML)
            estimate_AIC_naive = np.array(estimate_AIC_naive)
            estimate_BIC_naive = np.array(estimate_BIC_naive)
            estimate_AIC_naive_from_latent = np.array(
                estimate_AIC_naive_from_latent)
            estimate_BIC_naive_from_latent = np.array(
                estimate_BIC_naive_from_latent)
            estimate_MinGE = np.array(estimate_MinGE)

            print("n_nodes:", n_nodes)
            print("dimensionality:", D_true)
            print("DNML:", np.mean(bene_DNML), "±", np.std(bene_DNML))
            print("AIC_naive:", np.mean(bene_AIC_naive),
                  "±", np.std(bene_AIC_naive))
            print("BIC_naive:", np.mean(bene_BIC_naive),
                  "±", np.std(bene_BIC_naive))
            print("AIC_naive_from_latent:", np.mean(
                bene_AIC_naive_from_latent), "±", np.std(bene_AIC_naive_from_latent))
            print("BIC_naive_from_latent:", np.mean(
                bene_BIC_naive_from_latent), "±", np.std(bene_BIC_naive_from_latent))
            print("MinGE:", np.mean(bene_MinGE), "±", np.std(bene_MinGE))

            print("DNML:", np.mean(estimate_DNML), "±", np.std(estimate_DNML))
            print("AIC_naive:", np.mean(estimate_AIC_naive),
                  "±", np.std(estimate_AIC_naive))
            print("BIC_naive:", np.mean(estimate_BIC_naive),
                  "±", np.std(estimate_BIC_naive))
            print("AIC_naive_from_latent:", np.mean(
                estimate_AIC_naive_from_latent), "±", np.std(estimate_AIC_naive_from_latent))
            print("BIC_naive_from_latent:", np.mean(
                estimate_BIC_naive_from_latent), "±", np.std(estimate_BIC_naive_from_latent))
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

    D_DNML = result["model_n_dims"].values[
        np.argmin(result["DNML_codelength"].values)]
    D_AIC = result["model_n_dims"].values[
        np.argmin(result["AIC_naive"].values)]
    D_BIC = result["model_n_dims"].values[
        np.argmin(result["BIC_naive"].values)]
    D_MinGE = result["model_n_dims"].values[
        np.argmin(result["MinGE"].values)]

    model_n_dims = np.array(result["model_n_dims"])
    AUC_latent = np.array(result["AUC_latent"])
    AUC_naive = np.array(result["AUC_naive"])

    criterion_DNML_list = []
    criterion_AIC_list = []
    criterion_BIC_list = []
    criterion_MinGE_list = []

    DIV = 1000
    eps_range = 0.02

    # AUC latentの方
    AUC_max = max(AUC_latent)
    AUC_min = min(AUC_latent)
    eps_list = np.arange(DIV) * eps_range / DIV
    # eps_list = np.arange(DIV) * (AUC_max - AUC_min) / DIV

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

    for eps in eps_list:
        D_eps_list = model_n_dims[np.where(AUC_latent >= AUC_max - eps)[0]]
        D_min = min(D_eps_list)
        # print(D_eps_list)
        # print(D_min)

        criterion_DNML_list.append(calc_criterion(D_DNML, D_eps_list))
        criterion_MinGE_list.append(calc_criterion(D_MinGE, D_eps_list))

    criterion_DNML_list = np.array(criterion_DNML_list)
    criterion_MinGE_list = np.array(criterion_MinGE_list)

    # AUC latentの方
    AUC_max = max(AUC_naive)
    AUC_min = min(AUC_naive)
    # eps_list = np.arange(DIV) * (AUC_max - AUC_min) / DIV
    for eps in eps_list:
        D_eps_list = model_n_dims[np.where(AUC_naive >= AUC_max - eps)[0]]
        D_min = min(D_eps_list)

        criterion_AIC_list.append(calc_criterion(D_AIC, D_eps_list))
        criterion_BIC_list.append(calc_criterion(D_BIC, D_eps_list))

    criterion_AIC_list = np.array(criterion_AIC_list)
    criterion_BIC_list = np.array(criterion_BIC_list)

    # print(criterion_DNML_list)
    # print(criterion_AIC_list)
    # print(criterion_BIC_list)
    # print(criterion_MinGE_list)

    print("DNML:", np.average(criterion_DNML_list))
    print("AIC:", np.average(criterion_AIC_list))
    print("BIC:", np.average(criterion_BIC_list))
    print("MinGE:", np.average(criterion_MinGE_list))


def realworld():
    dataset_name_list = ["ca-AstroPh", "ca-CondMat", "ca-GrQc", "ca-HepPh"]
    # dataset_name_list = ["ca-GrQc",  "ca-HepPh"]

    n_dim_list = [2, 4, 8, 16, 32, 64]

    for dataset_name in dataset_name_list:

        result = pd.DataFrame()

        for n_dim in n_dim_list:
            row = pd.read_csv(RESULTS + "/" + dataset_name +
                              "/result_" + str(n_dim) + ".csv")
            result = result.append(row)

        result_MinGE = pd.read_csv(
            RESULTS + "/" + dataset_name + "/result_MinGE.csv")

        result = pd.merge(result, result_MinGE, on="model_n_dims", how="left")

        calc_new_metrics(result)

        D_DNML = result["model_n_dims"].values[
            np.argmin(result["DNML_codelength"].values)]
        D_AIC = result["model_n_dims"].values[
            np.argmin(result["AIC_naive"].values)]
        D_BIC = result["model_n_dims"].values[
            np.argmin(result["BIC_naive"].values)]
        D_MinGE = result["model_n_dims"].values[
            np.argmin(result["MinGE"].values)]

        print(dataset_name)
        print("DNML:", D_DNML)
        print("AIC:", D_AIC)
        print("BIC:", D_BIC)
        print("MinGE:", D_MinGE)

        print(result["AUC_latent"])
        print(result["AUC_naive"])

        # 各criterionの値
        plt.clf()

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.plot(result["model_n_dims"], result["AUC_latent"],
                label="AUC of -log p(y, z)", color="red")
        ax.plot(result["model_n_dims"], result["AUC_naive"],
                label="AUC of -log p(y|z)", color="blue")
        plt.xscale('log')
        plt.ylim(0.4, 0.9)
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

        result["DNML_codelength"] = normalize(result["DNML_codelength"])
        result["AIC_naive"] = normalize(result["AIC_naive"])
        result["BIC_naive"] = normalize(result["BIC_naive"])
        result["MinGE"] = normalize(result["MinGE"])

        ax.plot(result["model_n_dims"], result[
                "DNML_codelength"], label="DNML-HGG", color="red")
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


def result_wn(model_n_dims, dataset_name):
    print(dataset_name)
    is_a_score_naive_list = []
    is_a_score_latent_list = []

    data = np.load("dataset/wn_dataset/" + dataset_name +
                   "_data.npy", allow_pickle=True).item()
    is_a = data["is_a"]

    for model_n_dim in model_n_dims:
        lorentz_table_naive = np.load(
            RESULTS + "/" + dataset_name + '/embedding_' + str(model_n_dim) + '_naive.npy')
        lorentz_table_latent = np.load(
            RESULTS + "/" + dataset_name + '/embedding_' + str(model_n_dim) + '_latent.npy')

        # is_a = np.load('results/' + dataset_name + '/.npy')
        is_a_score_naive_list.append(is_a_score(
            is_a, model_n_dim, lorentz_table_naive, 100))
        is_a_score_latent_list.append(is_a_score(
            is_a, model_n_dim, lorentz_table_latent, 100))

    is_a_score_naive_list = np.array(is_a_score_naive_list)
    is_a_score_latent_list = np.array(is_a_score_latent_list)

    result = pd.read_csv(RESULTS + "/" + dataset_name + "/result.csv")

    result_MinGE = pd.read_csv(
        RESULTS + "/" + dataset_name + "/result_MinGE.csv")

    result = pd.merge(result, result_MinGE, on="model_n_dims", how="left")

    # result = result.iloc[1:, :]

    D_DNML = result["model_n_dims"].values[
        np.argmin(result["DNML_codelength"].values)]
    D_AIC_naive = result["model_n_dims"].values[
        np.argmin(result["AIC_naive"].values)]
    D_BIC_naive = result["model_n_dims"].values[
        np.argmin(result["BIC_naive"].values)]
    D_MinGE = result["model_n_dims"].values[
        np.argmin(result["MinGE"].values)]

    best_D_latent = result["model_n_dims"].values[
        np.argmax(is_a_score_latent_list)]
    best_D_naive = result["model_n_dims"].values[
        np.argmax(is_a_score_naive_list)]

    print("best latent:", best_D_latent)
    print("best naive:", best_D_naive)
    print("DNML:", D_DNML)
    print("AIC_naive:", D_AIC_naive)
    print("BIC_naive:", D_BIC_naive)
    print("MinGE:", D_MinGE)

    T_gap = 2
    ret = {
        "bene_DNML": max(0, 1 - abs(np.log2(D_DNML) - np.log2(best_D_latent)) / T_gap),
        "bene_AIC_naive": max(0, 1 - abs(np.log2(D_AIC_naive) - np.log2(best_D_naive)) / T_gap),
        "bene_BIC_naive": max(0, 1 - abs(np.log2(D_BIC_naive) - np.log2(best_D_naive)) / T_gap),
        "bene_MinGE": max(0, 1 - abs(np.log2(D_MinGE) - np.log2(best_D_naive)) / T_gap)
    }

    cor_DNML, _ = stats.spearmanr(
        is_a_score_latent_list, -result["DNML_codelength"].values)
    cor_AIC, _ = stats.spearmanr(
        is_a_score_naive_list, -result["AIC_naive"].values)
    cor_BIC, _ = stats.spearmanr(
        is_a_score_naive_list, -result["BIC_naive"].values)
    cor_MinGE, _ = stats.spearmanr(
        is_a_score_latent_list, -result["MinGE"].values)

    print("cor_DNML:", cor_DNML)
    print("cor_AIC:", cor_AIC)
    print("cor_BIC:", cor_BIC)
    print("cor_MinGE:", cor_MinGE)

    plt.clf()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    def normalize(x):
        return (x - np.min(x.values)) / (np.max(x.values) - np.min(x.values))

    result["DNML_codelength"] = normalize(result["DNML_codelength"])
    result["AIC_naive"] = normalize(result["AIC_naive"])
    result["BIC_naive"] = normalize(result["BIC_naive"])
    result["MinGE"] = normalize(result["MinGE"])

    ax.plot(result["model_n_dims"], result[
            "DNML_codelength"], label="DNML-HGG", color="red")
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

    ax.plot(result["model_n_dims"], is_a_score_latent_list,
            label="Is-a scores of -log p(y, z)", color="red")
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


if __name__ == "__main__":

    print("Plot Example Figure")
    plot_figure()
    print("Results of Artificial Datasets")
    artificial()
    print("Results of Scientific Collaboration Networks")
    realworld()
    print("Results of WN dataset")
    dataset_name_list = [
        "animal",
        "group",
        "mammal",
        "solid",
        "tree",
        "verb",
        "worker"
    ]
    bene_DNML = []
    bene_AIC_naive = []
    bene_BIC_naive = []
    bene_MinGE = []
    for dataset_name in dataset_name_list:
        ret = result_wn(
            model_n_dims=[2, 4, 8, 16, 32, 64], dataset_name=dataset_name)
        bene_DNML.append(ret["bene_DNML"])
        bene_AIC_naive.append(ret["bene_AIC_naive"])
        bene_BIC_naive.append(ret["bene_BIC_naive"])
        bene_MinGE.append(ret["bene_MinGE"])

    bene_DNML = np.array(bene_DNML)
    bene_AIC_naive = np.array(bene_AIC_naive)
    bene_BIC_naive = np.array(bene_BIC_naive)
    bene_MinGE = np.array(bene_MinGE)

    print("DNML:", np.mean(bene_DNML), "±", np.std(bene_DNML))
    print("AIC_naive:", np.mean(bene_AIC_naive),
          "±", np.std(bene_AIC_naive))
    print("BIC_naive:", np.mean(bene_BIC_naive),
          "±", np.std(bene_BIC_naive))
    print("MinGE:", np.mean(bene_MinGE), "±", np.std(bene_MinGE))
