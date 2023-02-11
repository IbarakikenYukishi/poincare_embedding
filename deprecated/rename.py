import os


true_dim_list = [8, 16]
dim_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
n_nodes_list = [400, 800, 1600, 3200, 6400, 12800]
n_graph_list = range(12)

for true_dim in true_dim_list:
    for dim in dim_list:
        for n_nodes in n_nodes_list:
            for n_graph in n_graph_list:
                print("results/WND/dim_" + str(true_dim) + "/result_" + str(
                        dim) + "_" + str(n_nodes) + "_" + str(n_graph) + "_hgg.pth")
                print(os.path.exists("results/WND/dim_" + str(true_dim) + "/result_" + str(
                        dim) + "_" + str(n_nodes) + "_" + str(n_graph) + "_hgg.pth"))
                if os.path.exists("results/WND/dim_" + str(true_dim) + "/result_" + str(
                        dim) + "_" + str(n_nodes) + "_" + str(n_graph) + "_hgg.pth"):
                    os.rename(
                        "results/WND/dim_" + str(true_dim) + "/result_" + str(
                            dim) + "_" + str(n_nodes) + "_" + str(n_graph) + "_hgg.pth",
                        "results/WND/dim_" + str(true_dim) + "/result_" + str(
                            dim) + "_" + str(n_nodes) + "_" + str(n_graph) + "_wnd.pth",
                    )
