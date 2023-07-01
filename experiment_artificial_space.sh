#!/bin/bash
echo "dataset: ${1}, true_dim: ${2}"
partitions=(0 1 2 3 4 5 6 7 8 9 10 11)
n_nodes_list=(0 1 2)
for n_nodes in "${n_nodes_list[@]}"
do
for partition in "${partitions[@]}"
do
    echo "conda activate embed; python experiment_lvm_space.py ${1} ${n_nodes} ${2} ${partition}"
    screen -dm bash -c "conda activate embed; python experiment_lvm_space.py ${1} ${n_nodes} ${2} ${partition}"
    sleep 2
done
done
