#!/bin/bash
echo "dataset: ${1}"
model_n_dims=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 32 64)
device=2
n_devices=4
for model_n_dim in "${model_n_dims[@]}"
do
    echo "conda activate embed; python experiment_realworld_space.py ${1} ${model_n_dim} ${device}"
    screen -dm bash -c "conda activate embed; python experiment_realworld_space.py ${1} ${model_n_dim} ${device}"
    sleep 2
    device=$((device+1))
    device=$((device % n_devices))
done