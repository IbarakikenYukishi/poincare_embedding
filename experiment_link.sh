#!/bin/bash
echo "dataset: $1"
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 2 0"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 3 1"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 4 2"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 5 3"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 6 0"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 7 1"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 8 2"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 9 3"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 10 0"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 11 1"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 12 2"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 13 3"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 14 0"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 15 1"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 16 2"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 32 3"
sleep 2
screen -dm bash -c "conda activate embed; python experiment_realworld_lorentz.py ${1} 64 3"
sleep 2