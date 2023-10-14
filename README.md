# DNML-HGG

## 1. About
This repository contains the implementation code of dimensionality and curvature selection.

## 2. Environment
- CPU: AMD EPYC 7452 (32 core) 2.35GHz
- OS: Ubuntu 18.04 LTS
- Memory: 512GB
- GPU: GeForce RTX 3090
- python: 3.6.9. with Anaconda.
The implementation assumes the availability of CUDA device.

## 3. How to Run
### Artificial Dataset
1. Execute `python datasets.py` to generate artificial datasets.

2. Execute `python experiment_lvm_space.py X Y Z W`, where X \in {euclidean, hyperbolic, spherical} is the possible datasets, Y \in {8} is the true dimensionality, and Z \in {0, 1, 2, 3} indicates the number of nodes 2^(Z+1)*100, and W \in {1, 2, ..., 12}. The variables X, Y, Z, and, W should be taken for all possible values.

### Real-world Dataset

1. Download the dataset from the URLs below. Then, put the txt files in `dataset/ca-AstroPh`, `dataset/ca-HepPh`, and the .graph and .p files in `dataset/airport`.
- AstroPh: https://snap.stanford.edu/data/ca-AstroPh.html
- HepPh: https://snap.stanford.edu/data/ca-HepPh.html
- Airport: https://github.com/HazyResearch/hgcn/raw/master/data/airport/airport.p

2. Execute `python transitive_closure.py`

3. Execute `python network_datasets.py`

4. Execute `python experiment_wn.py 0 0 0`,

3. Execute `python experiment_realworld_space.py X Y Z`, where X \in {0, 1, 2, 3, 4} is the id of the dataset (i.e, 0: AstroPh, 1:HepPh, 2: Airport, 3: WN-mammal, 4:WN-solid), Y \in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64} is the model dimensionality, and Z is the CUDA device in which the program runs. The combinations of X and Y are taken to be all possible ones.

### Results

1. Run `calc_metric_space.py`. For artificial dataset, selected dimensionality and metrics are shown in command line. For link prediction, selected dimensionalities, conciseness are shown in command line.

## 4. Author & Mail address
Ryo Yuki
- jie-cheng-ling@g.ecc.u-tokyo.ac.jp

## 5. Requirements & License
### Requirements
- torch==1.8.1
- nltk==3.6.7
- numpy
- scipy
- pandas
- matplotlib
