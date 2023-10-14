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

2. Execute `python experiment_lvm_lorentz.py X Y Z W`, where X \in {HGG, WND} is the possible datasets (HGG means PUD here), Y \in {8, 16} is the true dimensionality, and Z \in {0, 1, 2, 3, 4} indicates the number of nodes 2^(Z+2)*100, and W \in {1, 2, ..., 12}. The variables X, Y, Z, and, W should be taken for all possible values.

### Real-world Dataset

1. Download the dataset from the URLs below. Then, put the txt files in `dataset/ca-AstroPh`, `dataset/ca-CondMat`, `dataset/ca-GrQc`, `dataset/ca-HepPh`, and the .graph and .p files in `dataset/cora`, `dataset/pubmed`, `dataset/airport`, and `dataset/bio-yeast-protein-inter`.
- AstroPh: https://snap.stanford.edu/data/ca-AstroPh.html
- CondMat: https://snap.stanford.edu/data/ca-CondMat.html
- GrQc: https://snap.stanford.edu/data/ca-GrQc.html
- HepPh: https://snap.stanford.edu/data/ca-HepPh.html
- Cora: https://github.com/HazyResearch/hgcn/blob/master/data/cora/ind.cora.graph
- PubMed: https://github.com/HazyResearch/hgcn/blob/master/data/pubmed/ind.pubmed.graph
- Airport: https://github.com/HazyResearch/hgcn/raw/master/data/airport/airport.p
- PPI(bio-yeast-protein-inter): https://networkrepository.com/bio-yeast-protein-inter.php

2. Execute `python transitive_closure.py`

3. Execute `python network_datasets.py`

3. Execute `python experiment_realworld_lorentz.py X Y Z`, where X \in {0, 1, 2, 3} is the id of the dataset (i.e, 0: AstroPh, 1:HepPh, 2: CondMat, 3: GrQc, 4:Airport, 5: Cora, 6: PubMed, and 7: PPI), Y \in {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64} is the model dimensionality, and Z is the CUDA device in which the program runs. The combinations of X and Y are taken to be all possible ones.

4. Execute `python experiment_wn.py X Y Z`, where X \in {0, 1, ..., 7} is the associated number of WN datasets (e.g., zero is associated with WN-mammal), Y is the model dimensionality, and Z is the CUDA device in which the program runs.

5. run `MinGE.py`

### Results

1. Run `calc_metric.py`. For artificial dataset, selected dimensionality and metrics are shown in command line. For link prediction, selected dimensionalities and AUC are shown in command line. For WN dataset, the selected dimensionalities and best dimensionalities are shown in command line. At the same time, the figures of each criterion are generated in `results`.

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
