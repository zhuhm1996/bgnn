# BGNN
This is the TensorFlow implementation of our paper accepted by IJCAI 2020:

>Hongmin Zhu, Fuli Feng, Xiangnan He, Xiang Wang, Yan Li, Kai Zheng, Yongdong Zhang, Bilinear Graph Neural Network with Neighbor Interactions. [Paper in arXiv](https://arxiv.org/abs/2002.03575).

## Introduction
We propose a new graph convolution operator, augmenting the weighted sum with pairwise interactions of the representations of neighbor nodes. We specify two BGNN models named BGCN and BGAT, based on the well-known GCN and GAT, respectively.

## Dependencies
The code is tested by server with RTX 1080Ti running in a docker container which includes the following packages:
* python 3.6.3
* tensorflow 1.4.0
* numpy 1.13.3
* scipy 1.0.0
* networkx 2.0

In addition, CUDA 8 and cuDNN 6 have been used.

## Simulation example

## Dataset
We utilize three benchmark datasets of citation network---Pubmed, Cora and Citeseer. Download from [this](https://github.com/tkipf/gcn).
