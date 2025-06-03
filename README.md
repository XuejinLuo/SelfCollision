# Minimum Distance Prediction Network

![Minimum Distance Prediction Network](assets/DistanceNetwork.png)
> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**\
> Xuejin Luo*, Runshi Zhang*, Siqin Yang*, Ruizhi Zhang*, Junchen Wang*\
> Paper: 

## About

This repository contains a minimum distance prediction network for dual-arm manipulators to avoid self-collision. The network inputs are the joint angles of neighboring frames of the dual arms, and the outputs are the minimum distance and gradient. The latter can be computed in real time as a self-touch avoidance constraint in quadratic programming at an operating frequency of 50Hz.

`checkpoints`: folder that saves the trained model.
`data`: store the minimum distance data set of dual-arm manipulators.
`models`: contains the internal structure of each model.
`optimize`: testing model constraint calculations in quadratic programming.
`scripts`: scripts that contain training and testing models
`train`: training code manager.
`utils`: other functions.

## Requirements

- Ubuntu 22.04
- NVIDIA GPU
- PyTorch 2.0.0+
- CUDA 11.7+
- mamba_ssm

## Train

```bash
sh scripts\train\SSM_cross_attn_train.sh
```

## Eval

```bash
sh scripts\eval\SSM_cross_attn_eval.sh
```

## Data
Each piece of data in the dataset is saved in the form of a minimum distance + 14 joint angles (normalized). Since the network relies on temporal prediction, it is desirable that each data group contains multiple consecutive pieces of data. Our data set has been uploaded to the Google cloud disk, you only need to download and put it in the data folder.

## Citation
If you find this repository helpful, please consider citing:

