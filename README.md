# **SparseC-AFM**: Fast 2D-Material Acquisition & Analysis with Super Resolution Models
---

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](TODO)

This is the official Pytorch implementation of our paper: **SparseC-AFM**: a deep learning method for fast and accurate characterization of $MoS_{2}$ with C-AFM. We present a novel method for rapid acquisition and analysis of C-AFM scans using a super-resolution model based on the work of SwinIR. In this repository, you can find the datasets and model weights used in our paper, as well as scripts to **train** and **deploy** our model on ***your own datasets***.

Below we include our enviornments, data, and model weights.

## Getting Started

We use [anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) for all Python enviornment management. Clone our enviornment using the command below.

```bash
conda env create -f environment.yml
```

Once installed, activate the enviornment.

```bash
conda activate sparse-cafm
```

## Datasets

| Path | Material | Height Maps | Current Maps | Substrate | Scanning Mode | # Samples | # Data Points | Resolutions |
| :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Model Weights

## Citation
    @article{harris2025sparsec-afm,
      title={SparseC-AFM: a deep learning method for fast and accurate characterization of MoS2 with C-AFM},
      author={Harris, Hossain, Qui, Zhang, Ma, Chen, Gu, Tongay, Celano},
      journal={...},
      year={2025}
    }