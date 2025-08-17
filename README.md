# **SparseC-AFM**: fast 2D-material acquisition & analysis with super resolution models
---

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2507.13527v1)

This is the official Pytorch implementation of our paper: **SparseC-AFM**: a deep learning method for fast and accurate characterization of MoS<sub>2</sub> with C-AFM. We present a novel method for rapid acquisition and analysis of C-AFM scans using a super-resolution model based on the work of SwinIR. In this repository, you can find the datasets and model weights used in our paper, as well as scripts to **train** and **deploy** our model on ***your own datasets***.

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

| Path | Material | Height Maps | Current Maps | Substrate | Mode | # Samples | # Data Points | Resolutions |
| ---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |  :---: |
| [`data/raw-data/3-12-25`](data/raw-data/3-12-25)   | BTO             | ✅  | ❌ | ---                          | Tapping (AFM Only) | 4 | 16 | {64, 128, 256, 512} |
| [`data/raw-data/2-6-25`](data/raw-data/2-6-25)     | MoS<sub>2</sub> | ✅  | ✅ | SiO<sub>2</sub>-Si           | Contact            | 1 | 4  | {64, 128, 256, 512} |
| [`data/raw-data/1-23-25`](data/raw-data/1-23-25)   | MoS<sub>2</sub> | ✅  | ✅ | SiO<sub>2</sub>-Si           | Contact            | 1 | 5  | {512}|
| [`data/raw-data/11-19-24`](data/raw-data/11-19-24) | MoS<sub>2</sub> | ✅  | ✅ | SiO<sub>2</sub>-Si, Sapphire | Contact            | 2 | 10 | {512} |

## Model Weights

| Path | Upscaling Factor | Material| Height Maps | Current Maps | Substrates |
| ---: | :---: | :---: |:---: | :---: | :---: |
| [`data/weights/...2x.pth`](data/weights/2x/2x.pth) | $$\times2$$ | MoS<sub>2</sub> | ✅ | ✅ | SiO<sub>2</sub>-Si, Sapphire |
| [`data/weights/...4x.pth`](data/weights/4x/4x.pth) | $$\times4$$ | MoS<sub>2</sub>  | ✅ | ✅ | SiO<sub>2</sub>-Si, Sapphire | 
| [`data/weights/...8x.pth`](data/weights/8x/8x.pth) | $$\times8$$ | MoS<sub>2</sub> |  ✅ | ✅ | SiO<sub>2</sub>-Si, Sapphire |

## Citation
    @article{harris2025sparsec-afm,
      title={SparseC-AFM: a deep learning method for fast and accurate characterization of MoS2 with C-AFM},
      author={Harris, Hossain, Qui, Zhang, Ma, Chen, Gu, Tongay, Celano},
      conference={SPIE},
      year={2025}
    }

## License

We release our work under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) ❤️. 