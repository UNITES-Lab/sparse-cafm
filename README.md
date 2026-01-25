---
title: SparseC-AFM
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: apache-2.0
---

# **SparseC-AFM**: fast 2D-material acquisition & analysis with super resolution models
---

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2507.13527v1)

This is the official Pytorch implementation of our paper: **SparseC-AFM**: a deep learning method for fast and accurate characterization of MoS<sub>2</sub> with C-AFM. We present a novel method for rapid acquisition and analysis of C-AFM scans using a super-resolution model based on the work of SwinIR. In this repository, you can find the datasets and model weights used in our paper, as well as scripts to **train** and **deploy** our model on ***your own datasets***.

## Getting Started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
uv sync
uv run python app.py
```

Then open http://127.0.0.1:7860 in your browser.

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
    @inproceedings{Harris2025,
      title = {Sparse C-AFM: a deep learning method for fast and accurate characterization of MoS2 with conductive atomic force microscopy},
      url = {http://dx.doi.org/10.1117/12.3067427},
      DOI = {10.1117/12.3067427},
      booktitle = {Low-Dimensional Materials and Devices 2025},
      publisher = {SPIE},
      author = {Harris,  Levi and Hossain,  Md Jayed and Qui,  Mufan and Zhang,  Ruichen and Ma,  Pingchuan and Chen,  Tianlong and Gu,  Jiaqi and Tongay,  Seth Ariel and Celano,  Umberto},
      editor = {Kobayashi,  Nobuhiko P. and Talin,  A. Alec and Davydov,  Albert V. and Islam,  M. Saif},
      year = {2025},
      month = sep,
      pages = {35}
    }

## License

We release our work under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) ❤️. 
