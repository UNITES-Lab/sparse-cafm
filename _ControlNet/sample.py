from genericpath import isfile
import pickle
import torch
import os
import cv2
import numpy as np

from typing import Optional
from datasets.sapphire import (
    SapphireDataset,
    Formulation,
    SapphireDatasetFixedGridSampling,
)
from torch.utils.data import DataLoader
from cldm.cldm import ControlLDM
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


MODEL_PICKLE_FP = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__repos__/ControlNet/__weights__/sd_21_controlnet.pkl"
FT_CHECKPOINT_FP = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__repos__/ControlNet/__weights__/ft.pkl"
SD_CHECKPOINT = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__repos__/ControlNet/models/control_sd21_ini.ckpt"


def save_results_to_fp(results: dict, split: str, index: int):

    out_dir = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/full-sized-c-asm-data/target/y_hat"
    sample: torch.Tensor = results["samples_cfg_scale_9.00"].squeeze(0).detach().cpu()
    sample: np.ndarray = sample.numpy()

    # [H, W, C]
    # TODO: use actual sample max and min to perform [0, 1] norm
    # sample is normalized betwen [-1, 1]

    raveled_sample = sample.ravel()
    min_val = raveled_sample.min()
    max_val = raveled_sample.max()

    # normalize to [0, 1]
    sample = (sample - min_val) / (max_val - min_val)
    sample = (sample * 255).astype(np.uint8)
    sample = sample.transpose(0, 1, 2).transpose(1, 2, 0)
    out_fp = os.path.join(out_dir, split, f"{index:05}.png")

    # HACK: hard-coded file path
    cv2.imwrite(out_fp, sample)


def main():
    """
    Sample y_hat predictions from Saphire MoS2 dataset.
    Save all resulting samples as a local file.
    """

    # TODO
    # 1. DONE: correctly load checkpointed model
    # 2. verify data augmentations / pre-proc are correct
    # 2A. what img size was this checkpoint pre-trained on? [64, 224, 256]?
    # 2A. 64x64
    # 2B. does this model offer a fair baseline

    # model: ControlLDM = create_model("./models/cldm_v21.yaml").cpu()
    # model.load_state_dict(load_state_dict(SD_CHECKPOINT, location="cpu"))

    print(f"Loading ControlLDM model + weights from: {MODEL_PICKLE_FP}")
    model: Optional[ControlLDM] = None

    # still slow as all sin; model is probably just a very big boy
    # model serial : ~8.5 GB ):
    with open(FT_CHECKPOINT_FP, "rb") as f:
        model = pickle.load(f)

    model.cuda()
    model.eval()

    train_dataset = SapphireDatasetFixedGridSampling(
        split="train", formulation=Formulation.P_Z_BAR_X
    )
    # return

    train_dataloader = DataLoader(
        train_dataset, num_workers=0, batch_size=1, shuffle=False
    )

    val_dataset = SapphireDatasetFixedGridSampling(
        split="val", formulation=Formulation.P_Z_BAR_X
    )
    val_dataloader = DataLoader(val_dataset, num_workers=0, batch_size=1, shuffle=False)

    # how to align samples to file_paths / file_names?
    # is img size a problem?
    # what is shape of actual, generated samples?

    # training set samples
    iter = train_dataloader._get_iterator()
    
    out_dir = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/data/full-sized-c-asm-data/target/y_hat"
    
    for i in range(64 * 4):
        # skip existing samples
        out_fp = os.path.join(out_dir, "train", f"{i:05}.png")
        if os.path.isfile(out_fp):
            continue
        batch: dict = next(iter)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v.cuda()
        log: dict = model.log_images(batch, sample=True)
        save_results_to_fp(log, "train", i)

    # validation set samples
    iter = val_dataloader._get_iterator()
    for i in range(64 * 1):
        # skip existing samples
        out_fp = os.path.join(out_dir, "val", f"{i:05}.png")
        if os.path.isfile(out_fp):
            continue
        batch: dict = next(iter)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                v.cuda()
        log: dict = model.log_images(batch, sample=True)
        save_results_to_fp(log, "val", i)


if __name__ == "__main__":
    main()
