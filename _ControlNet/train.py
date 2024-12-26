import os
import sys
import yaml
import datetime
import pytorch_lightning as pl

from share import *
from torch.utils.data import DataLoader
from datasets.sapphire import SapphireDataset, Formulation
from util.logger import ExperimentLogger
from cldm.model import create_model, load_state_dict
from cldm.logger import ImageLogger, ScuffedLogger

TRAIN_CONFIG_FP = (
    "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/configs/train.yaml"
)
RUNS_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__exps__/__controlnet_runs__"


def parse_config(fp: str) -> dict:
    with open(fp, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Train a controlnet to predict X -> model -> y_hat.
    """

    # configs
    resume_path = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/ControlNet/models/control_sd21_ini.ckpt"

    # dataset params
    dataset_name = "saphire-50-50-mask"

    # hyper params
    batch_size = 1
    logger_freq = 1000
    learning_rate = 1e-4
    sd_locked = True
    only_mid_control = False

    # set up the logger
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_fp = os.path.join(RUNS_DIR, f"{date}-{dataset_name}", "log.csv")
    os.makedirs(os.path.dirname(log_fp), exist_ok=True)
    logger = ScuffedLogger.get_instance()
    logger.set_csv_path(log_fp)

    # first use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model("./models/cldm_v21.yaml").cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # train_dataset, val_dataset = MyDataset.get_train_val_datasets(dataset_name)

    train_dataset, val_dataset = SapphireDataset(
        split="train", formulation=Formulation.P_Y_BAR_X_CN
    ), SapphireDataset(
        split="val", formulation=Formulation.P_Y_BAR_X_CN, steps_per_epoch=1
    )

    # get train/val splits
    train_dataloader, val_dataloader = DataLoader(
        train_dataset, num_workers=0, batch_size=batch_size, shuffle=False
    ), DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False)

    config = parse_config(TRAIN_CONFIG_FP)
    logger = ExperimentLogger(
        config_fp=TRAIN_CONFIG_FP,
        root=config["logging"]["root"],
        exp_name=config["logging"]["exp_name"],
        log_interval=config["logging"]["log_interval"],
    )
    logger.add_result_columns(["step", "train_l1", "train_psnr", "val_l1", "val_pnsr"])
    img_logger = ImageLogger(batch_frequency=logger_freq, exp_root_dir=logger.exp_dir)
    img_logger.register_logger(logger)

    # start training
    trainer = pl.Trainer(gpus=[0], precision=32, callbacks=[img_logger])

    # train!
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


# torch mp can't spawn new procs w/o a guard clause
if __name__ == "__main__":
    main()
