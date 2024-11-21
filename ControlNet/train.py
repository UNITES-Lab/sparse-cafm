
import datetime
import pytorch_lightning as pl
import os

from share import *
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger, ScuffedLogger
from cldm.model import create_model, load_state_dict

RUNS_DIR = "/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/ControlNet/__runs__"

def main():

    # configs
    resume_path = "./models/control_sd21_ini.ckpt"
    
    # dataset params
    dataset_name = "bs-ds-64x64"

    # hyper params
    batch_size = 1
    logger_freq = 300
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

    train_dataset, val_dataset = MyDataset.get_train_val_datasets(dataset_name)

    # get train/val splits
    train_dataloader, val_dataloader = DataLoader(
        train_dataset, num_workers=0, batch_size=batch_size, shuffle=True
    ), DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False)

    img_logger = ImageLogger(batch_frequency=logger_freq)
    
    # start training
    trainer = pl.Trainer(gpus=[2], precision=32, callbacks=[img_logger])

    # train!
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )


# torch mp can't spawn new procs w/o a guard clause
if __name__ == "__main__":
    main()
