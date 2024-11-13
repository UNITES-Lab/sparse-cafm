from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger, LossLogger
from cldm.model import create_model, load_state_dict


def main():

    # Configs
    resume_path = './models/control_sd21_ini.ckpt'

    # is this per gpu or total?
    batch_size = 1

    logger_freq = 300
    learning_rate = 1e-4
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21.yaml').cpu()
    
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    train_dataset, val_dataset = MyDataset.get_train_val_datasets()
    
    # get train/val splits
    train_dataloader, val_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, num_workers=0, batch_size=batch_size, shuffle=False)
    
    img_logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[img_logger])
    
    # Train!
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# torch mp can't spawn new procs w/o a guard clause
if __name__ == "__main__":
    main()