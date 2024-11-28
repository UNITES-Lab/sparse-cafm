import torch
import torch.nn as nn
from util.loss import DiceLoss, FocalLoss

LOSS_FUNCTIONS = {
    "MSE": nn.MSELoss,
    "L1": nn.L1Loss,
    "CrossEntropy": nn.CrossEntropyLoss,
    "SmoothL1": nn.SmoothL1Loss,
    "Dice": DiceLoss,
    "Focal": FocalLoss,
    "Huber": nn.HuberLoss
}

OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}
