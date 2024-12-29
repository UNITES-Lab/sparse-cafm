import torch
import cv2 
import numpy as np 

from math import log10, sqrt 
from torchmetrics.functional.image import peak_signal_noise_ratio
from typing import Optional, Tuple
  
def PSNR(preds: torch.tensor, target: torch.tensor, range: Optional[Tuple[float, float]] = (-1., 1.)):
    """
    Source: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    """
    
    val = peak_signal_noise_ratio(preds, target)
    return val

if __name__ == "__main__":
    original = cv2.imread("/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__repos__/ControlNet/image_log/64x64-crop-lr-1e-5/reconstruction_gs-071185_e-000619_b-000000.png")
    compressed = cv2.imread("/playpen/mufan/levi/tianlong-chen-lab/material-super-resolution/__repos__/ControlNet/image_log/64x64-crop-lr-1e-5/samples_cfg_scale_9.00_gs-071185_e-000619_b-000000.png")
    print(PSNR(original, compressed))