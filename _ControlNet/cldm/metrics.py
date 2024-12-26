import torch
import cv2 
import numpy as np 

from math import log10, sqrt 
from typing import Optional, Tuple
  
def calc_psnr(preds: torch.tensor, target: torch.tensor, range: Optional[float] = 1.):
    """
    Source: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    """
    return 0.0