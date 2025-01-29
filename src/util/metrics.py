import torch
import numpy as np

from torchmetrics.functional.image.psnr import psnr
from typing import Optional, Tuple


def PSNR(
    preds: torch.tensor,
    target: torch.tensor,
    range: Optional[Tuple[float, float]] = (-1.0, 1.0),
):
    """
    Source: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    """
    val = psnr(preds, target, data_range=range)
    return val


def OLDER(y_char: dict, y_sparse_char: dict) -> float:
    """
    Offline Domain-Expert Rating.
    
    A weighted average of percent difference of characterization of two current-maps using Celano labs scripts.
    """
    
    diffs = []
    for k1, k2 in zip(y_char.keys(), y_sparse_char.keys()):
        val1, val2 = y_char[k1], y_sparse_char[k2]
        if val1 == 0 and val2 == 0:
            diffs.append(0.0)
        else:
            avg_val = (val1 + val2) / 2.0
            # % diff = 2 * || val1 - val2 || / (val1 + val2)
            pdiff = abs(val1 - val2) / avg_val
            diffs.append(abs(pdiff))
    return np.mean(diffs)


if __name__ == "__main__":
    pass
