import torch
import numpy as np
from typing import List


# https://discuss.pytorch.org/t/pytorch-tensor-to-device-for-a-list-of-dict/66283
def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def convert_to_img_like(*args: torch.Tensor) -> List[np.ndarray]:
    """
    Convert one or more tensors from any range to [0, 255].
    Cast to int, move to CPU, and return them all as NumPy arrays.
    """
    results = []
    for x in args:
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val)
        x = x * 255
        x = x.detach().cpu().int().numpy()
        results.append(x)
    return results
