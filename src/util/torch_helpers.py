import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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


def apply_color_palette(map_like: np.ndarray) -> np.ndarray:
    """
    Applies a color palette to a 2D input array (H, W) and returns
    an RGB image (H, W, C).

    Parameters:
    -----------
    map_tensor : np.ndarray
        A  NumPy array representing
        some feature map or heatmap data.

    Returns:
    --------
    colored_image : np.ndarray
        An array of shape (H, W, C), where each pixel has RGB values
        in the [0, 255] range.
    """
    # -> [0, 1]
    map_like = map_like.astype(np.float32)
    min_val, max_val = np.min(map_like), np.max(map_like)
    if max_val > min_val:
        normalized_map = (map_like - min_val) / (max_val - min_val)
    else:
        # edge case: map has all the same value
        normalized_map = np.zeros_like(map_like, dtype=np.float32)
    cmap = cm.get_cmap("viridis")
    colored_map = cmap(normalized_map)
    colored_image = colored_map[..., :3]
    # -> [0, 255]
    return colored_image * 255


def convert_to_img_like(*args: torch.Tensor) -> List[np.ndarray]:
    """
    Convert one or more tensors from any range to [0, 255].
    Cast to int, move to CPU, and return them all as NumPy arrays.
    """
    results = []
    for x in args:
        x = x.detach().cpu().numpy()
        results.append(apply_color_palette(x))
    return results
