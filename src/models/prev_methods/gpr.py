"""
Guassian Process Regression.
Inspired by: https://onlinelibrary.wiley.com/doi/pdf/10.1002/smll.202002878?casa_token=OP1n_oLqe4kAAAAA%3Aiovq39gdeNfEIR8Vyi_FRd3Ec9lz8cDm3m9MtmCoOXbg6w1ohs5YPom5x9uVK9S3wsqmssIPFzfsCIBM9w
"""

import gpim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def gpr_sr(y_sparse: torch.Tensor, SR: int = 2) -> torch.Tensor:
    
    x = y_sparse.clone()
    x = x.cpu().numpy()
    H, W = x.shape

    # create a train/test set
    train_indices = np.argwhere(~np.isnan(x))
    y_train = x[~np.isnan(x)]
    X_train = train_indices.astype(np.float64)
    X_train[:, 0] /= (H - 1)
    X_train[:, 1] /= (W - 1)

    # normalize -> [0, 1]
    X_train = train_indices.astype(np.float64)
    X_train[:, 0] /= (H - 1)
    X_train[:, 1] /= (W - 1)

    kernel = RBF(length_scale=0.1, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_train, y_train)

    H_hr, W_hr = SR * H, SR * W
    grid_x = np.linspace(0, 1, H_hr)
    grid_y = np.linspace(0, 1, W_hr)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing='ij')
    X_pred = np.column_stack([xx.ravel(), yy.ravel()])  # shape: (SE*H*SR*W, SR)

    y_pred, y_std = gp.predict(X_pred, return_std=True)
    y_pred_img = y_pred.reshape(H_hr, W_hr)
    y_std_img = y_std.reshape(H_hr, W_hr)

    return torch.Tensor(y_pred_img)


class GPReconstuctionInpainter(nn.Module):
    """
    [DEP]: An older implementation.
    """

    def __init__(
        self,
    ):
        """
        ...
        """
        super(GPReconstuctionInpainter, self).__init__()

    @torch.enable_grad()
    def forward(self, y_sparse: torch.Tensor) -> torch.Tensor:
        """
        :param y_sparse: EXACTLY [1, 3, 128, 128]

        Following the examples provided in gpim repo:
        https://github.com/ziatdinovmax/GPim/blob/master/examples/notebooks/GP_2D3D_images.ipynb.
        """

        # HACK: assume y_sparse has EXACTLY shape: [1, 3, 128, 128]
        R = y_sparse.clone().cpu()[0, 0, :, :]
        R = R.numpy().astype(float)

        # HACK: we hard coded the sparsity of incoming y_sparse, assume always 50%
        R[:, ::2] = np.NaN

        # Get full (ideal) grid indices
        X_full = gpim.utils.get_full_grid(R, dense_x=1)

        # Get sparse grid indices
        X_sparse = gpim.utils.get_sparse_grid(R)

        # run GP reconstruction to obtain mean prediction and uncertainty for each predictied point
        recon = gpim.reconstructor(
            X_sparse,
            R,
            X_full,
            learning_rate=0.1,
            iterations=2,
            use_gpu=True,
            verbose=False,
        )

        # train + predict
        mean, sd, hyperparams = recon.run()

        e1, e2 = R.shape

        # (128, 128)
        pred = mean.reshape(e1, e2)

        # (128, 128) -> (3, 128, 128)
        pred = np.stack([pred] * 3, axis=0)

        # (3, 128, 128) -> (1, 3, 128, 128)
        pred = np.expand_dims(pred, 0)

        # -> tensor -> gpu
        pred = torch.Tensor(pred).cuda()

        return pred

    @staticmethod
    def get(weights=None):
        return GPReconstuctionInpainter()
