import torch
import numpy as np

from tqdm import tqdm
from gpim import gprutils
from gpim.gpreg import skgpr


class GPSTRUCT(torch.nn.Module):
    """
    ...
    """

    def __init__(self):
        super(GPSTRUCT, self).__init__()

    @staticmethod
    def GP_Structured(
        imgdata: torch.Tensor,
        imgdata_gt: torch.Tensor,
        R2: torch.Tensor,
        iter__: int = 50,
    ):
        """
        Replicates the logic of the GP_Structured function.

        Parameters
        ----------
        :param imgdata:
            Input image array.
        :param imgdata_gt:
            Ground truth image array for error computation.
        :param R2:
            Binary mask or other reference array used to zero out invalid regions.
        :param iter__:
            Number of iterations.
        """

        # -> np.ndarray
        imgdata = imgdata.detach().cpu().numpy()
        imgdata_gt = imgdata_gt.detach().cpu().numpy()
        R2 = R2.detach().cpu().numpy()

        # HACK: assume BS=1
        # [B, H, W] -> [H, W]
        imgdata = imgdata.squeeze(0)
        imgdata_gt = imgdata_gt.squeeze(0)
        R2 = R2.squeeze(0)

        # ---------------------------------------------
        # 1) Normalize input image into [0, 1]
        # ---------------------------------------------
        orig_min = np.min(imgdata)
        orig_ptp = np.ptp(imgdata)  # max - min
        R = (imgdata - orig_min) / (orig_ptp + 1e-8)  # +1e-8 for safety

        # Use the value at [1, 1] as a "missing data" placeholder
        R[R == R[1, 1]] = np.nan

        # ---------------------------------------------
        # 2) Set up GP
        # ---------------------------------------------
        e1, e2 = R.shape
        xx, yy = np.mgrid[:e1, :e2]

        # Ensure float dtype
        xx = xx.astype(float)
        yy = yy.astype(float)

        X_true = np.array([xx, yy])

        # Build “sparse” (X, R_sparse) from the data and mask
        X, R_sparse = gprutils.corrupt_data_xy(X_true, R)

        lengthscale = [[1.0, 1.0], [4.0, 4.0]]
        kernel = "RBF"

        # ---------------------------------------------
        # 3) Run GP for iter__ iterations
        #    We'll only keep the final iteration result.
        # ---------------------------------------------

        gp_data_norm = None  # will store the final GP reconstruction in [0, 1]
        
        with torch.enable_grad():
            
            for ii in tqdm(range(iter__), desc="Training.."):
                skreconstructor = skgpr.skreconstructor(
                    X,
                    R_sparse,
                    X_true,
                    kernel,
                    lengthscale=lengthscale,
                    input_dim=2,
                    grid_points_ratio=1.0,
                    learning_rate=0.1,
                    iterations=ii,
                    calculate_sd=True,
                    num_batches=1,
                    use_gpu=True,
                    verbose=False,
                )

                mean, sd, hyperparams = skreconstructor.run()

                # Reshape the final GP output back to image shape (H, W)
                gp_data = mean.reshape(e1, e2)

                # In this code, gp_data is already on a [0, 1] scale
                gp_data_norm = gp_data.copy()

        # ---------------------------------------------
        # 4) Un-normalize final GP reconstruction
        # ---------------------------------------------
        # Bring gp_data_norm back to the original image distribution
        # shape: (H, W)
        final_pred_unorm = gp_data_norm * (orig_ptp + 1e-8) + orig_min

        # If you want to respect the zeroed-out region in R2, you could do:
        # final_pred_unorm[R2_np == 0] = imgdata_np[R2_np == 0]
        # Or some other strategy; depends on your exact goal.

        # ---------------------------------------------
        # 5) Expand dims back to (B, H, W) and return
        # ---------------------------------------------
        # Because we originally squeezed out batch=1, let's reintroduce it
        final_pred_unorm = final_pred_unorm[None, ...]  # shape: (1, H, W)

        return final_pred_unorm

    def forward(
        self,
        y: torch.Tensor,
        y_sparse: torch.Tensor,
        y_mask: torch.Tensor,
        iter__: int = 20,
    ):
        x: np.ndarray = GPSTRUCT.GP_Structured(
            y_sparse,
            y,
            y_mask,
            iter__,
        )
        return torch.tensor(x).cuda()

    @staticmethod
    def get(weights=None):
        """
        Returns an instance of the GPSTRUCT class.
        """
        return GPSTRUCT()


if __name__ == "__main__":
    model = GPSTRUCT()
    x = torch.rand((1, 128, 128))
    y = torch.rand((1, 128, 128))
    mask = torch.rand((1, 128, 128))
    model(x, y, mask)
