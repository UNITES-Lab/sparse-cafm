import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.ndimage import distance_transform_edt


class NearestNeighborsInpainter(nn.Module):
    """
    Fill each missing pixel by copying the color from its nearest known
    neighbor (by Euclidean distance). Uses SciPy's distance_transform_edt
    under the hood, adapted to (B, H, W) inputs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, target_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param target_image: (B, H, W), float or uint8
        :param mask:         (B, H, W), values in {0,1},
                             1 => known region, 0 => missing region
        :return predicted_image: (B, H, W), where:
                                 - For mask=1 (known), predicted_image=0
                                 - For mask=0 (missing), predicted_image is filled
                                   with the nearest known neighbor's value.
        """
        # Ensure mask is boolean
        mask_bool = mask > 0

        B, H, W = target_image.shape
        device = target_image.device
        dtype = target_image.dtype

        # Convert to NumPy for distance_transform_edt
        target_np = target_image.detach().cpu().numpy()
        mask_np   = mask_bool.detach().cpu().numpy()

        # Prepare an output buffer in NumPy to hold the inpainted values
        filled_np = np.zeros_like(target_np)  # same shape as target_np: (B, H, W)

        for b in range(B):
            # mask_np[b] is shape (H, W), True where known, False where missing
            # distance_transform_edt expects True=foreground if we want distance to background.
            # But we want the "missing" region to be the foreground, so we use ~mask_np[b].
            dist, (idx_y, idx_x) = distance_transform_edt(
                ~mask_np[b],
                return_distances=True,
                return_indices=True
            )
            # For each pixel, copy from the nearest known pixel.
            filled_np[b] = target_np[b, idx_y, idx_x]

        # Convert back to torch
        filled_torch = torch.from_numpy(filled_np).to(device=device, dtype=dtype)

        # According to the spec:
        # * For known (mask=1), output 0
        # * For missing (mask=0), output the filled color
        # => Multiply the filled result by the inverse of mask
        predicted_image = filled_torch * (~mask_bool).float()

        return predicted_image

    @staticmethod
    def get(weights=None):
        return NearestNeighborsInpainter()


class LinearInterpolationInpainter(nn.Module):
    """
    TODO: verify correctness.
    """

    def __init__(self):
        super(LinearInterpolationInpainter, self).__init__()

    def __init__(self):
        super(LinearInterpolationInpainter, self).__init__()

    def forward(self, target_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Given:
          - target_image: (B, H, W) ground-truth image
          - mask: (B, H, W) boolean or {0,1}, where 1 => known, 0 => missing

        Returns:
          - predicted_image: (B, H, W), with nonzero values only for mask=0.
            The known region is zeroed out, and the missing region is filled
            by linear interpolation of nearby known values.
        """

        # 1) Zero out missing pixels in the image.
        #    masked_image is (B, H, W).
        masked_image = target_image * mask  # (B, H, W)

        # 2) Create a normalized 3×3 kernel (sum=1).
        kernel = torch.ones(
            1, 1, 3, 3, device=target_image.device, dtype=target_image.dtype
        )
        kernel = kernel / kernel.sum()  # Each element 1/9

        # 3) Use conv2d with an extra channel dimension to sum known neighbors.
        #    The result shape after conv2d is (B, 1, H, W).
        sum_of_neighbors = F.conv2d(
            masked_image.unsqueeze(1), kernel, padding=1  # (B, 1, H, W)
        )

        # 4) Convolve the mask (converted to float) likewise
        #    to determine how many neighbors contributed.
        sum_of_masks = F.conv2d(
            mask.float().unsqueeze(1), kernel, padding=1  # (B, 1, H, W)
        )

        # 5) Compute average of neighbors. Add small eps to avoid div-by-zero.
        eps = 1e-8
        interpolated_values = sum_of_neighbors / (sum_of_masks + eps)  # (B, 1, H, W)

        # 6) We only want these interpolated values where mask=0.
        #    We'll multiply by ~mask to keep them only in the missing region.
        #    Then add the known pixels (which remain zeroed in predicted_image).
        predicted_image = interpolated_values * (~mask).unsqueeze(
            1
        ) + masked_image.unsqueeze(1)

        # Squeeze out the extra channel dimension to return shape (B, H, W).
        return predicted_image.squeeze(1)

    @staticmethod
    def get(weights=None):
        return LinearInterpolationInpainter()


class BicubicInterpolationInpainter(nn.Module):
    """
    A simple inpainter that uses PyTorch's built-in bicubic interpolation
    to fill missing rows in an image. This implementation assumes that
    every other row is known (i.e., the mask is True for those rows).
    """

    def __init__(self):
        super().__init__()

    def forward(self, target_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_image (torch.Tensor): (B, H, W) ground-truth images.
            mask (torch.Tensor): (B, H, W) boolean mask, where True = known and
                                 False = missing (every other row is True).

        Returns:
            torch.Tensor: (B, H, W) images with missing rows inpainted by
                          bicubic interpolation. Known pixels are preserved.
        """
        B, H, W = target_image.shape

        masked_image = target_image * mask

        # Add a channel dimension to make it (B, 1, H, W)
        # so that we can call PyTorch’s interpolation utilities:
        x_with_channel = target_image.unsqueeze(1)  # (B, 1, H, W)

        # For simplicity, assume the mask pattern is consistent across the batch:
        # Identify which row indices are True in the first sample’s mask
        known_rows = mask[0].any(dim=1)  # shape: (H,)
        known_row_indices = torch.nonzero(known_rows).squeeze(
            1
        )  # shape: (#known_rows,)

        # Gather only the known rows from every image in the batch
        known_image = x_with_channel[
            :, :, known_row_indices, :
        ]  # (B, 1, #known_rows, W)

        # Upsample these known rows back to full resolution using bicubic
        interpolated_values = F.interpolate(
            known_image, size=(H, W), mode="bicubic", align_corners=False
        ).squeeze(
            1
        )  # (B, H, W)

        # Copy the true known pixels back in (so they match exactly):
        output = interpolated_values * (~mask) + masked_image

        return output

    @staticmethod
    def get(weights=None):
        return BicubicInterpolationInpainter()


class AMPInpainter(nn.Module):
    """
    A simple demonstration of how to use CAMP to fill in missing pixels
    (mask=0) in an image.
    """

    def __init__(
        self,
        rho=0.5,  # Bernoulli-Gaussian prior: fraction of non-zero pixels
        theta_bar=0.0,  # Mean of the Gaussian prior
        theta_hat=1.0,  # Variance of the Gaussian prior
        max_iter=300,  # Max iterations for CAMP
        sol_tol=1e-2,  # Residual tolerance
    ):
        super(AMPInpainter, self).__init__()
        self.rho = rho
        self.theta_bar = theta_bar
        self.theta_hat = theta_hat
        self.max_iter = max_iter
        self.sol_tol = sol_tol

    @staticmethod
    def _iBGoG(y_bar, y_hat, ipars):
        """
        Prior estimator - Bernoulli-Gaussian input and Gaussian output.
        Returns (a, v) which are the means and variances of the posterior estimate.
        """

        # Unpack input channel parameters
        theta_bar = ipars["theta_bar"]
        theta_hat = ipars["theta_hat"]
        rho = ipars["rho"]

        # Common parameters
        common_denominator = y_hat + theta_hat
        M_bar = (y_hat * theta_bar + y_bar * theta_hat) / (common_denominator)
        V_hat = y_hat * theta_hat / (common_denominator)

        # Common part in the Bernoulli-Gaussian prior
        z = (1.0 - rho) / rho * np.sqrt(theta_hat / V_hat)
        z *= np.exp(
            -0.5 * (y_bar**2 / y_hat - (theta_bar - y_bar) ** 2 / (y_hat + theta_hat))
        )

        # Estimated mean and variance
        a = M_bar / (z + 1.0)
        v = z * a**2 + V_hat / (z + 1.0)

        return a, v

    @staticmethod
    def camp(y, F, d_s, ipars, T=300, sol_tol=1e-2):
        """
        CAMP: Computational Approximate Message Passing.

        :param y:      Shape (M,) – measurement vector
        :param F:      Shape (M, N) – system matrix
        :param d_s:    float – initial step size
        :param ipars:  dict with keys ['rho', 'theta_bar', 'theta_hat']
        :param T:      int – max number of iterations
        :param sol_tol:float – tolerance for early stopping
        :return:
        a_est_col – shape (N, 1) final estimate of the unknown vector
        v_est_col – shape (N, 1) final estimate of the variance
        """
        M, N = F.shape
        a_est = np.zeros(N)
        v_est = np.ones(N)
        w = y.copy()
        v = np.ones(M)
        F_2 = F * F  # elementwise square of F

        for t in range(T):
            # 1) Message passing update for 'w' and 'v'
            w = np.dot(F, a_est) - np.dot(F_2, v_est) * (y - w) / (d_s + v)
            v = np.dot(F_2, v_est)

            # 2) Compute y_hat and y_bar
            y_hat = 1.0 / np.dot(F_2.T, 1.0 / (d_s + v))
            y_bar = a_est + y_hat * np.dot(F.T, (y - w) / (d_s + v))

            # 3) Prior update (Bernoulli-Gaussian)
            a_est, v_est = AMPInpainter._iBGoG(y_bar, y_hat, ipars)

            # # 4) Optionally update d_s
            # d_s = d_s * np.sum(((y - w) / (d_s + v)) ** 2) / np.sum(1.0 / (d_s + v))

            # 5) Check residual
            r = y - np.dot(F, a_est)
            if np.linalg.norm(r, ord=2) < sol_tol * np.linalg.norm(y, ord=2):
                break

        # Reshape final estimates
        a_est_col = a_est.reshape((N, 1))
        v_est_col = v_est.reshape((N, 1))
        return a_est_col, v_est_col

    def forward(self, target_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param target_image: (B, C, H, W) – original image
        :param mask:        (B, C, H, W) – binary {0,1} or boolean
                            1 => known pixel, 0 => missing
        :return:
            predicted_image: (B, C, H, W) – same shape. The missing region
                             is reconstructed using the CAMP solver.
        """
        B, C, H, W = target_image.shape
        predicted_image = torch.zeros_like(target_image)

        # Prior parameters for Bernoulli-Gaussian
        ipars = {
            "rho": self.rho,
            "theta_bar": self.theta_bar,
            "theta_hat": self.theta_hat,
        }

        # We'll do the inpainting one sample at a time for clarity:
        for b in range(B):
            # (1) Convert sample to numpy
            x_np = target_image[b].detach().cpu().numpy().flatten()  # shape: (C*H*W,)
            m_np = mask[b].detach().cpu().numpy().flatten()  # shape: (C*H*W,)

            # (2) Identify which pixels are known
            known_indices = np.where(m_np > 0.5)[0]  # or m_np == 1 if strictly {0,1}

            # (3) Build measurement matrix F (size M×N, with M = #known, N = total pixels)
            N = x_np.size
            M = len(known_indices)
            F = np.zeros((M, N), dtype=np.float32)
            for i, idx in enumerate(known_indices):
                F[i, idx] = 1.0

            # (4) Measurement is the known pixel values from the image
            y = x_np[known_indices]

            # (5) Run CAMP to estimate the entire N-dim image vector
            #     Start with a small step size, say 1e-2:
            d_s_init = 1e-2
            a_est_col, _ = AMPInpainter.camp(
                y=y,
                F=F,
                d_s=d_s_init,
                ipars=ipars,
                T=self.max_iter,
                sol_tol=self.sol_tol,
            )
            a_est = a_est_col[:, 0]  # shape: (N,)

            # (6) Combine the CAMP estimate with known pixels.
            #     - We only overwrite missing pixels in the final output.
            #       That means for positions where mask=1, keep the original.
            #       Where mask=0, we use the newly computed a_est.
            reconstructed_np = x_np.copy()
            missing_indices = np.where(m_np < 0.5)[0]
            reconstructed_np[missing_indices] = a_est[missing_indices]

            # (7) Reshape to (C, H, W) and copy to predicted_image
            reconstructed_np = reconstructed_np.reshape(C, H, W)
            predicted_image[b] = torch.from_numpy(reconstructed_np).to(
                target_image.device, dtype=target_image.dtype
            )

        return predicted_image

    @staticmethod
    def get(weights=None, **kwargs):
        """
        Optional convenience constructor if you want to mirror the pattern
        in your other inpainters.
        """
        return AMPInpainter(**kwargs)


if __name__ == "__main__":
    pass
