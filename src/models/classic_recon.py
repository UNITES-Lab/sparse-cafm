import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import deque


class NearestNeighborsInpainter(nn.Module):
    """
    TODO: verify correctness.
    
    A simple inpainting method that fills each missing pixel with
    the color of its nearest known neighbor in terms of spatial distance.
    The method uses a BFS expansion from known pixels to fill holes.
    """

    def __init__(self, connectivity=4):
        """
        :param connectivity: 4 or 8, how many directions to expand during BFS.
                             4 => up, down, left, right
                             8 => includes diagonals
        """
        super(NearestNeighborsInpainter, self).__init__()
        self.connectivity = connectivity

    def forward(self, target_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Given a target_image and a mask, produce a 'predicted_image' that
        contains nonzero values only for the missing region (mask=0). Each
        missing pixel is filled by its closest (in a BFS sense) known neighbor.

        :param target_image: (B, C, H, W) ground truth image
        :param mask:        (B, C, H, W) boolean or {0,1}
                            1 => known region, 0 => missing region
        :return predicted_image: (B, C, H, W), where
                                 * For mask=1 (known region), predicted_image=0
                                 * For mask=0 (missing region), predicted_image
                                   is filled by the nearest known neighbor.
        """
        # We'll ensure the mask is binary boolean so that we can do BFS checks easily.
        # If mask is {0,1} in integer form, the comparison below becomes boolean.
        mask_bool = mask > 0  # shape: B, C, H, W
        
        masked_image = target_image * mask

        B, C, H, W = target_image.shape

        # We'll create a working copy that we can fill in fully.
        # This is the "complete" inpainted image from BFS, containing
        # the nearest known-pixel color for every pixel (both known and missing).
        filled_image = target_image.clone()

        # We also track which pixels we have visited in BFS.
        visited = mask_bool.clone()  # True at known pixels, False at missing

        # Choose neighbor offsets based on connectivity
        if self.connectivity == 4:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:  # 8-connectivity
            directions = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]

        # Process one image at a time in the batch
        for b_idx in range(B):
            # Gather all initially-known pixels into a BFS queue
            queue = deque()
            # We'll look at mask_bool in the first channel (or across all channels) to see if a pixel is known.
            # Often, the mask is the same across channels. If your mask is distinct per channel,
            # you might adapt the check below.
            known_locs = mask_bool[b_idx, 0] == True  # shape: (H, W)

            # Enqueue each known pixel
            known_coords = known_locs.nonzero(
                as_tuple=False
            )  # shape: (#known_pixels, 2) => [y, x]
            for yx in known_coords:
                y, x = yx
                queue.append((x.item(), y.item()))

            # BFS expansion
            while queue:
                x_cur, y_cur = queue.popleft()

                for dx, dy in directions:
                    nx = x_cur + dx
                    ny = y_cur + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        if not visited[b_idx, 0, ny, nx]:
                            # Assign the color from (y_cur, x_cur) to this neighbor
                            filled_image[b_idx, :, ny, nx] = filled_image[
                                b_idx, :, y_cur, x_cur
                            ]
                            visited[b_idx, :, ny, nx] = True
                            queue.append((nx, ny))

        # We only want to output nonzero values in the missing region (where mask=0).
        # The BFS above fills the entire image (including known region). But the spec
        # says that for known pixels, we should return 0 in 'predicted_image'. So we
        # multiply by (1 - mask).
        # If mask is float/binary, (1 - mask) highlights the holes.
        predicted_image = (filled_image * ~mask) + masked_image
        return predicted_image

    @staticmethod
    def get(connectivity=4):
        return NearestNeighborsInpainter(connectivity=connectivity)


class LinearInterpolationInpainter(nn.Module):
    """
    TODO: verify correctness.
    """

    def __init__(self):
        super(LinearInterpolationInpainter, self).__init__()

    def forward(self, target_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
    Simple linear inpainter for data shaped (B, H, W).
    It performs a 3x3 average over known neighbors and
    fills in the missing pixels.
    """

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
            1, 1, 3, 3, 
            device=target_image.device, 
            dtype=target_image.dtype
        )
        kernel = kernel / kernel.sum()  # Each element 1/9

        # 3) Use conv2d with an extra channel dimension to sum known neighbors.
        #    The result shape after conv2d is (B, 1, H, W).
        sum_of_neighbors = F.conv2d(
            masked_image.unsqueeze(1),  # (B, 1, H, W)
            kernel,
            padding=1
        )

        # 4) Convolve the mask (converted to float) likewise
        #    to determine how many neighbors contributed.
        sum_of_masks = F.conv2d(
            mask.float().unsqueeze(1),  # (B, 1, H, W)
            kernel,
            padding=1
        )

        # 5) Compute average of neighbors. Add small eps to avoid div-by-zero.
        eps = 1e-8
        interpolated_values = sum_of_neighbors / (sum_of_masks + eps)  # (B, 1, H, W)

        # 6) We only want these interpolated values where mask=0.
        #    We'll multiply by ~mask to keep them only in the missing region.
        #    Then add the known pixels (which remain zeroed in predicted_image).
        predicted_image = interpolated_values * (~mask).unsqueeze(1) + masked_image.unsqueeze(1)
        
        # Squeeze out the extra channel dimension to return shape (B, H, W).
        return predicted_image.squeeze(1)

    @staticmethod
    def get(weights=None):
        return LinearInterpolationInpainter()


class BicubicInterpolationInpainter(nn.Module):
    """
    TODO: verify correctness.
    
    Inpaint missing regions (mask=0) by:
      1) Zeroing out the missing region in 'target_image'.
      2) Downsampling that masked image.
      3) Upsampling it back with bicubic interpolation.
      4) Filling in the missing region from the upsampled result.

    This is a simple demonstration of bicubic-based filling using PyTorch.
    """

    def __init__(self, downsample_scale=0.5):
        """
        :param downsample_scale: factor by which we reduce the resolution
                                 before upsampling back.
                                 E.g., 0.5 => downsample to half-size, then
                                 bicubic upsample to original size.
        """
        super(BicubicInterpolationInpainter, self).__init__()
        self.downsample_scale = downsample_scale

    def forward(self, target_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param target_image: (B, C, H, W) ground truth image
        :param mask: (B, C, H, W) boolean or {0,1}:
                     1 => known region, 0 => missing region
        :return predicted_image: (B, C, H, W), where the missing region is
                                 filled via naive bicubic interpolation.
        """
        # 1) Zero out the missing region in the image so we only keep known pixels.
        #    The masked_image has 0 where mask=0, original pixel where mask=1.
        masked_image = target_image * mask  # shape: (B, C, H, W)
        B, C, H, W = masked_image.shape
        # 2) Downsample the masked image to a smaller size. We choose 'area' mode
        #    (which averages pixels) for the downsampling step.
        #    The new height/width are determined by downsample_scale.
        #    For example, if downsample_scale=0.5, we go to (H//2, W//2).
        downsampled = F.interpolate(
            masked_image,
            scale_factor=self.downsample_scale,
            mode="area",  # area = "average pooling" style downsampling
        )
        # 3) Upsample the downsampled tensor back to the original size,
        #    this time using bicubic interpolation.
        #    align_corners=False is typically recommended for scaling tasks.
        bicubic_upsampled = F.interpolate(
            downsampled, size=(H, W), mode="bicubic", align_corners=False
        )
        # 4) Combine. We only want the upsampled values in the missing region
        #    (where mask=0). Everywhere else, we reuse the original (masked_image).
        #    If 'mask' is boolean, '~mask' is True in the missing region.
        #    If mask is integer {0, 1}, do (1 - mask) instead.
        #    We'll also keep the known pixels from the original "masked_image"
        #    so they remain unaltered.
        missing_region = (~mask) if mask.dtype == torch.bool else (1 - mask)
        predicted_image = bicubic_upsampled * missing_region + masked_image

        return predicted_image

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
    model = LinearInterpolationInpainter()
    img = torch.rand(1, 3, 224, 224)
    mask = torch.ones(1, 3, 224, 224).bool()
    out = model(img, mask)
