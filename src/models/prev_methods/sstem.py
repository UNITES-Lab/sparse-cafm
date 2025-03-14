"""
Inspired by: https://onlinelibrary.wiley.com/doi/pdf/10.1002/smll.202002878?casa_token=OP1n_oLqe4kAAAAA%3Aiovq39gdeNfEIR8Vyi_FRd3Ec9lz8cDm3m9MtmCoOXbg6w1ohs5YPom5x9uVK9S3wsqmssIPFzfsCIBM9w
"""

import pywt
import torch


class SSTEM(torch.nn.Module):
    """
    Implemented from: https://github.com/ziatdinovmax/Notebooks-for-papers/blob/master/GP_spiral_scans_GP.ipynb
    """

    def __init__(
        self,
    ):
        super(SSTEM, self).__init__()

    @staticmethod
    def SSTEM(
        yspar: torch.Tensor,
        mask: torch.Tensor,
        itern: int = 20,
        levels: int = 2,
        lambd: float = 0.8,
    ) -> torch.Tensor:
        """
        Parameters
        ---
        yspar: sparse image as an array, to reduce iteration numer, better rescaling the value to [0,1]
        mask: binary array, 1 indicationg sampled pixel locations
        itern: iteration number, usually 20 is enough
        levels: wavelet level, common choice 2,3,4, larger value for larger feature size, if too blur, change to smaller one
        lambd: threshold value, usually 0.8 is fine
        Output: Inpaited image
        """

        # bool -> float
        mask = mask.float()

        # -> numpy()
        yspar = yspar.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        fSpars = yspar
        W_thr = [0] * levels

        ProjC = lambda f, Omega: (1 - Omega) * f + Omega * yspar

        for i in range(itern):
            fSpars = ProjC(fSpars, mask)
            W_pro = pywt.swt2(fSpars, "db2", levels)
            for j in range(levels):
                sA = W_pro[j][0]
                sH = W_pro[j][1][0]
                sV = W_pro[j][1][1]
                sD = W_pro[j][1][2]
                W_thr[j] = (pywt.threshold(sA, 0, "soft")), (
                    pywt.threshold(sH, lambd, "soft"),
                    pywt.threshold(sV, lambd, "soft"),
                    pywt.threshold(sD, lambd, "soft"),
                )
            fSpars = pywt.iswt2(W_thr, "db2")
        return fSpars

    def forward(self, y_sparse: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        x = SSTEM.SSTEM(y_sparse, y_mask)
        x = torch.Tensor(x).cuda()
        return x

    @staticmethod
    def get(weights=None):
        return SSTEM()


if __name__ == "__main__":
    sstem = SSTEM()
