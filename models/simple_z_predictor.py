import torch

from typing import Optional
from torch.nn.modules import Transformer, TransformerEncoder
from torchvision.models import vit_b_16, VisionTransformer

H_DIM = 512
H_DIM_VT = 768
N_OUTPUT_TOKENS_VT = 16 * 197


class SimpleZRegressionTransformer(torch.nn.Module):

    def __init__(self) -> None:
        super(SimpleZRegressionTransformer, self).__init__()
        self.transformer = Transformer()
        self.encoder: TransformerEncoder = self.transformer.encoder
        self.regression_head = torch.nn.modules.Linear(H_DIM, 1)

    def forward(self, x):
        r"""
        Shape:
            - x: :math:`(N, S, E)` where `E = 512` by default.
        """

        x = self.encoder(x)
        x = self.regression_head(x)
        x = torch.nn.functional.sigmoid(x)
        return x


class SimpleZRegressionVisionTransformer(torch.nn.Module):

    def __init__(self):
        super(SimpleZRegressionVisionTransformer, self).__init__()
        self.vit = vit_b_16()
        self.encoder: TransformerEncoder = self.vit.encoder
        self.regression_head: Optional[torch.nn.Linear] = None

    def forward(self, x):
        r"""

        Shape:
            - x: :math:`(N, C, H, W)` where `H = W = 224` by default.
        """
        
        # (16, 3, 224, 224)
        # (16, 196, 768)
        x = self.vit._process_input(x)
        n = x.shape[0]
        
        # (16, 1, 768)
        batch_class_token = self.vit.class_token.expand(n, -1, -1)  

        # (16, 197, 768)
        x = torch.cat([batch_class_token, x], dim=1)  
        
        # (16 * 197, 768)
        x: torch.Tensor = self.encoder(x)
        
        # (16 * 197, 1)
        x = x.view((x.shape[0], x.shape[1] * x.shape[2]))

        if self.regression_head == None:
            self.regression_head = torch.nn.Linear(
                in_features=x.shape[1], out_features=1
            ).cuda()
        
        # (16, 1)
        x = self.regression_head(x)
        x = torch.nn.functional.sigmoid(x)
        return x

    @staticmethod
    def get(weights=None):
        return SimpleZRegressionVisionTransformer()


if __name__ == "__main__":
    m = SimpleZRegressionTransformer()
    x = torch.rand((10, 3, 512))
    m(x)
