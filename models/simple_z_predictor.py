import torch
from torch.nn.modules import Transformer, TransformerEncoder
from torchvision.models import vit_b_16, VisionTransformer

H_DIM = 512
H_DIM_VT = 768


class SimpleZRegressionTransformer(torch.nn.Module):

    def __init__(self) -> None:
        super(SimpleZRegressionTransformer, self).__init__()
        self.transformer = Transformer()
        self.encoder: TransformerEncoder = self.transformer.encoder
        self.regression_head = torch.nn.modules.Linear(H_DIM, 1)

    def forward(self, x):
        
        breakpoint()
        x = self.encoder(x)
        x = self.regression_head(x)
        x = torch.nn.functional.sigmoid(x)
        return x


class SimpleZRegressionVisionTransformer(torch.nn.Module):

    def __init__(self):
        super(SimpleZRegressionVisionTransformer, self).__init__()
        self.vit = vit_b_16()
        self.encoder: TransformerEncoder = self.vit.encoder
        self.regression_head = torch.nn.modules.Linear(H_DIM_VT, 1)

    def forward_vt(self, x):
        r"""

        Shape:
            - x: :math:`(N, C, H, W)` where `H = W = 224` by default.
        """

        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = self.regression_head(x)
        x = torch.nn.functional.sigmoid(x)
        return x


if __name__ == "__main__":
    m = SimpleZRegressionTransformer()
    x = torch.rand((10, 3, 512))
    m(x)
