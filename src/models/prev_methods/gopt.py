import torch
from microstructure_inpainter.micro_config import Config
from microstructure_inpainter.networks import make_nets


class GOpt(torch.nn.Module):
    """
    An implementation of Generator Optimization (G-opt) from: https://github.com/tldr-group/microstructure-inpainter.
    """

    def __init__(self):
        super(GOpt, self).__init__()
        self.config = Config("")
        dis, gen = make_nets(self.config)
        self.discriminator = dis
        self.generator = gen

    def forward(self, y: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        pass
    
if __name__ == "__main__":
    GOpt()
