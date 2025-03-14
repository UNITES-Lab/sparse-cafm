import torch
import hiera.hiera as H
from models.unet.unet import HieraUNetDecoder

x = torch.rand(1, 3, 224, 224)
model = H.mae_hiera_base_224(pretrained=True, checkpoint="mae_in1k")
model(x)
breakpoint()
print("hey!")