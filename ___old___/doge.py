import torch
import torch.nn as nn
import torchvision.models as models


class DoGE(nn.Module):
    """
    A Domain-Guided Encoder; returns multi-level features.
    """
    def __init__(self):
        super(DoGE, self).__init__()
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        self.features = vgg.features                                     # full feature extractor
        self.activations = []                                            # to store activations from hooks
        self.hooks = []                                                  # keep references to hooks to avoid garbage collection

        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.ReLU):
                hook = layer.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        self.activations.append(output)

    def forward(self, x: torch.Tensor) -> list:
        self.activations = []
        # [B, H, W] -> [B, 1, H, W]
        x = x.unsqueeze(1)
        # [B, 1, H, W] -> [B, 3, H, W]
        x = x.repeat(1, 3, 1, 1)
        _ = self.features(x)
        return self.activations

if __name__ == "__main__": pass