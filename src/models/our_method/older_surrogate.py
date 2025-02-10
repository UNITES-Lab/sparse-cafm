import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as resnet

NUM_HEADS = 9


class MultiHeadOlderSurrogate(nn.Module):
    """
    Predict Celano-Lab characterizations of samples.
    
    - data -> surrogate -> [9]
    - data -> swinir -> surrogate -> [9]
    
    TODO: we currently use nine of the celano lab characteristics as the output feature set
    - We have no way of knowing if this set of features is the optimal subset of all features
    - Each feature is weighted equally, where in reality we want to weight features that correspond to L1 loss
    - Is there a way to see which of our features best corresponds to L1?
    """

    def __init__(self, num_heads: int = NUM_HEADS):
        
        super(MultiHeadOlderSurrogate, self).__init__()
        self.num_heads = num_heads
        
        # ---- Resnet-152 Backbone ----
        # self.backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        
        # # replace the final layer of the backbone
        # # allows us to grab the feature representation just after
        # # global pooling is applied
        # self.backbone.fc = nn.Identity()

        # # ---- scalar value heads for each characteristic ----
        # self.heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(2048, 512),
        #         nn.ReLU(),
        #         nn.LayerNorm(512),
        #         nn.Dropout(p=0.3),
        #         nn.Linear(512, 256),
        #         nn.ReLU(),
        #         nn.LayerNorm(256),
        #         nn.Dropout(p=0.3),
        #         nn.Linear(256, 1),
        # ) for _ in range(NUM_HEADS)
        # ])
        #  -----------------------------
        
        #  ------- ViT Backbone --------
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Dropout(p=0.3),
                nn.Linear(256, 1),
            ) for _ in range(num_heads)
        ])
        # -----------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---
        :param y: current-map [H, W]

        Returns
        ---
        older_predicted_value: [1]
        """
        
        # [B, H, W] -> [B, 224, 224]
        if isinstance(self.backbone, torchvision.models.vision_transformer.VisionTransformer):
            x = torchvision.transforms.Resize((224, 224))(x)
        # [B, H, W] -> [B, 1, H, W]
        x = x.unsqueeze(1)
        # [B, 1, H, W] -> [B, 3, H, W]
        x = x.repeat(1, 3, 1, 1)
        # [B, 3, H, W] -> [B, 1000]
        x = self.backbone(x)
        # predict each characteristic
        preds = [head(x) for head in self.heads]
        out = torch.cat(preds, dim=-1)
        # [B, NUM_HEADS]
        return out

    @staticmethod
    def get(weights=None):
        return MultiHeadOlderSurrogate()


class OlderSurrogate(nn.Module):
    """
    Predict OLDER: [0, inf) from ground-truth current-maps y.
    TODO: this design is upper naive; improve.
    """

    def __init__(
        self,
    ):
        super(OlderSurrogate, self).__init__()
        self.backbone = models.resnet152()
        self.regression_head = nn.Linear(1000, 1)

    def forward(self, y_sparse: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ---
        :param y_sparse: masked current-map [H, W]
        :param y_hat: predicted current-map [H, W]

        Returns
        ---
        older_predicted_value: [1]
        """

        # [B, H, W] -> [B, 1, H, W]
        y_sparse = y_sparse.unsqueeze(1)
        # [B, H, W] -> [B, 1, H, W]
        y_hat = y_hat.unsqueeze(1)
        # [B, 1, H, W] -> [B, 2, H, W]
        y_hat = y_hat.repeat(1, 2, 1, 1)

        # TODO: verify-is this a good way to pad the channel dims?
        # [B, 2, H, W] + [B, 1, H, W] = [B, 3, H, W]
        x = torch.concat([y_sparse, y_hat], dim=1)
        # [B, 3, H, W] -> [B, 1000]
        x = self.backbone(x)
        # [B, 1000] -> [B, 1]
        x = self.regression_head(x)

        return x

    @staticmethod
    def get(weights=None):
        return OlderSurrogate()


if __name__ == "__main__":
    breakpoint()
