import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.models.resnet as resnet

from torchvision.models import VisionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

NUM_HEADS = 4


class TransformerBlock(nn.Module):
    """
    A simple transformer block that applies multi-head self-attention to a tokenized representation.
    Assumes input features can be reshaped into (num_tokens, embed_dim) with num_tokens * embed_dim = input_dim.
    """
    def __init__(self, num_tokens=16, embed_dim=64, nhead=8, dropout=0.3):
        super(TransformerBlock, self).__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, activation='relu'
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        # Reshape to (batch_size, num_tokens, embed_dim)
        x = x.view(batch_size, self.num_tokens, self.embed_dim)
        # nn.TransformerEncoderLayer expects (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)  # Now (num_tokens, batch_size, embed_dim)
        x = self.transformer_layer(x)  # Apply self-attention
        # Transpose back and flatten to (batch_size, input_dim)
        x = x.transpose(0, 1).contiguous().view(batch_size, -1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, features, hidden_features, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(features, hidden_features)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_features)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_features, features)
        self.norm2 = nn.LayerNorm(features)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.norm1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        out = out + identity
        out = self.relu(out)
        return out


class MultiHeadOLDERSurrogate(nn.Module):
    """
    Predict Celano-Lab characterizations of samples.
    
    - [H, W] -> surrogate -> [9]
    - [H, W] -> swinir -> surrogate -> [9]
    
    TODO:
    - Perform a rigorous study of feature importance;
    - Determine which features are most positively correlated with L1, negatively, spuriously
    """

    def __init__(self, num_heads: int = NUM_HEADS):
        
        super(MultiHeadOLDERSurrogate, self).__init__()
        self.num_heads = num_heads
        
        # ---- VGG-19 Feature Extractor ----
        self.backbone = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        
        # remove the final layer
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])

        # ---- classification heads ----
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4096, 512),
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

        # self.heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(4096, 1024),
        #         nn.ReLU(),
        #         nn.LayerNorm(1024),
        #         ResidualBlock(features=1024, hidden_features=512, dropout=0.3),
        #         ResidualBlock(features=1024, hidden_features=512, dropout=0.3),
        #         nn.Linear(1024, 256),
        #         nn.ReLU(),
        #         nn.LayerNorm(256),
        #         ResidualBlock(features=256, hidden_features=128, dropout=0.3),
        #         ResidualBlock(features=256, hidden_features=128, dropout=0.3),
        #         nn.Linear(256, 1)
        #     ) for _ in range(num_heads)
        # ])

        # self.heads = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(4096, 1024),
        #         nn.ReLU(),
        #         nn.LayerNorm(1024),
        #         TransformerBlock(num_tokens=16, embed_dim=64, nhead=8, dropout=0.3),
        #         ResidualBlock(features=1024, hidden_features=512, dropout=0.3),
        #         ResidualBlock(features=1024, hidden_features=512, dropout=0.3),
        #         nn.Linear(1024, 1024),
        #         nn.ReLU(),
        #         nn.LayerNorm(1024),
        #         TransformerBlock(num_tokens=16, embed_dim=64, nhead=8, dropout=0.3),
        #         ResidualBlock(features=1024, hidden_features=512, dropout=0.3),
        #         ResidualBlock(features=1024, hidden_features=512, dropout=0.3),
        #         nn.Linear(1024, 1024),
        #         nn.ReLU(),
        #         nn.LayerNorm(1024),
        #         TransformerBlock(num_tokens=16, embed_dim=64, nhead=8, dropout=0.3),
        #         ResidualBlock(features=1024, hidden_features=512, dropout=0.3),
        #         ResidualBlock(features=1024, hidden_features=512, dropout=0.3),
        #         nn.Linear(1024, 256),
        #         nn.ReLU(),
        #         nn.LayerNorm(256),
        #         ResidualBlock(features=256, hidden_features=128, dropout=0.3),
        #         ResidualBlock(features=256, hidden_features=128, dropout=0.3),
        #         nn.Linear(256, 1)
        #     ) for _ in range(num_heads)
        # ])

        # # ------------------------------

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
    def get(weights=None): return MultiHeadOLDERSurrogate()

if __name__ == "__main__": pass