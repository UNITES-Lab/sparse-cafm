import torch
import torch.nn as nn
import torchvision.models as models

class OlderSurrogate(nn.Module):
    """
    Predict OLDER: [0, inf) from ground-truth current-maps y.
    """
    def __init__(self, ): 
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
    
if __name__ == '__main__':
    model = OlderSurrogate()
    y = torch.rand((1, 128, 128))
    model(y)