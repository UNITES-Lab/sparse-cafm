import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    """
    Custom classification head used for predicting the final output value z.
    """
    def __init__(self, in_channels):
        super(RegressionHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1)
    def forward(self, x):
        return self.fc1(x)

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
in_features = model.fc.in_features
# change classification to have size 1
model.fc = RegressionHead(in_features)

print(model)