
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=32):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels, width, 3, 3)
        self.conv_b = nn.Conv2d(width, out_channels, 3, 3)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv_a(x))
        x = self.sig(self.conv_b(x))
        return x.flatten()
