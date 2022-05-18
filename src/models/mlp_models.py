"""Python file with MLP model classes."""

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
