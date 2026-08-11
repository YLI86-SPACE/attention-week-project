from collections import OrderedDict

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classification_head = nn.Linear(256, num_classes)

    def forward_features(self, x: torch.Tensor, return_intermediates: bool = False):
        intermediates = OrderedDict()
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            intermediates[f"block{i}"] = x
        feature = self.pool(x).flatten(1)
        if return_intermediates:
            return feature, intermediates
        return feature

    def forward(self, x: torch.Tensor):
        feature = self.forward_features(x)
        return self.classification_head(feature)
