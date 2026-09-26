"""Свёрточная сеть из первого задания."""

from torch import nn


class HomeworkCNN(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(128, 256, kernel_size=1, padding=0, bias=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(256, 512, kernel_size=1, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100, bias=False),
        )
