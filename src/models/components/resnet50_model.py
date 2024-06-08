import cv2
from matplotlib import pyplot as plt
import albumentations as A
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class Resnet50(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

        # self.resnet50 = resnet50()
        self.resnet50 = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        in_feature = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_feature, 14 * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet50(x)
    




if __name__ == "__main__":
    model = Resnet50()
    input = torch.zeros((64, 3, 128, 128))
    print(input.shape)
    output = model(input)
    print(output.shape)
