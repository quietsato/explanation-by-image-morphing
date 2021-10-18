import torch
from torch import nn
from typing import List


class Classifier(nn.Module):
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 num_classes: int,
                 conv_out_channels: List[int],
                 conv_kernel_size: List[int],
                 pool_kernel_size: List[int]):

        super().__init__()

        self.image_size = image_size
        self.image_channels = image_channels
        self.num_classes = num_classes

        self.model = nn.Sequential()

        n = min(len(conv_out_channels),
                len(conv_kernel_size),
                len(pool_kernel_size))

        for i in range(n):
            c_in = conv_out_channels[i-1] if i > 0 else image_channels
            c_out = conv_out_channels[i]
            ck = conv_kernel_size[i]
            pk = pool_kernel_size[i]
            image_size //= pk

            self.model.add_module(f"conv2d_{i+1}", nn.Conv2d(c_in, c_out, ck, padding=ck//2))
            self.model.add_module(f"relu_{i+1}", nn.ReLU())
            self.model.add_module(f"maxpool2d_{i+1}", nn.MaxPool2d(pk))

        linear_in = image_size * image_size * conv_out_channels[n-1]
        self.model.add_module("flatten", nn.Flatten())
        self.model.add_module("dropout", nn.Dropout(.5))
        self.model.add_module("linear", nn.Linear(linear_in, num_classes))
        self.model.add_module("softmax", nn.Softmax(dim=1))

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


def build_classifier(dataset: str = "mnist") -> Classifier:
    if dataset.lower() == "mnist":
        return Classifier(
            28,
            1,
            10,
            conv_out_channels=[32, 64],
            conv_kernel_size=[3, 3],
            pool_kernel_size=[2, 2]
        )
