import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST

import os

mnist_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x + 1.) / 2.),
])

train_dataset: MNIST = datasets.MNIST(
    root=os.path.join(os.path.dirname(__file__), '..', 'data'),
    train=True,
    transform=mnist_transforms,
    download=True,
)
