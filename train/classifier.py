from torch import nn
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST

import matplotlib.pyplot as plt
import os

image_size = 28
image_channels = 1
num_classes = 10

C_export_path = os.path.join(os.path.dirname(__file__), "../out/C.onnx")


def main():
    mnist_transforms = transforms.Compose([
        transforms.ToTensor(),  # [-1, 1]
        transforms.Lambda(lambda x: (x + 1.) / 2.),  # [0, 1]
    ])

    train_datasets: MNIST = datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), '..', 'data'),
        train=True,
        transform=mnist_transforms,
        download=True,
    )

    test_datasets: MNIST = datasets.MNIST(
        root=os.path.join(os.path.dirname(__file__), '..', 'data'),
        train=False,
        transform=mnist_transforms,
        download=True,
    )

    C = Classifier(image_size,
                   image_channels,
                   num_classes,
                   conv_out_channels=[32, 64],
                   conv_kernel_size=[3, 3],
                   pool_kernel_size=[2, 2])

    if not os.path.exists(os.path.dirname(C_export_path)):
        os.mkdir(os.path.dirname(C_export_path))
    C.export(C_export_path, verbose=False)


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.classifier import Classifier

    main()
