import torch
from torch._C import dtype
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from utils.logger import get_time_str
from models.classifier import build_classifier
from models.id_cvae import build_id_cvae

import matplotlib.pyplot as plt
import imageio
import os


C_STATE_DICT_EPOCH = 10
VAE_STATE_DICT_EPOCH = 4

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
SCRIPT_OUT_DIR = os.path.join(OUT_DIR, "main")


def main():
    if not os.path.exists(SCRIPT_OUT_DIR):
        os.makedirs(SCRIPT_OUT_DIR)

    C = build_classifier("mnist")
    VAE = build_id_cvae("mnist")
    for param in C.parameters():
        param.requires_grad = False
    for param in VAE.parameters():
        param.requires_grad = False

    c_state_dict_path = os.path.join(OUT_DIR, "C", f"{C_STATE_DICT_EPOCH:03}_state_dict")
    vae_state_dict_path = os.path.join(OUT_DIR, "VAE", f"{VAE_STATE_DICT_EPOCH:03}_state_dict")

    C.load_state_dict(torch.load(c_state_dict_path))
    VAE.load_state_dict(torch.load(vae_state_dict_path))

    test_dataset = MNIST(DATA_DIR, train=False, transform=ToTensor(), download=True)
    dataset = list(test_dataset)

    encode_decode_test(dataset[:10], VAE)


def encode_decode_test(dataset, VAE):
    n = len(dataset)
    plt.figure(figsize=(2, n))
    for i, (x, y) in enumerate(dataset):
        plt.subplot(n, 2, 2*i+1)
        plt.imshow(x[0], cmap='gray')
        plt.axis('off')

        plt.subplot(n, 2, 2*i+2)
        x_rec = VAE(
            x.reshape(1, 1, 28, 28),
            torch.tensor(y).long().reshape(1,)
        )
        plt.imshow(x_rec[0, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(SCRIPT_OUT_DIR, f"{get_time_str()}_encode_decode.png"))


if __name__ == "__main__":
    main()
