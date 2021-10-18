from typing import List
import torch
from torch import Tensor, nn


class Encoder(torch.nn.Module):
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 latent_dim: int,
                 conv_out_channels: List[int],
                 conv_kernel_size: List[int],
                 conv_stride: List[int]):
        super().__init__()

        self.seq = nn.Sequential()
        n = min(len(conv_out_channels), len(conv_kernel_size), len(conv_stride))
        for i in range(n):
            c_in = conv_out_channels[i-1] if i > 0 else image_channels
            c_out = conv_out_channels[i]
            ck = conv_kernel_size[i]
            cs = conv_stride[i]
            image_size //= cs
            self.seq.add_module(f"conv2d_{i+1}", nn.Conv2d(c_in, c_out, ck, cs, padding=(ck-1)//2))
            self.seq.add_module(f"relu", nn.ReLU())
        linear_in = image_size * image_size * conv_out_channels[n-1]
        self.seq.add_module(f"flatten", nn.Flatten())

        self.W_z_mean = nn.Linear(linear_in, latent_dim)
        self.W_log_var = nn.Linear(linear_in, latent_dim)

    def forward(self, x: Tensor):
        f = self.seq(x)

        z_mean: Tensor = self.W_z_mean(f)
        z_log_var: Tensor = self.W_log_var(f)
        epsilon = torch.randn(list(z_mean.size())[:2])

        z = z_mean + torch.exp(.5 * z_log_var) * epsilon

        return z, z_mean, z_log_var


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: Tensor, y: Tensor):
        pass


class ID_CVAE(torch.nn.Module):
    def __init__(self, E: Encoder, D: Decoder):
        super().__init__()

        self.E = E
        self.D = D

    def forward(self, x: Tensor, y: Tensor):
        z: Tensor = self.E(x)[0]
        x_ = self.D(z, y)
        return x_
