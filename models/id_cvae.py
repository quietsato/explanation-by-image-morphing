from typing import List, Tuple
import torch
from torch import Tensor, nn
from math import ceil


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
            self.seq.add_module(f"conv2d_{i+1}",
                                nn.Conv2d(c_in, c_out, ck, cs, padding=(ck - cs + 1)//2))
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
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 latent_dim: int,
                 num_classes: int,
                 conv_t_in_channels: List[int],
                 conv_t_kernel_size: List[int],
                 conv_t_stride: List[int]):
        super().__init__()
        self.num_classes = num_classes

        n = min(len(conv_t_in_channels), len(conv_t_kernel_size), len(conv_t_stride))

        self.image_channels_start = conv_t_in_channels[0]
        self.image_size_start = image_size
        for s in conv_t_stride:
            self.image_size_start //= s

        linear_out = self.image_size_start * self.image_size_start * conv_t_in_channels[0]
        self.W_start = nn.Linear(latent_dim + num_classes, linear_out)

        self.seq = nn.Sequential()
        image_size = self.image_size_start
        for i in range(n):
            c_in = conv_t_in_channels[i]
            c_out = conv_t_in_channels[i+1] if i < n - 1 else image_channels
            ck = conv_t_kernel_size[i]
            cs = conv_t_stride[i]
            self.seq.add_module(f"conv_t_{i+1}",
                                nn.ConvTranspose2d(c_in, c_out, ck, cs,
                                                   padding=ceil(ck-cs+1)//2,
                                                   output_padding=(ck-cs) % 2))
            if i < n - 1:
                self.seq.add_module(f"relu", nn.ReLU())
            else:
                self.seq.add_module(f"sigmoid", nn.Sigmoid())

    def forward(self, z: Tensor, y: Tensor):
        y_onehot = torch.eye(self.num_classes, dtype=torch.float32)[y]
        zy = torch.cat([z, y_onehot], 1)
        f: Tensor = self.W_start(zy)
        f = f.reshape([len(f),
                       self.image_channels_start,
                       self.image_size_start,
                       self.image_size_start])
        x = self.seq(f)
        return x


class ID_CVAE(torch.nn.Module):
    def __init__(self,
                 image_size: int,
                 image_channels: int,
                 latent_dim: int,
                 num_classes: int,
                 conv_out_channels: List[int],
                 conv_kernel_size: List[int],
                 conv_stride: List[int],
                 conv_t_in_channels: List[int],
                 conv_t_kernel_size: List[int],
                 conv_t_stride: List[int]
                 ):
        super().__init__()

        self.E = Encoder(
            image_size,
            image_channels,
            latent_dim,
            conv_out_channels,
            conv_kernel_size,
            conv_stride
        )
        self.D = Decoder(
            image_size,
            image_channels,
            latent_dim,
            num_classes,
            conv_t_in_channels,
            conv_t_kernel_size,
            conv_t_stride
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.E(x)

    def decode(self, z: Tensor, y: Tensor) -> Tensor:
        return self.D(z, y)

    def forward(self, x: Tensor, y: Tensor):
        z, _, _ = self.E(x)
        x_ = self.D(z, y)
        return x_
