import unittest
import torch
from torch import Tensor


class ID_CVAETest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.batch_size = 8
        self.image_size = 28
        self.image_channels = 1
        self.num_classes = 10
        self.latent_dim = 2

    def test_encoder(self):
        E = Encoder(
            self.image_size,
            self.image_channels,
            self.latent_dim,
            conv_out_channels=[4, 8, 8],
            conv_kernel_size=[4, 4, 3],
            conv_stride=[1, 2, 2]
        )

        x = torch.randn([
            self.batch_size,
            self.image_channels,
            self.image_size,
            self.image_size])

        z, z_mean, z_log_var = E(x)

        self.assertEqual(list(z.shape), [self.batch_size, self.latent_dim])
        self.assertEqual(list(z_mean.shape), [self.batch_size, self.latent_dim])
        self.assertEqual(list(z_log_var.shape), [self.batch_size, self.latent_dim])

    def test_decoder(self):
        D = Decoder(
            self.image_size,
            self.image_channels,
            self.latent_dim,
            self.num_classes,
            conv_t_in_channels=[8, 4, 4],
            conv_t_kernel_size=[4, 3, 3],
            conv_t_stride=[2, 2, 1]
        )

        z = torch.zeros([
            self.batch_size,
            self.latent_dim
        ])

        y = torch.zeros([
            self.batch_size,
            self.num_classes
        ])

        x: Tensor = D(z, y)

        self.assertEqual(list(x.shape),
                         [self.batch_size, self.image_channels, self.image_size, self.image_size])


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.id_cvae import *

    unittest.main()
