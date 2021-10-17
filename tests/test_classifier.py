import unittest
import torch


class ClassifierTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.batch_size = 8
        self.image_size = 28
        self.image_channels = 1
        self.num_classes = 10

    def test_forward(self):
        model = Classifier(
            self.image_size,
            self.image_channels,
            self.num_classes,
            conv_out_channels=[4, 8],
            conv_kernel_size=[4, 4],
            pool_kernel_size=[2, 2]
        )

        x = torch.zeros([
            self.batch_size,
            self.image_channels,
            self.image_size,
            self.image_size])

        y = model.forward(x)

        self.assertEqual(list(y.shape), [self.batch_size, self.num_classes])


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from models.classifier import *

    unittest.main()
