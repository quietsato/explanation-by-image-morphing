#%%
from posixpath import basename, dirname
from torch import nn, Tensor
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST
from torchvision.transforms.transforms import Normalize

import matplotlib.pyplot as plt
import os

#%%
mnist: MNIST = datasets.MNIST(
    root=os.path.join(dirname(__file__), '..', 'data'),
    train=True,
    transform=[transforms.ToTensor()],
    download=True,
)

#%%
print(mnist.data.shape)
plt.imshow(mnist.data[0], cmap='gray')
plt.axis(False)

# %%
