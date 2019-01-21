import torch
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from .constants import MNIST_PATH, MNIST_MEAN, MNIST_STD

input_size = 64
batch_size = 128

class RandomPos():
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height

    def __call__(self, data):
        tensor = torch.zeros((data.size(0), self.width, self.height))
        for i in range(data.size(0)):
            x = np.random.randint(0, self.width - 28)
            y = np.random.randint(0, self.height - 28)
            tensor[i, x:x + 28, y:y + 28] = data

        return tensor


transform = transforms.Compose([transforms.ToTensor(),
                                RandomPos(input_size, input_size),
                                transforms.Normalize((MNIST_MEAN[input_size], ),
                                                     (MNIST_STD[input_size], )
                                                     )])

train_set = MNIST(MNIST_PATH, train=True, transform=transform, download=True)
test_set = MNIST(MNIST_PATH, train=False, transform=transform, download=True)

mnist_loader = {'train': torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True),
                'test': torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          shuffle=False)}