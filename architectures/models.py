import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .base import Conv2dBnRelu


class STNet(nn.Module):
    def __init__(self, activation, width=64, height=64):
        super().__init__()
        self.activation = activation
        self.width = width
        self.height = height
        self.conv1 = Conv2dBnRelu(input_channels=1, output_channels=8, activation=activation)
        self.conv2 = Conv2dBnRelu(input_channels=8, output_channels=16, activation=activation)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv3 = Conv2dBnRelu(input_channels=16, output_channels=16, activation=activation)
        self.conv4 = Conv2dBnRelu(input_channels=16, output_channels=32, activation=activation)
        self.max_pool2 = nn.MaxPool2d(2)

        flatten_size = 8192

        self.lin1 = nn.Linear(flatten_size, flatten_size // 4)
        self.bn1 = nn.BatchNorm1d(flatten_size // 4)
        self.lin2 = nn.Linear(flatten_size // 4, 6)
        self.bn2 = nn.BatchNorm1d(6)


    def forward(self, x):
        x_old = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1).contiguous()
        x = F.relu(self.bn1(self.lin1(x)))
        x = self.lin2(x)

        x = x.view(-1, 2, 3)
        grid = F.affine_grid(x, [x.size(0),
                                 1,
                                 self.height,
                                 self.width])
        x = F.grid_sample(x_old, grid)
        return x, grid


class LeNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
        self.stn_module = STNet(activation=activation)
        self.conv1 = Conv2dBnRelu(input_channels=1, output_channels=8, activation=activation)
        self.conv2 = Conv2dBnRelu(input_channels=8, output_channels=16, activation=activation)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv3 = Conv2dBnRelu(input_channels=16, output_channels=16, activation=activation)
        self.conv4 = Conv2dBnRelu(input_channels=16, output_channels=32, activation=activation)
        self.max_pool2 = nn.MaxPool2d(2)

        flatten_size = 8192


        self.lin1 = nn.Linear(flatten_size, flatten_size // 4)
        self.bn1 = nn.BatchNorm1d(flatten_size // 4)
        self.lin2 = nn.Linear(flatten_size // 4, 10)
        self.bn2 = nn.BatchNorm1d(10)

    def plot_res(self, x, path):
        fig = plt.figure(figsize=(30, 30))
        for i in range(100):
            plt.subplot(10, 10, i+1)
            plt.imshow((x.data[i,:,:,:] \
                         .unsqueeze(1) \
                         .expand(1, 3, 64, 64) \
                         .permute(0, 2, 3, 1) \
                         .numpy()[0,:,:,:]) \
                         .astype(np.float32))
        plt.tight_layout()
        plt.savefig(path)

    def forward(self, x):

        self.plot_res(x, 'x.png')

        x, grid = self.stn_module(x)

        self.plot_res(x, 'stn.png')

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1).contiguous()
        x = F.relu(self.bn1(self.lin1(x)))
        x = self.lin2(x)

        return x
