import torch.nn as nn
import torch.nn.functional as F

from .base import Conv2dBnRelu


class LeNet(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
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


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2(x)
        x = x.view(x.size(0), -1).contiguous()
        x = F.relu(self.bn1(self.lin1(x)))
        x = self.bn2(self.lin2(x))
        return x
