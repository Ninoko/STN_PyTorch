import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBnRelu(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3),
                 use_padding=True, use_batch_norm=True, activation=None):
        super(Conv2dBnRelu, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = nn.ZeroPad2d(padding=(0,
                                             self.kernel_size[0] - 1,
                                             self.kernel_size[1] - 1,
                                             0)) if use_padding else None
        self.batch_norm = nn.BatchNorm2d(output_channels) if use_batch_norm else None
        self.activation = activation

    def forward(self, x):
        if self.padding:
            x = self.padding(x)
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    

class ResidaulBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=(3, 3),
                 use_padding=True, use_batch_norm=True, activation=None):
        super(ResidaulBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.use_padding = use_padding
        self.use_batch_nrom = use_batch_norm
        self.activation = activation

        self.conv = nn.Sequential(Conv2dBnRelu(input_channels=input_channels,
                                               output_channels=output_channels - input_channels,
                                               kernel_size=kernel_size,
                                               use_padding=use_padding,
                                               use_batch_norm=use_batch_norm,
                                               activation=activation),
                                  Conv2dBnRelu(input_channels=output_channels - input_channels,
                                               output_channels=output_channels - input_channels,
                                               kernel_size=kernel_size,
                                               use_padding=use_padding,
                                               use_batch_norm=use_batch_norm,
                                               activation=None))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        return self.relu(torch.cat([conv, x], 1))
