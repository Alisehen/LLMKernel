import torch
import torch.nn as nn


class Model(nn.Module):
    """
    ConvTranspose2d + GlobalAvgPool + BiasAdd + LogSumExp + Sum + Multiply
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        x = x + self.bias
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = torch.sum(x, dim=(2, 3))
        x = x * 10.0
        return x


def get_inputs():
    return [torch.randn(16, 64, 32, 32).cuda()]


def get_init_inputs():
    return [64, 128, 3, (128, 1, 1)]
