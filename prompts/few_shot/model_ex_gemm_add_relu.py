import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Fusion example: Gemm + Add + ReLU
    Three operations fused together.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = x + self.bias
        x = torch.relu(x)
        return x


def get_inputs():
    return [torch.randn(128, 512).cuda()]


def get_init_inputs():
    return [512, 512, (512,)]
