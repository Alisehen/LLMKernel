
import torch

class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + y

def get_inputs():
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    return [x, y]
