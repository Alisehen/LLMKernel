#!/usr/bin/env python3
"""Test script to discover available NCU metrics on RTX 4090"""

import torch
import torch.nn as nn

class SimpleKernel(nn.Module):
    def forward(self, x):
        return x * 2 + 1

if __name__ == "__main__":
    model = SimpleKernel().cuda().eval()
    x = torch.randn(1024, 1024).cuda()

    # Warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = model(x)
    torch.cuda.synchronize()

    # Run for profiling
    with torch.inference_mode():
        for _ in range(10):
            out = model(x)
    torch.cuda.synchronize()

    print("Test completed successfully")
