#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple CUDA extension smoke test for Nsight Compute profiling."""

import torch
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void simple_add_kernel(const float* x,
                                  const float* y,
                                  float* out,
                                  int64_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}

torch::Tensor simple_add_cuda(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(y.scalar_type() == at::kFloat, "y must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.numel() == y.numel(), "x and y must have the same number of elements");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    simple_add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "simple_add_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
"""

cpp_src = "torch::Tensor simple_add_cuda(torch::Tensor x, torch::Tensor y);"

simple_add_module = load_inline(
    name="simple_add_cuda_test",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["simple_add_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


def test_cuda_kernel() -> torch.Tensor:
    size = 1024
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    out = simple_add_module.simple_add_cuda(x, y)

    expected = x + y
    assert torch.allclose(out, expected), "Kernel output mismatch"
    print("✓ CUDA kernel executed successfully")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Max error: {(out - expected).abs().max().item()}")
    return out


if __name__ == "__main__":
    print("=" * 60)
    print("Testing NCU Profiling")
    print("=" * 60)
    print(f"\n1. CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")

    print("\n2. Running simple CUDA kernel...")
    test_cuda_kernel()
    print("\n3. Kernel execution complete!")
    print("=" * 60)
