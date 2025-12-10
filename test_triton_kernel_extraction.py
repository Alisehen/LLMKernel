#!/usr/bin/env python3
"""Test Triton kernel name extraction."""

from pathlib import Path
from utils.kernel_io import extract_triton_kernel_names

# Test case 1: Simple Triton kernel
triton_code_1 = """
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output
"""

# Test case 2: Autotune decorated kernel
triton_code_2 = """
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=4),
    ],
    key=['M', 'N']
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pass

class ModelNew(torch.nn.Module):
    def forward(self, a, b):
        pass
"""

# Test case 3: Multiple kernels
triton_code_3 = """
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_1(x_ptr):
    pass

@triton.jit
def kernel_2(y_ptr):
    pass

@triton.autotune(configs=[], key=[])
@triton.jit
def kernel_3(z_ptr):
    pass
"""

def test_extraction():
    print("Test 1: Simple @triton.jit kernel")
    names = extract_triton_kernel_names(triton_code_1)
    print(f"  Extracted: {names}")
    assert "add_kernel" in names, f"Expected 'add_kernel', got {names}"
    print("  ✓ PASS\n")

    print("Test 2: @triton.autotune + @triton.jit kernel")
    names = extract_triton_kernel_names(triton_code_2)
    print(f"  Extracted: {names}")
    assert "matmul_kernel" in names, f"Expected 'matmul_kernel', got {names}"
    print("  ✓ PASS\n")

    print("Test 3: Multiple kernels")
    names = extract_triton_kernel_names(triton_code_3)
    print(f"  Extracted: {names}")
    assert "kernel_1" in names, f"Expected 'kernel_1', got {names}"
    assert "kernel_2" in names, f"Expected 'kernel_2', got {names}"
    assert "kernel_3" in names, f"Expected 'kernel_3', got {names}"
    assert len(names) == 3, f"Expected 3 kernels, got {len(names)}"
    print("  ✓ PASS\n")

    print("All tests passed! ✓")

if __name__ == "__main__":
    test_extraction()
