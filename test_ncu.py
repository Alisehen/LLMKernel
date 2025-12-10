#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to test if NCU profiling works
"""

import torch
import triton
import triton.language as tl

# Simple Triton kernel for testing
@triton.jit
def simple_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def test_triton_kernel():
    """Test a simple Triton kernel"""
    size = 1024
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    simple_add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)

    # Verify correctness
    expected = x + y
    assert torch.allclose(output, expected), "Kernel output mismatch!"
    print("✓ Triton kernel executed successfully")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Max error: {(output - expected).abs().max().item()}")

    return output


if __name__ == "__main__":
    print("=" * 60)
    print("Testing NCU Profiling")
    print("=" * 60)

    # Check CUDA availability
    print(f"\n1. CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")

    # Run simple kernel
    print("\n2. Running simple Triton kernel...")
    result = test_triton_kernel()

    print("\n3. Kernel execution complete!")
    print("=" * 60)
