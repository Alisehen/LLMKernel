#!/usr/bin/env python3
"""
Comprehensive Flash Attention Benchmark
Compares 3 implementations:
  1. Triton Flash Attention (triton_attention from TritonBench)
  2. Custom Triton Flash Attention (flash_attention_causal from test_kernel_analysis_seed0.py)
  3. PyTorch F.scaled_dot_product_attention

Tests with sequence lengths from 128 to 4096
"""

import torch
import torch.nn.functional as F
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add Triton implementation paths
sys.path.insert(0, '/home/hyc/TritonBench/data/TritonBench_G_v1')
sys.path.insert(0, '/home/hyc/LLMKernel/flash')

from triton_attention import attention as triton_attention
from test_kernel_analysis_seed0 import flash_attention_causal


# Configuration as specified by user
CONFIG = {
    "batch_size": 32,
    "num_heads": 32,
    "head_dim": 128,
    "description": "batch=1, heads=32, head_dim=128 (4096 dim total)"
}


def benchmark_implementation(
    impl_name: str,
    impl_func,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    sm_scale: float,
    warmup: int = 50,
    repeat: int = 200,
    impl_type: str = "triton"
) -> Dict:
    """
    Benchmark a single attention implementation.

    Args:
        impl_name: Name of the implementation
        impl_func: Function to benchmark
        Q, K, V: Input tensors
        sm_scale: Scaling factor
        warmup: Warmup iterations
        repeat: Measurement iterations
        impl_type: "triton", "triton_custom", or "pytorch"

    Returns:
        Dictionary with benchmark results
    """
    # Warmup
    for _ in range(warmup):
        if impl_type == "pytorch":
            _ = impl_func(Q, K, V, is_causal=True)
        elif impl_type == "triton":
            _ = impl_func(Q, K, V, sm_scale)
        elif impl_type == "triton_custom":
            _ = impl_func(Q, K, V)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        if impl_type == "pytorch":
            _ = impl_func(Q, K, V, is_causal=True)
        elif impl_type == "triton":
            _ = impl_func(Q, K, V, sm_scale)
        elif impl_type == "triton_custom":
            _ = impl_func(Q, K, V)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_latency_ms = (elapsed / repeat) * 1000

    return {
        "name": impl_name,
        "avg_latency_ms": avg_latency_ms,
        "total_time_s": elapsed,
        "iterations": repeat
    }


def test_correctness(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    sm_scale: float
) -> Dict:
    """Test correctness by comparing all implementations against PyTorch."""

    # PyTorch output (reference)
    pytorch_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    # Triton TritonBench output
    triton_out = triton_attention(Q, K, V, sm_scale)

    # Custom Triton output
    custom_out = flash_attention_causal(Q, K, V)

    # Compute differences
    diff_triton = torch.abs(pytorch_out - triton_out)
    diff_custom = torch.abs(pytorch_out - custom_out)

    max_diff_triton = diff_triton.max().item()
    mean_diff_triton = diff_triton.mean().item()
    is_close_triton = torch.allclose(pytorch_out, triton_out, atol=1e-2, rtol=1e-2)

    max_diff_custom = diff_custom.max().item()
    mean_diff_custom = diff_custom.mean().item()
    is_close_custom = torch.allclose(pytorch_out, custom_out, atol=1e-2, rtol=1e-2)

    return {
        "triton_bench": {
            "max_diff": max_diff_triton,
            "mean_diff": mean_diff_triton,
            "is_close": is_close_triton
        },
        "triton_custom": {
            "max_diff": max_diff_custom,
            "mean_diff": mean_diff_custom,
            "is_close": is_close_custom
        }
    }


def run_comprehensive_benchmark(
    seq_lengths: List[int] = [128, 256, 512, 1024, 2048],
    warmup: int = 50,
    repeat: int = 200,
    output_file: str = None
):
    """
    Run comprehensive benchmark across all sequence lengths.

    Args:
        seq_lengths: List of sequence lengths to test
        warmup: Warmup iterations
        repeat: Measurement iterations
        output_file: Optional JSON output file
    """

    print("="*120)
    print("FLASH ATTENTION COMPREHENSIVE BENCHMARK - 3 IMPLEMENTATIONS")
    print("="*120)
    print(f"\nConfiguration:")
    print(f"  Batch size:  {CONFIG['batch_size']}")
    print(f"  Num heads:   {CONFIG['num_heads']}")
    print(f"  Head dim:    {CONFIG['head_dim']}")
    print(f"  Description: {CONFIG['description']}")
    print(f"\nBenchmark settings:")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"  Warmup:  {warmup}")
    print(f"  Repeat:  {repeat}")
    print(f"  Dtype:   torch.float16")
    print(f"  Device:  {torch.cuda.get_device_name()}")
    print(f"\nImplementations:")
    print(f"  1. PyTorch F.scaled_dot_product_attention (reference)")
    print(f"  2. Triton Flash Attention (TritonBench)")
    print(f"  3. Custom Triton Flash Attention (test_kernel_analysis_seed0.py)")
    print("="*120)

    batch_size = CONFIG["batch_size"]
    num_heads = CONFIG["num_heads"]
    head_dim = CONFIG["head_dim"]

    results_by_seqlen = {}

    for seq_len in seq_lengths:
        print(f"\n{'='*120}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'='*120}")

        # Create input tensors
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', dtype=torch.float16)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', dtype=torch.float16)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                       device='cuda', dtype=torch.float16)

        sm_scale = 1.0 / (head_dim ** 0.5)

        # Test correctness
        print(f"\n  Correctness Check:")
        print(f"  {'-'*116}")
        correctness = test_correctness(Q, K, V, sm_scale)

        triton_bench_status = "✓ PASS" if correctness["triton_bench"]["is_close"] else "✗ FAIL"
        triton_custom_status = "✓ PASS" if correctness["triton_custom"]["is_close"] else "✗ FAIL"

        print(f"    Triton (TritonBench):  {triton_bench_status} (max_diff: {correctness['triton_bench']['max_diff']:.2e}, mean_diff: {correctness['triton_bench']['mean_diff']:.2e})")
        print(f"    Triton (Custom):       {triton_custom_status} (max_diff: {correctness['triton_custom']['max_diff']:.2e}, mean_diff: {correctness['triton_custom']['mean_diff']:.2e})")

        # Benchmark all implementations
        print(f"\n  Performance Benchmark:")
        print(f"  {'-'*116}")

        # 1. PyTorch
        print(f"    Benchmarking PyTorch...", end=" ", flush=True)
        pytorch_result = benchmark_implementation(
            "PyTorch F.sdpa",
            F.scaled_dot_product_attention,
            Q, K, V, sm_scale,
            warmup=warmup,
            repeat=repeat,
            impl_type="pytorch"
        )
        print(f"{pytorch_result['avg_latency_ms']:.4f} ms")

        # 2. Triton TritonBench
        print(f"    Benchmarking Triton (TritonBench)...", end=" ", flush=True)
        triton_bench_result = benchmark_implementation(
            "Triton TritonBench",
            triton_attention,
            Q, K, V, sm_scale,
            warmup=warmup,
            repeat=repeat,
            impl_type="triton"
        )
        print(f"{triton_bench_result['avg_latency_ms']:.4f} ms")

        # 3. Custom Triton
        print(f"    Benchmarking Triton (Custom)...", end=" ", flush=True)
        triton_custom_result = benchmark_implementation(
            "Triton Custom",
            flash_attention_causal,
            Q, K, V, sm_scale,
            warmup=warmup,
            repeat=repeat,
            impl_type="triton_custom"
        )
        print(f"{triton_custom_result['avg_latency_ms']:.4f} ms")

        # Calculate speedups
        speedup_bench = pytorch_result['avg_latency_ms'] / triton_bench_result['avg_latency_ms']
        speedup_custom = pytorch_result['avg_latency_ms'] / triton_custom_result['avg_latency_ms']

        print(f"\n  Speedup vs PyTorch:")
        print(f"    Triton (TritonBench): {speedup_bench:.2f}x")
        print(f"    Triton (Custom):      {speedup_custom:.2f}x")

        # Store results
        results_by_seqlen[seq_len] = {
            "pytorch": pytorch_result,
            "triton_bench": triton_bench_result,
            "triton_custom": triton_custom_result,
            "speedup_bench": speedup_bench,
            "speedup_custom": speedup_custom,
            "correctness": correctness
        }

        # Free memory
        del Q, K, V
        torch.cuda.empty_cache()

    # Print summary table
    print(f"\n\n{'='*120}")
    print("SUMMARY TABLE")
    print(f"{'='*120}\n")

    print(f"{'Seq Len':<12} {'PyTorch (ms)':<15} {'Triton-Bench (ms)':<18} {'Triton-Custom (ms)':<18} {'Speedup-Bench':<15} {'Speedup-Custom':<15}")
    print(f"{'-'*120}")

    for seq_len, results in results_by_seqlen.items():
        pytorch_ms = results['pytorch']['avg_latency_ms']
        bench_ms = results['triton_bench']['avg_latency_ms']
        custom_ms = results['triton_custom']['avg_latency_ms']
        speedup_bench = results['speedup_bench']
        speedup_custom = results['speedup_custom']

        print(f"{seq_len:<12} {pytorch_ms:<15.4f} {bench_ms:<18.4f} {custom_ms:<18.4f} {speedup_bench:<15.2f}x {speedup_custom:<15.2f}x")

    # Calculate overall statistics
    print(f"\n\n{'='*120}")
    print("OVERALL STATISTICS")
    print(f"{'='*120}\n")

    speedups_bench = [r['speedup_bench'] for r in results_by_seqlen.values()]
    speedups_custom = [r['speedup_custom'] for r in results_by_seqlen.values()]

    avg_speedup_bench = sum(speedups_bench) / len(speedups_bench)
    avg_speedup_custom = sum(speedups_custom) / len(speedups_custom)

    import math
    geo_mean_bench = math.exp(sum(math.log(s) for s in speedups_bench) / len(speedups_bench))
    geo_mean_custom = math.exp(sum(math.log(s) for s in speedups_custom) / len(speedups_custom))

    print(f"Triton (TritonBench) vs PyTorch:")
    print(f"  Average speedup:    {avg_speedup_bench:.2f}x")
    print(f"  Geometric mean:     {geo_mean_bench:.2f}x")
    print(f"  Max speedup:        {max(speedups_bench):.2f}x")
    print(f"  Min speedup:        {min(speedups_bench):.2f}x")

    print(f"\nTriton (Custom) vs PyTorch:")
    print(f"  Average speedup:    {avg_speedup_custom:.2f}x")
    print(f"  Geometric mean:     {geo_mean_custom:.2f}x")
    print(f"  Max speedup:        {max(speedups_custom):.2f}x")
    print(f"  Min speedup:        {min(speedups_custom):.2f}x")

    # Compare the two Triton implementations
    print(f"\nTriton-Bench vs Triton-Custom:")
    custom_vs_bench = []
    for seq_len, results in results_by_seqlen.items():
        ratio = results['triton_bench']['avg_latency_ms'] / results['triton_custom']['avg_latency_ms']
        custom_vs_bench.append(ratio)

    avg_ratio = sum(custom_vs_bench) / len(custom_vs_bench)
    geo_mean_ratio = math.exp(sum(math.log(r) for r in custom_vs_bench) / len(custom_vs_bench))

    winner = "Custom" if geo_mean_ratio > 1.0 else "TritonBench"
    print(f"  Average ratio (Bench/Custom): {avg_ratio:.2f}x")
    print(f"  Geometric mean:               {geo_mean_ratio:.2f}x")
    print(f"  Overall winner:               {winner}")

    # Save to JSON
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "config": CONFIG,
            "benchmark_config": {
                "seq_lengths": seq_lengths,
                "warmup": warmup,
                "repeat": repeat,
                "device": torch.cuda.get_device_name(),
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            },
            "results": results_by_seqlen,
            "statistics": {
                "triton_bench": {
                    "avg_speedup": avg_speedup_bench,
                    "geo_mean_speedup": geo_mean_bench,
                    "max_speedup": max(speedups_bench),
                    "min_speedup": min(speedups_bench),
                },
                "triton_custom": {
                    "avg_speedup": avg_speedup_custom,
                    "geo_mean_speedup": geo_mean_custom,
                    "max_speedup": max(speedups_custom),
                    "min_speedup": min(speedups_custom),
                },
                "bench_vs_custom": {
                    "avg_ratio": avg_ratio,
                    "geo_mean_ratio": geo_mean_ratio,
                    "winner": winner
                }
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    print(f"\n{'='*120}\n")

    return results_by_seqlen


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark 3 Flash Attention implementations")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048, 4096],
        help="Sequence lengths to test (default: 128 256 512 1024 2048 4096)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Warmup iterations (default: 50)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=500,
        help="Measurement iterations (default: 200)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flash_attention_3way_comparison.json",
        help="Output JSON file (default: flash_attention_3way_comparison.json)"
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_comprehensive_benchmark(
        seq_lengths=args.seq_lengths,
        warmup=args.warmup,
        repeat=args.repeat,
        output_file=args.output
    )

    print("✓ Benchmark completed successfully!")
