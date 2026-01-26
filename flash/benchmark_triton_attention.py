#!/usr/bin/env python3
"""
Benchmark script for triton_attention implementation
Runs 5 rounds and reports average latency for each round
"""

import torch
import sys
import time

# Add the data directory to path
sys.path.insert(0, '/home/hyc/TritonBench/data/TritonBench_G_v1')

from triton_attention import attention as triton_attention


def benchmark_triton_attention(num_rounds=5, num_warmup=50, num_iters=200):
    """
    Benchmark triton_attention implementation

    Args:
        num_rounds: Number of benchmark rounds to run
        num_warmup: Number of warmup iterations per round
        num_iters: Number of measurement iterations per round
    """

    # Test configuration (same as triton_attention.py)
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    print("="*80)
    print("TRITON ATTENTION BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Batch size:  {batch_size}")
    print(f"  Num heads:   {num_heads}")
    print(f"  Seq length:  {seq_len}")
    print(f"  Head dim:    {head_dim}")
    print(f"  Data type:   torch.float16")
    print(f"\nBenchmark settings:")
    print(f"  Rounds:      {num_rounds}")
    print(f"  Warmup iters: {num_warmup}")
    print(f"  Measure iters: {num_iters}")
    print("="*80)

    # Create input tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device='cuda', dtype=torch.float16)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device='cuda', dtype=torch.float16)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                    device='cuda', dtype=torch.float16)

    sm_scale = 1.0 / (head_dim ** 0.5)

    # Run benchmark rounds
    latencies = []

    for round_num in range(1, num_rounds + 1):
        print(f"\nRound {round_num}/{num_rounds}")
        print("-"*80)

        # Warmup
        print(f"  Warming up ({num_warmup} iterations)...", end=" ", flush=True)
        for _ in range(num_warmup):
            _ = triton_attention(Q, K, V, sm_scale)
        torch.cuda.synchronize()
        print("Done")

        # Measure
        print(f"  Measuring ({num_iters} iterations)...", end=" ", flush=True)
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_iters):
            _ = triton_attention(Q, K, V, sm_scale)

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        print("Done")

        # Calculate average latency for this round
        avg_latency_ms = ((end_time - start_time) / num_iters) * 1000
        latencies.append(avg_latency_ms)

        print(f"  Average latency: {avg_latency_ms:.4f} ms")

    # Summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Round':<10} {'Avg Latency (ms)':<20}")
    print("-"*80)

    for i, latency in enumerate(latencies, 1):
        print(f"{i:<10} {latency:<20.4f}")

    print("-"*80)
    mean_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_latency = (sum((x - mean_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5

    print(f"{'Mean:':<10} {mean_latency:<20.4f}")
    print(f"{'Min:':<10} {min_latency:<20.4f}")
    print(f"{'Max:':<10} {max_latency:<20.4f}")
    print(f"{'Std Dev:':<10} {std_latency:<20.4f}")
    print("="*80)

    return latencies


if __name__ == "__main__":
    # Run benchmark
    latencies = benchmark_triton_attention(
        num_rounds=5,
        num_warmup=50,
        num_iters=200
    )

    print(f"\n✓ Benchmark completed successfully!")
    print(f"  Average across all rounds: {sum(latencies) / len(latencies):.4f} ms")
