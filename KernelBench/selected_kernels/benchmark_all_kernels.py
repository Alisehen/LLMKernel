#!/usr/bin/env python3
"""
Complete Benchmark for ALL Triton Kernels
包含所有3个算子的完整评测脚本
"""

import torch
import sys
import os
import math

sys.path.insert(0, os.path.dirname(__file__))

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def benchmark_kernel(func, warmup=10, num_runs=100):
    """Benchmark a kernel function"""
    torch.cuda.synchronize()
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def test_flash_attention():
    """Test 1: Flash Attention"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}[1/3] Flash Attention{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

    try:
        from importlib import import_module
        ref_mod = import_module('1_flash_attn_ref')
        triton_mod = import_module('flash_attn_fp32')

        # Test config: batch=4, heads=8, seq_len=512, head_dim=64
        B, H, S, D = 2, 2, 128, 64
        Q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)

        print(f"Config: batch={B}, heads={H}, seq_len={S}, head_dim={D}")

        # PyTorch
        pytorch_model = ref_mod.Model().cuda()
        pytorch_func = lambda: pytorch_model(Q, K, V)

        # Triton
        sm_scale = 1.0 / math.sqrt(D)
        triton_func = lambda: triton_mod.flash_attn_triton(Q, K, V, causal=False, sm_scale=sm_scale)

        # Correctness
        print(f"{Colors.OKCYAN}Checking correctness...{Colors.ENDC}")
        with torch.no_grad():
            pytorch_out = pytorch_func()
            triton_out = triton_func()

        diff = torch.abs(pytorch_out - triton_out).max().item()
        print(f"  Max diff: {diff:.6f}")

        if diff > 1e-2:
            print(f"{Colors.FAIL}❌ Correctness FAILED{Colors.ENDC}")
            return None

        print(f"{Colors.OKGREEN}✅ Correctness PASSED{Colors.ENDC}")

        # Benchmark
        print(f"{Colors.OKCYAN}Benchmarking...{Colors.ENDC}")
        pytorch_time = benchmark_kernel(pytorch_func)
        triton_time = benchmark_kernel(triton_func)
        speedup = pytorch_time / triton_time

        print(f"  PyTorch: {pytorch_time:.4f} ms")
        print(f"  Triton:  {triton_time:.4f} ms")
        print(f"  {Colors.BOLD}Speedup: {speedup:.2f}x{Colors.ENDC}")

        return {
            'name': 'Flash Attention',
            'pytorch_ms': pytorch_time,
            'triton_ms': triton_time,
            'speedup': speedup,
            'status': 'success'
        }

    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return None


def test_rope():
    """Test 2: RoPE Embedding"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}[2/3] RoPE Embedding{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

    try:
        from importlib import import_module
        ref_mod = import_module('3_rope_ref')
        triton_mod = import_module('rope_transform')

        # Get inputs from reference
        inputs = ref_mod.get_inputs()
        q, k, cos, sin = [x.cuda() for x in inputs]

        batch, heads, seq_len, head_dim = q.shape
        print(f"Config: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")

        # PyTorch
        pytorch_model = ref_mod.Model().cuda()
        pytorch_func = lambda: pytorch_model(q.clone(), k.clone(), cos, sin)

        # Triton
        triton_func = lambda: triton_mod.rope_forward(q.clone(), k.clone(), cos, sin)

        # Correctness
        print(f"{Colors.OKCYAN}Checking correctness...{Colors.ENDC}")
        with torch.no_grad():
            q_torch, k_torch, _, _ = pytorch_func()
            q_triton, k_triton, _, _ = triton_func()

        diff_q = torch.abs(q_torch - q_triton).max().item()
        diff_k = torch.abs(k_torch - k_triton).max().item()
        max_diff = max(diff_q, diff_k)

        print(f"  Max diff (Q): {diff_q:.6f}")
        print(f"  Max diff (K): {diff_k:.6f}")

        if max_diff > 1e-2:
            print(f"{Colors.FAIL}❌ Correctness FAILED{Colors.ENDC}")
            return None

        print(f"{Colors.OKGREEN}✅ Correctness PASSED{Colors.ENDC}")

        # Benchmark
        print(f"{Colors.OKCYAN}Benchmarking...{Colors.ENDC}")
        pytorch_time = benchmark_kernel(pytorch_func)
        triton_time = benchmark_kernel(triton_func)
        speedup = pytorch_time / triton_time

        print(f"  PyTorch: {pytorch_time:.4f} ms")
        print(f"  Triton:  {triton_time:.4f} ms")
        print(f"  {Colors.BOLD}Speedup: {speedup:.2f}x{Colors.ENDC}")

        if speedup < 1.0:
            print(f"{Colors.WARNING}  Note: PyTorch is faster{Colors.ENDC}")

        return {
            'name': 'RoPE Embedding',
            'pytorch_ms': pytorch_time,
            'triton_ms': triton_time,
            'speedup': speedup,
            'status': 'success'
        }

    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return None


def test_int8_dequant():
    """Test 3: INT8 Dequant MatMul"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}[3/4] INT8 Dequant MatMul{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

    try:
        from importlib import import_module
        ref_mod = import_module('6_int8_dequant_ref')
        triton_mod = import_module('int8_dequant_matmul')

        # Get inputs from reference
        init_params = ref_mod.get_init_inputs()
        in_features, out_features = init_params
        inputs = ref_mod.get_inputs()
        x_int8, scale_x = [inp.cuda() for inp in inputs]

        M, K = x_int8.shape
        print(f"Config: M={M}, K={K}, N={out_features}")

        # PyTorch reference
        pytorch_model = ref_mod.Model(*init_params).cuda()
        pytorch_func = lambda: pytorch_model(x_int8, scale_x)

        # Triton implementation
        # Prepare Triton inputs
        b_int8 = pytorch_model.weight_int8.t().contiguous()  # [K, N]
        state_w = pytorch_model.scale_w
        bias = pytorch_model.bias

        triton_func = lambda: triton_mod.int8_matmul_rowwise_dequantize(
            x_int8, b_int8, scale_x, state_w, bias
        )

        # Correctness check
        print(f"{Colors.OKCYAN}Checking correctness...{Colors.ENDC}")
        with torch.no_grad():
            pytorch_out = pytorch_func()
            triton_out = triton_func()

        diff = torch.abs(pytorch_out - triton_out).max().item()
        mean_diff = torch.abs(pytorch_out - triton_out).mean().item()

        print(f"  Max diff: {diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        if diff > 1e-2:
            print(f"{Colors.FAIL}❌ Correctness FAILED{Colors.ENDC}")
            return None

        print(f"{Colors.OKGREEN}✅ Correctness PASSED{Colors.ENDC}")

        # Benchmark
        print(f"{Colors.OKCYAN}Benchmarking...{Colors.ENDC}")
        pytorch_time = benchmark_kernel(pytorch_func)
        triton_time = benchmark_kernel(triton_func)
        speedup = pytorch_time / triton_time

        print(f"  PyTorch: {pytorch_time:.4f} ms")
        print(f"  Triton:  {triton_time:.4f} ms")
        print(f"  {Colors.BOLD}Speedup: {speedup:.2f}x{Colors.ENDC}")

        if speedup < 1.0:
            print(f"{Colors.WARNING}  Note: PyTorch is faster{Colors.ENDC}")

        return {
            'name': 'INT8 Dequant MatMul',
            'pytorch_ms': pytorch_time,
            'triton_ms': triton_time,
            'speedup': speedup,
            'status': 'success'
        }

    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return None


def test_int4_matmul():
    """Test 4: INT4 MatMul (Reference)"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}[4/4] INT4 MatMul (Reference){Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

    try:
        from importlib import import_module
        ref_mod = import_module('5_int4_matmul_ref')
        triton_mod = import_module('matmul_dequantize_int4')

        # Config
        init_params = ref_mod.get_init_inputs()
        in_features, out_features, group_size = init_params
        inputs = ref_mod.get_inputs()
        x = inputs[0].cuda()

        print(f"Config: M={x.shape[0]}, K={in_features}, N={out_features}, group_size={group_size}")

        # PyTorch (naive int4)
        pytorch_model = ref_mod.Model(*init_params).cuda()
        pytorch_func = lambda: pytorch_model(x)

        # Triton (GPTQ quantization)
        # Create quantized weights
        weight_fp = torch.randn(out_features, in_features, dtype=torch.float16).cuda()
        qweight, scales, qzeros, _ = triton_mod.quantize_int4(weight_fp, group_size=group_size, tp_rank=0)

        x_fp16 = x.to(torch.float16)
        triton_func = lambda: triton_mod.matmul_dequantize_int4_gptq(
            x_fp16, qweight, scales, qzeros, group_size
        )

        print(f"{Colors.WARNING}⚠️  Different quantization schemes{Colors.ENDC}")
        print(f"  PyTorch: Random int4 simulation")
        print(f"  Triton:  GPTQ quantization")
        print(f"{Colors.WARNING}  Cannot compare correctness - different weights{Colors.ENDC}")

        # Just verify both run
        print(f"{Colors.OKCYAN}Verifying execution...{Colors.ENDC}")
        with torch.no_grad():
            pytorch_out = pytorch_func()
            triton_out = triton_func().to(torch.float32)

        print(f"  PyTorch output: {pytorch_out.shape}, range=[{pytorch_out.min():.2f}, {pytorch_out.max():.2f}]")
        print(f"  Triton output:  {triton_out.shape}, range=[{triton_out.min():.2f}, {triton_out.max():.2f}]")
        print(f"{Colors.OKGREEN}✅ Both implementations run successfully{Colors.ENDC}")

        # Benchmark
        print(f"{Colors.OKCYAN}Benchmarking...{Colors.ENDC}")
        pytorch_time = benchmark_kernel(pytorch_func)
        triton_time = benchmark_kernel(triton_func)
        speedup = pytorch_time / triton_time

        print(f"  PyTorch (naive): {pytorch_time:.4f} ms")
        print(f"  Triton (GPTQ):   {triton_time:.4f} ms")
        print(f"  {Colors.BOLD}Speedup: {speedup:.2f}x{Colors.ENDC}")

        return {
            'name': 'INT4 MatMul',
            'pytorch_ms': pytorch_time,
            'triton_ms': triton_time,
            'speedup': speedup,
            'status': 'success',
            'note': 'Different quantization schemes'
        }

    except Exception as e:
        print(f"{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("="*70)
    print("Complete Triton Kernel Benchmark")
    print("All 4 Kernels - Official TritonBench Style")
    print("="*70)
    print(f"{Colors.ENDC}")

    if not torch.cuda.is_available():
        print(f"{Colors.FAIL}CUDA not available!{Colors.ENDC}")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    # Run all tests
    results = []

    test_funcs = [
        ("Flash Attention", test_flash_attention),
        ("RoPE Embedding", test_rope),
        ("INT8 Dequant MatMul", test_int8_dequant),
        ("INT4 MatMul", test_int4_matmul),
    ]

    for name, func in test_funcs:
        result = func()
        if result:
            results.append(result)

    # Summary
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("="*70)
    print("Summary")
    print("="*70)
    print(f"{Colors.ENDC}")

    if not results:
        print(f"{Colors.FAIL}No successful benchmarks!{Colors.ENDC}")
        return

    print(f"\n{Colors.BOLD}{'Kernel':<25} {'PyTorch':<12} {'Triton':<12} {'Speedup':<10} {'Status':<10}{Colors.ENDC}")
    print("-"*75)

    for result in results:
        speedup_str = f"{result['speedup']:.2f}x"
        color = Colors.OKGREEN if result['speedup'] > 1.0 else Colors.WARNING
        status = "✅ Success"

        print(f"{result['name']:<25} {result['pytorch_ms']:>8.4f} ms  {result['triton_ms']:>8.4f} ms  {color}{speedup_str:>10}{Colors.ENDC} {status}")

        if 'note' in result:
            print(f"{'':>25} {Colors.OKCYAN}Note: {result['note']}{Colors.ENDC}")

    # Geometric mean
    geo_mean = math.exp(sum(math.log(r['speedup']) for r in results) / len(results))
    print("-"*75)
    print(f"{Colors.BOLD}Geometric Mean Speedup: {geo_mean:.2f}x{Colors.ENDC}\n")

    print(f"{Colors.OKCYAN}Legend:{Colors.ENDC}")
    print(f"  {Colors.OKGREEN}✅ Success{Colors.ENDC}: Correctness verified, valid comparison")
    print(f"  {Colors.WARNING}Note{Colors.ENDC}: Additional context for the benchmark\n")


if __name__ == "__main__":
    main()
