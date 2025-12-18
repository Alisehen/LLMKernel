#!/usr/bin/env python3
"""
Benchmark generated Triton kernels against PyTorch reference implementations.
Generates summary markdown tables for level1, level2, level3.
"""
import os
import sys
import re
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.compile_and_run import compare_and_bench, CompilationError, AccuracyError


def _subprocess_worker(queue, args):
    """子进程 worker 函数，运行单个 kernel benchmark"""
    test_py, ref_py, device_idx, warmup, repeat, tol, rtol = args

    result = {
        "name": test_py.stem,
        "runnable": False,
        "score": 0.0,
        "error": None,
        "ref_latency_ms": None,
        "test_latency_ms": None,
    }

    try:
        metrics = compare_and_bench(
            ref_py=ref_py,
            test_py=test_py,
            device_idx=device_idx,
            warmup=warmup,
            repeat=repeat,
            tol=tol,
            rtol=rtol,
        )

        ref_avg = metrics["ref_latency_ms"]["avg"]
        test_avg = metrics["test_latency_ms"]["avg"]
        speedup = ref_avg / max(1e-9, test_avg)

        result["runnable"] = True
        result["score"] = speedup
        result["ref_latency_ms"] = ref_avg
        result["test_latency_ms"] = test_avg
        result["max_abs_err"] = metrics.get("max_abs_err", -1)

    except CompilationError as e:
        result["error"] = f"CompilationError: {str(e)[:500]}"
    except AccuracyError as e:
        result["error"] = f"AccuracyError: {str(e)[:500]}"
    except Exception as e:
        err_str = str(e)
        err_lines = err_str.strip().split('\n')
        last_lines = '\n'.join(err_lines[-5:]) if len(err_lines) > 5 else err_str
        result["error"] = f"{type(e).__name__}: {last_lines[:500]}"

    queue.put(result)


def natural_sort_key(path: Path) -> tuple:
    """Extract leading number from filename for natural sorting."""
    name = path.stem
    match = re.match(r'^(\d+)', name)
    return (int(match.group(1)), name) if match else (float('inf'), name)


def get_kernel_pairs(generated_dir: Path, ref_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching kernel pairs between generated and reference directories."""
    pairs = []
    generated_files = {f.name: f for f in generated_dir.glob("*.py")}
    ref_files = {f.name: f for f in ref_dir.glob("*.py")}

    # Match by filename
    for name, gen_path in generated_files.items():
        if name in ref_files:
            pairs.append((gen_path, ref_files[name]))

    # Sort by natural order
    pairs.sort(key=lambda x: natural_sort_key(x[0]))
    return pairs


def bench_single_kernel(
    test_py: Path,
    ref_py: Path,
    device_idx: int = 0,
    warmup: int = 5,
    repeat: int = 20,
    tol: float = 0.1,
    rtol: float = 0.1,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Benchmark a single kernel pair in a subprocess to isolate CUDA errors."""
    # 使用子进程运行，隔离 CUDA 错误
    ctx = mp.get_context('spawn')  # 使用 spawn 确保干净的 CUDA 上下文
    result_queue = ctx.Queue()

    args = (test_py, ref_py, device_idx, warmup, repeat, tol, rtol)
    p = ctx.Process(target=_subprocess_worker, args=(result_queue, args))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        # 超时，杀掉进程
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()
        return {
            "name": test_py.stem,
            "runnable": False,
            "score": 0.0,
            "error": f"Timeout: kernel took longer than {timeout}s",
            "ref_latency_ms": None,
            "test_latency_ms": None,
        }

    if p.exitcode != 0:
        # 进程异常退出（如 CUDA 错误导致 crash）
        return {
            "name": test_py.stem,
            "runnable": False,
            "score": 0.0,
            "error": f"Process crashed with exit code {p.exitcode} (likely CUDA error)",
            "ref_latency_ms": None,
            "test_latency_ms": None,
        }

    try:
        result = result_queue.get_nowait()
    except Exception:
        result = {
            "name": test_py.stem,
            "runnable": False,
            "score": 0.0,
            "error": "Failed to get result from subprocess",
            "ref_latency_ms": None,
            "test_latency_ms": None,
        }

    return result


def generate_markdown_table(results: List[Dict[str, Any]], level: str) -> str:
    """Generate markdown table from results."""
    lines = [
        f"# Kernel Benchmark Summary - {level}",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| # | Kernel Name | Speedup | Status | Ref (ms) | Triton (ms) |",
        "|---|-------------|---------|--------|----------|-------------|",
    ]

    successful = 0
    total_speedup = 0.0

    for i, r in enumerate(results, 1):
        name = r["name"]
        if r["runnable"]:
            score = r["score"]
            status = "✅"
            ref_ms = f"{r['ref_latency_ms']:.4f}" if r['ref_latency_ms'] else "N/A"
            test_ms = f"{r['test_latency_ms']:.4f}" if r['test_latency_ms'] else "N/A"
            score_str = f"{score:.4f}"
            successful += 1
            total_speedup += score
        else:
            score_str = "N/A"
            status = "❌"
            ref_ms = "N/A"
            test_ms = "N/A"

        lines.append(f"| {i} | {name} | {score_str} | {status} | {ref_ms} | {test_ms} |")

    # Summary
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total kernels: {len(results)}")
    lines.append(f"- Successful: {successful}")
    lines.append(f"- Failed: {len(results) - successful}")
    if successful > 0:
        lines.append(f"- Average speedup (successful only): {total_speedup / successful:.4f}")
    lines.append(f"- Success rate: {successful / len(results) * 100:.1f}%")

    # Failed kernels details
    failed = [r for r in results if not r["runnable"]]
    if failed:
        lines.append("")
        lines.append("## Failed Kernels")
        for r in failed:
            lines.append(f"- **{r['name']}**: {r['error'][:100]}...")

    return "\n".join(lines)


def _bench_worker_wrapper(args):
    """Pool worker wrapper"""
    test_py, ref_py, device_idx, warmup, repeat, tol, rtol, timeout = args
    return bench_single_kernel(test_py, ref_py, device_idx, warmup, repeat, tol, rtol, timeout)


def bench_level(
    level: str,
    generated_base: Path,
    ref_base: Path,
    output_dir: Path,
    device_idx: int = 0,
    warmup: int = 5,
    repeat: int = 20,
    tol: float = 0.1,
    rtol: float = 0.1,
    parallel: int = 1,
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    """Benchmark all kernels for a level."""
    generated_dir = generated_base / level
    ref_dir = ref_base / level

    if not generated_dir.exists():
        print(f"[WARN] Generated dir not found: {generated_dir}")
        return []
    if not ref_dir.exists():
        print(f"[WARN] Reference dir not found: {ref_dir}")
        return []

    pairs = get_kernel_pairs(generated_dir, ref_dir)
    print(f"\n{'='*60}")
    print(f"[{level}] Found {len(pairs)} kernel pairs to benchmark (parallel={parallel})")
    print(f"{'='*60}")

    # 准备参数
    task_args = [
        (test_py, ref_py, device_idx, warmup, repeat, tol, rtol, timeout)
        for test_py, ref_py in pairs
    ]

    if parallel > 1:
        # 并行执行
        print(f"Running {len(pairs)} benchmarks with {parallel} workers...")
        with mp.Pool(processes=parallel) as pool:
            results = []
            for i, result in enumerate(pool.imap(_bench_worker_wrapper, task_args), 1):
                results.append(result)
                if result["runnable"]:
                    print(f"[{i}/{len(pairs)}] ✅ {result['name']}: {result['score']:.2f}x")
                else:
                    print(f"[{i}/{len(pairs)}] ❌ {result['name']}: {result['error'][:50]}...")
    else:
        # 串行执行
        results = []
        for i, (test_py, ref_py) in enumerate(pairs, 1):
            print(f"\n[{i}/{len(pairs)}] Benchmarking: {test_py.stem}")

            result = bench_single_kernel(
                test_py=test_py,
                ref_py=ref_py,
                device_idx=device_idx,
                warmup=warmup,
                repeat=repeat,
                tol=tol,
                rtol=rtol,
                timeout=timeout,
            )
            results.append(result)

            if result["runnable"]:
                print(f"  ✅ Score: {result['score']:.4f} (ref: {result['ref_latency_ms']:.4f}ms, triton: {result['test_latency_ms']:.4f}ms)")
            else:
                print(f"  ❌ Failed: {result['error'][:80]}...")

    # Generate and save markdown
    md_content = generate_markdown_table(results, level)
    md_path = output_dir / f"summary_table_{level}.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"\n[{level}] Summary saved to: {md_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton kernels against PyTorch reference")
    parser.add_argument("--generated", type=Path, default=Path("/home/hyc/generated_kernels"),
                        help="Path to generated kernels directory")
    parser.add_argument("--ref", type=Path, default=Path("/home/hyc/LLMKernel/KernelBench"),
                        help="Path to reference KernelBench directory")
    parser.add_argument("--output", type=Path, default=Path("/home/hyc/LLMKernel"),
                        help="Output directory for summary tables")
    parser.add_argument("--device", type=int, default=6, help="CUDA device index")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--tol", type=float, default=0.1, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.1, help="Relative tolerance")
    parser.add_argument("--levels", nargs="+", default=["level1", "level2", "level3"],
                        help="Levels to benchmark")
    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    print(f"Benchmarking Triton kernels")
    print(f"  Generated: {args.generated}")
    print(f"  Reference: {args.ref}")
    print(f"  Output: {args.output}")
    print(f"  Device: {args.device}")
    print(f"  Levels: {args.levels}")

    all_results = {}
    for level in args.levels:
        results = bench_level(
            level=level,
            generated_base=args.generated,
            ref_base=args.ref,
            output_dir=args.output,
            device_idx=0,  # After CUDA_VISIBLE_DEVICES, use index 0
            warmup=args.warmup,
            repeat=args.repeat,
            tol=args.tol,
            rtol=args.rtol,
        )
        all_results[level] = results

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for level, results in all_results.items():
        if not results:
            continue
        successful = sum(1 for r in results if r["runnable"])
        # 失败的记为 0，计算所有 kernel 的平均 speedup
        total_speedup = sum(r["score"] if r["runnable"] else 0.0 for r in results)
        avg_speedup = total_speedup / len(results)
        avg_speedup_success_only = total_speedup / max(1, successful)
        print(f"[{level}] {successful}/{len(results)} successful, avg speedup: {avg_speedup:.4f} (success only: {avg_speedup_success_only:.4f})")


if __name__ == "__main__":
    main()
