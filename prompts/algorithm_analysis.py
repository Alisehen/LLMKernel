#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt builder for **Algorithmic Structure Analysis** (Level 2/3 only).

This analyzes the seed kernel at a high level to identify:
- Fusion opportunities
- Algorithm replacement (e.g., Flash Attention, Winograd)
- Computational graph reordering
- Memory layout optimizations
"""

from __future__ import annotations
from pathlib import Path
from string import Template
from textwrap import dedent
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

__all__ = ["build_algorithm_analysis_prompt"]

# -----------------------------------------------------------------------------
# GPU spec loader (shared pattern)
# -----------------------------------------------------------------------------

def _load_gpu_spec() -> dict:
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]

# -----------------------------------------------------------------------------
# Algorithm Analysis Prompt Template
# -----------------------------------------------------------------------------

algorithm_analysis_tmpl = Template(
    dedent(
        """You are a GPU kernel optimization architect. Analyze the kernel and identify **ONE high-level algorithmic optimization**.

# PyTorch Reference
```python
$python_code
```

# Current Triton Kernel
```python
$cuda_code
```

$performance_section

---

## Analysis Steps

1. **Code Analysis**: Count kernels, identify operations, check for inefficiencies
2. **Performance Diagnosis**: Use metrics/latency to identify bottleneck type
3. **Root Cause**: Combine code + performance to find the core issue

## Optimization Categories (pick ONE if worth optimizing):

### 1. Operator Fusion
Fuse consecutive ops into fewer kernels to reduce memory traffic and launch overhead.

### 2. Algorithm Replacement
Replace naive algorithm with optimized variant.
- For Attention: Flash Attention, online softmax
- For Convolution: Winograd, im2col
- **For RNN/GRU/LSTM**: Persistent kernel with HYBRID computation
  - **CRITICAL**: Use hybrid approach for best performance:
    * Precompute input-side gates ONCE (outside kernel): `gates_x = (T*B, In) @ W_ih`
    * Persistent kernel (inside): only recurrent-side: `for t: gates_h = h @ W_hh`
  - Time loop `for t in range(T)` must be inside kernel, NOT in Python
  - Launch kernel once per layer, not once per timestep
  - Expected speedup: 10-100x (vs per-timestep launches)

### 3. Kernel Launch Reduction
Combine multiple small kernels to reduce overhead.
- **For RNN/GRU/LSTM**: See "Algorithm Replacement" above for persistent kernel approach

### 4. Memory Layout Optimization
Use in-place operations, buffer reuse, or better layouts.

## Should We Optimize?

Before proposing optimization, determine if it's worthwhile:
- **Not worth optimizing** if:
  - Code is already near-optimal (expected speedup < 10%)
  - Bottleneck cannot be addressed (hardware limited, already optimal algorithm)
  - Optimization would add significant complexity with minimal gain

- **Worth optimizing** if:
  - Clear algorithmic inefficiency exists (multiple kernels, suboptimal algorithm)
  - Expected speedup >= 20%
  - Concrete optimization path available

## Output (JSON)

```json
{
  "worth_optimizing": "yes/no",
  "reason": "<Why worth or not worth optimizing, 1 sentence>",
  "bottleneck": "<Root cause in 1-2 sentences, empty if not worth optimizing>",
  "optimisation method": "<Specific optimization in 1-2 sentences, empty if not worth optimizing>",
  "modification plan": "<Implementation steps in 2-3 sentences, empty if not worth optimizing>",
  "expected_speedup": "<e.g., '30-40%', empty if not worth optimizing>"
}
```

Return JSON only.
"""
    )
)

# -----------------------------------------------------------------------------
# Builder Function
# -----------------------------------------------------------------------------

def build_algorithm_analysis_prompt(
    *,
    arch_path: Path,
    gpu_name: str,
    cuda_code: str,
    ncu_metrics_block: str = "",
    current_latency_ms: float | None = None,
    baseline_latency_ms: float | None = None,
) -> str:
    """Build algorithm analysis prompt for high-level optimization.

    Args:
        arch_path: Path to PyTorch reference model
        gpu_name: Target GPU name
        cuda_code: Current Triton kernel code
        ncu_metrics_block: NCU profiling metrics (optional, formatted string)
        current_latency_ms: Current kernel latency in ms (optional)
        baseline_latency_ms: PyTorch baseline latency in ms (optional)

    Returns:
        Complete prompt string for LLM
    """
    # Read PyTorch reference code
    python_code = Path(arch_path).read_text().strip()

    # Build performance section based on available data
    performance_section = ""

    if current_latency_ms is not None and baseline_latency_ms is not None:
        # Use latency comparison (more concise)
        speedup = baseline_latency_ms / current_latency_ms if current_latency_ms > 0 else 0
        gap_pct = ((baseline_latency_ms - current_latency_ms) / baseline_latency_ms * 100) if baseline_latency_ms > 0 else 0

        performance_section = f"""# Performance
- **PyTorch baseline**: {baseline_latency_ms:.2f} ms
- **Current Triton**: {current_latency_ms:.2f} ms
- **Current speedup**: {speedup:.2f}x ({gap_pct:+.1f}% vs baseline)
"""
    elif ncu_metrics_block:
        # Use NCU metrics if no latency data
        performance_section = f"""# NCU Metrics
{ncu_metrics_block.strip()}
"""
    else:
        # No performance data available
        performance_section = "# Performance\nNo performance data available. Analyze code structure only.\n"

    # Build prompt
    prompt = algorithm_analysis_tmpl.substitute(
        python_code=python_code,
        cuda_code=cuda_code.strip(),
        performance_section=performance_section,
    )

    return prompt


# -----------------------------------------------------------------------------
# CLI for testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build algorithm analysis prompt")
    parser.add_argument("arch_path", help="Path to PyTorch reference model")
    parser.add_argument("--gpu", default="Quadro RTX 6000", help="GPU name")
    parser.add_argument("--cuda_code", help="Path to current kernel code")
    parser.add_argument("-o", "--out", help="Save prompt to file")

    args = parser.parse_args()

    cuda_code = ""
    if args.cuda_code:
        cuda_code = Path(args.cuda_code).read_text()

    prompt = build_algorithm_analysis_prompt(
        arch_path=Path(args.arch_path),
        gpu_name=args.gpu,
        ncu_metrics_block="# Example metrics\nSM Throughput: 45%\nDRAM Throughput: 78%",
        cuda_code=cuda_code,
    )

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"Prompt saved to {args.out}")
    else:
        print(prompt)
