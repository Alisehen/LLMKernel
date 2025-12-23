#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt builder for generating optimized kernel based on algorithm analysis results.

This is used in the hybrid strategy: after analyzing a sub-1.0 seed, we generate
an optimized kernel based on the identified bottleneck and optimization strategy.
"""

from string import Template
from textwrap import dedent

__all__ = ["build_optimization_from_analysis_prompt"]

optimization_from_analysis_tmpl = Template(
    dedent(
        """You are optimizing a Triton kernel based on algorithmic analysis.

# PyTorch Reference (Target Behavior)

```python
$pytorch_reference
```

**CRITICAL**: Study the PyTorch code carefully to understand:
- What does `forward()` return? (full output sequence vs final hidden state only)
- What is the computational pattern?
- What are the input/output shapes?

Your optimized kernel MUST match this exact behavior.

---

# Analysis Results

**Bottleneck**: $bottleneck

**Optimization Strategy**: $optimization_method

**Implementation Plan**: $modification_plan

**Expected Speedup**: $expected_speedup

---

# Current Kernel (needs optimization)

```python
$current_kernel
```

---

# Your Task

Implement the optimization strategy above. Focus on the specific bottleneck identified.

## Key Requirements

1. **Preserve correctness**: Maintain the same input/output behavior
2. **Apply the optimization**: Follow the implementation plan exactly
3. **Use valid Triton syntax**:
   - Every kernel MUST have `@triton.jit` decorator
   - Grid size MUST be > 0: use `triton.cdiv(N, BLOCK)` or `max(1, N // BLOCK)`
   - BLOCK sizes MUST be power-of-2: 16, 32, 64, 128, 256
   - No `continue`, `break`, `return` inside kernels (use masking)
   - Prefer `tl.dot(a, b, allow_tf32=True)` for matmul operations

4. **CRITICAL for RNN/GRU/LSTM Persistent Kernels**:
   - Time loop MUST be inside @triton.jit kernel, NOT in Python forward()
   - **HYBRID computation strategy** (CRITICAL for performance):
     * Precompute input-side gates OUTSIDE kernel: `gates_x = (T*B, In) @ W_ih` (ONE large GEMM)
     * INSIDE kernel: only recurrent-side: `for t: gates_h = h @ W_hh` (T small GEMMs)
   - CORRECT (FAST - use this):
     ```python
     # Python forward():
     gates_x_all = x.reshape(T*B, In) @ W_ih + b_ih  # ONE large GEMM
     gates_x_all = gates_x_all.view(T, B, 3*H)
     gru_persistent_kernel[grid](gates_x_all, h0, W_hh, ...)  # Launch ONCE

     @triton.jit
     def gru_persistent_kernel(gates_x_ptr, h_ptr, W_hh_ptr, ...):
         for t in range(T):  # Inside kernel
             gates_x_t = tl.load(gates_x_ptr + t*...)  # Precomputed
             gates_h = h @ W_hh  # Only recurrent GEMM
             h = (1-z)*n + z*h   # Fuse and update
     ```

5. **Output format**:
   - Imports: `import torch, torch.nn as nn, triton, triton.language as tl`
   - `@triton.jit` kernel(s)
   - Wrapper function(s)
   - `class ModelNew(nn.Module)` — REQUIRED
   - NO testing code, NO `if __name__ == "__main__"`

---

Generate the optimized kernel now. Output ONLY the complete Python code.
"""
    )
)

def build_optimization_from_analysis_prompt(
    *,
    bottleneck: str,
    optimization_method: str,
    modification_plan: str,
    expected_speedup: str,
    current_kernel: str,
    pytorch_reference: str,
) -> str:
    """Build prompt for generating optimized kernel from analysis results.

    Args:
        bottleneck: Identified performance bottleneck (from algorithm analysis)
        optimization_method: Recommended optimization strategy (from algorithm analysis)
        modification_plan: Step-by-step implementation plan (from algorithm analysis)
        expected_speedup: Expected performance improvement (from algorithm analysis)
        current_kernel: Current kernel code that needs optimization
        pytorch_reference: PyTorch reference code to understand target behavior

    Returns:
        Complete prompt string for LLM
    """
    prompt = optimization_from_analysis_tmpl.substitute(
        bottleneck=bottleneck,
        optimization_method=optimization_method,
        modification_plan=modification_plan,
        expected_speedup=expected_speedup,
        current_kernel=current_kernel.strip(),
        pytorch_reference=pytorch_reference.strip(),
    )

    return prompt


# -----------------------------------------------------------------------------
# CLI for testing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Build optimization prompt from analysis")
    parser.add_argument("--kernel", required=True, help="Path to current kernel code")
    parser.add_argument("-o", "--out", help="Save prompt to file")

    args = parser.parse_args()

    kernel_code = Path(args.kernel).read_text()

    # Example analysis results
    prompt = build_optimization_from_analysis_prompt(
        bottleneck="Launching 64 separate kernels for GRU time steps causes severe overhead",
        optimization_method="Fuse all time steps into a single persistent kernel",
        modification_plan="1) Create single kernel with time-loop inside; 2) Use shared memory for h_t state; 3) Launch once for entire sequence",
        expected_speedup="800-1000%",
        current_kernel=kernel_code,
    )

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"Prompt saved to {args.out}")
    else:
        print(prompt)
