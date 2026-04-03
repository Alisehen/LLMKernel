#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prompt builder for generating an optimized CUDA kernel from analysis results."""

from string import Template
from textwrap import dedent

__all__ = ["build_optimization_from_analysis_prompt"]

optimization_from_analysis_tmpl = Template(
    dedent(
        """You are optimizing a CUDA implementation based on algorithmic analysis.

# PyTorch Reference (Target Behavior)
```python
$pytorch_reference
```

Study the PyTorch code carefully. Your generated implementation must preserve the same forward behavior, shapes, and semantics.

# Analysis Results
- Bottleneck: $bottleneck
- Optimization Strategy: $optimization_method
- Implementation Plan: $modification_plan
- Expected Speedup: $expected_speedup

# Current Kernel
```python
$current_kernel
```

# Your Task
Implement the strategy above and return the updated CUDA extension code.

## Requirements
1. Preserve correctness.
2. Return one complete Python module.
3. Use valid CUDA code that compiles via PyTorch inline extension tooling.
4. Keep launch configuration, bounds checks, and layout handling explicit.
5. Use `C10_CUDA_KERNEL_LAUNCH_CHECK()` after manual kernel launches.
6. Keep `class ModelNew(nn.Module)`.
7. Do not include testing code or `if __name__ == "__main__"`.

Output only the complete Python code.
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
    """Build prompt for generating optimized kernel from analysis results."""
    return optimization_from_analysis_tmpl.substitute(
        bottleneck=bottleneck,
        optimization_method=optimization_method,
        modification_plan=modification_plan,
        expected_speedup=expected_speedup,
        current_kernel=current_kernel.strip(),
        pytorch_reference=pytorch_reference.strip(),
    )
