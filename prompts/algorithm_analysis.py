#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prompt builder for high-level CUDA algorithm analysis."""

from __future__ import annotations

from pathlib import Path
from string import Template
from textwrap import dedent

__all__ = ["build_algorithm_analysis_prompt"]

algorithm_analysis_tmpl = Template(
    dedent(
        """You are a GPU kernel optimization architect. Analyze the implementation and identify one high-level algorithmic optimisation.

# PyTorch Reference
```python
$python_code
```

# Current CUDA Kernel
```python
$cuda_code
```

$performance_section

## Analysis Steps
1. Count kernels and identify the dominant operators.
2. Diagnose the main bottleneck from code structure and performance data.
3. Propose the single highest-value algorithmic change only if it is worth the complexity.

## Optimisation Categories
### 1. Operator Fusion
Fuse consecutive hot-path ops to reduce launches and intermediate traffic.

### 2. Algorithm Replacement
Replace a naive implementation with a better CUDA-oriented algorithm.

### 3. Kernel Launch Reduction
Collapse repeated tiny launches into a persistent or batched implementation when appropriate.

### 4. Memory/Layout Optimisation
Change layout, staging, or reuse strategy to reduce traffic and improve locality.

## Output (JSON)
```json
{
  "worth_optimizing": "yes/no",
  "reason": "<1 sentence>",
  "bottleneck": "<1-2 sentences, empty if not worth optimizing>",
  "optimisation method": "<1-2 sentences, empty if not worth optimizing>",
  "modification plan": "<2-3 sentences, empty if not worth optimizing>",
  "expected_speedup": "<e.g. 30-40%, empty if not worth optimizing>"
}
```

Return JSON only.
"""
    )
)


def build_algorithm_analysis_prompt(
    *,
    arch_path: Path,
    gpu_name: str,
    cuda_code: str,
    ncu_metrics_block: str = "",
    current_latency_ms: float | None = None,
    baseline_latency_ms: float | None = None,
) -> str:
    """Build algorithm analysis prompt for high-level optimization."""
    python_code = Path(arch_path).read_text().strip()

    if current_latency_ms is not None and baseline_latency_ms is not None:
        speedup = baseline_latency_ms / current_latency_ms if current_latency_ms > 0 else 0
        gap_pct = (
            (baseline_latency_ms - current_latency_ms) / baseline_latency_ms * 100
            if baseline_latency_ms > 0
            else 0
        )
        performance_section = f"""# Performance
- PyTorch baseline: {baseline_latency_ms:.2f} ms
- Current CUDA candidate: {current_latency_ms:.2f} ms
- Current speedup: {speedup:.2f}x ({gap_pct:+.1f}% vs baseline)
"""
    elif ncu_metrics_block:
        performance_section = f"""# NCU Metrics
{ncu_metrics_block.strip()}
"""
    else:
        performance_section = "# Performance\nNo performance data available. Analyze code structure only.\n"

    return algorithm_analysis_tmpl.substitute(
        python_code=python_code,
        cuda_code=cuda_code.strip(),
        performance_section=performance_section,
    )
