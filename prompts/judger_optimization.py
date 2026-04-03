#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prompt builder for single most impactful CUDA optimisation suggestion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from string import Template
from textwrap import dedent
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"


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


instruction_tmpl = Template(
    dedent(
        """You are a senior CUDA kernel optimization engineer. Read the target GPU spec, the PyTorch reference code, the current CUDA candidate, and the Nsight Compute metrics. Then identify exactly one highest-impact speed bottleneck, propose exactly one optimisation method, and propose a concise modification plan.

# PyTorch Reference
$python_code

# Current CUDA Kernel
```python
$CUDA_CODE
```

$STAGE_CONTEXT

# Nsight Compute Metrics
$NCU_METRICS

$BASELINE_COMPARISON

Rules:
- Return one and only one optimisation method.
- Focus on CUDA-specific changes such as launch geometry, tiling, shared memory, vectorized access, register pressure, and memory coalescing.
- Prefer changes directly supported by the metrics.
- Keep fields brief and concrete.

Output format (JSON):
```json
{
  "bottleneck": "<max 30 words>",
  "optimisation method": "<max 35 words>",
  "modification plan": "<max 35 words>"
}
```

Return only the JSON block.
"""
    )
)


def build_judger_optimization_prompts(
    *,
    arch_path: Path,
    gpu_name: str,
    ncu_metrics_block: str,
    cuda_code: str = "",
    stage_name: str = "",
    stage_description: str = "",
    baseline_metrics: str = "",
) -> Tuple[str, str]:
    """Return the instruction prompt for single-issue optimisation."""
    gpu_info = _load_gpu_spec()
    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    arch_src = Path(arch_path).read_text().strip()

    stage_context = ""
    if stage_name and stage_description:
        stage_context = f"""
## Current Optimization Stage
Stage: {stage_description}
Focus on why the recent change around `{stage_name}` degraded or underperformed.
"""

    baseline_comparison = ""
    if baseline_metrics:
        baseline_comparison = f"""
## Baseline Metrics (Previous Best Kernel)
{baseline_metrics}

Compare the current metrics with the baseline and explain the biggest regression through your chosen optimisation.
"""

    instruction = instruction_tmpl.substitute(
        python_code=arch_src,
        CUDA_CODE=cuda_code.strip(),
        STAGE_CONTEXT=stage_context,
        NCU_METRICS=ncu_metrics_block.strip(),
        BASELINE_COMPARISON=baseline_comparison,
    )
    return instruction
