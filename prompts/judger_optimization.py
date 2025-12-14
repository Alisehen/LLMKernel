#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt builder for **single most impactful optimisation** suggestion.

Use this when you do **not** provide an error log. Instead, supply:
  - NCU metrics block (text/markdown)
  - GPU name (looked up in prompts/hardware/gpu_specs.py)
  - PyTorch reference architecture file (contains `class Model`)
  - (Optional) current CUDA candidate code to inspect

The Judge LLM must return **exactly one** optimisation target with a minimal plan.
"""

from __future__ import annotations
from pathlib import Path
from string import Template
from textwrap import dedent
import importlib.util
import sys
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

__all__ = ["build_single_opt_prompts"]

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
# System prompt: exactly one optimisation target
# -----------------------------------------------------------------------------

from textwrap import dedent
from string import Template

# system_prompt_tmpl = Template(
#     dedent(
#         """
# You are a senior Triton kernel optimization engineer. Read the target GPU spec, the PyTorch
# reference code, the current Triton candidate, and the Nsight Compute
# metrics. Then identify **exactly one** highest-impact speed bottleneck, propose **exactly one** optimisation method and propose a
# modification plan. Be surgical and metrics-driven.

# Rules:
# - Return **one and only one** optimisation method — the largest expected speedup.
# - Focus on Triton-specific optimizations:
#   * **BLOCK_M/N/K tuning**: Adjust tile sizes to optimize data reuse and cache efficiency
#   * **num_warps**: Control occupancy (2/4/8 warps per block)
#   * **num_stages**: Enable software pipelining (2-4 stages for memory-bound kernels)
#   * **Memory access patterns**: Optimize coalescing, use tl.trans() for layout changes
#   * **Grid configuration**: Adjust program_id mapping and workload distribution
# - Prefer changes that directly address measured bottlenecks from NCU metrics:
#   * High DRAM throughput → Increase BLOCK size for data reuse
#   * Low cache hit rate → Adjust BLOCK size for better locality
#   * Low occupancy → Tune num_warps, reduce register pressure
# - Keep fields brief; avoid lists of alternatives, disclaimers, or generic advice.

# Output format (JSON):
# ```json
# {
#   "bottleneck": "<max 30 words>",
#   "optimisation method": "<max 35 words>",
#   "modification plan": "<max 35 words>"
# }
# """
# )
# )

# -----------------------------------------------------------------------------
# Instruction prompt injects code, metrics, GPU spec
# -----------------------------------------------------------------------------

instruction_tmpl = Template(
    dedent(
        """You are a senior Triton kernel optimization engineer. Read the target GPU spec, the PyTorch reference code, the current Triton candidate, and the Nsight Compute metrics. Then identify **exactly one** highest-impact speed bottleneck, propose **exactly one** optimisation method and propose a modification plan. Be surgical and metrics-driven.



# PyTorch Reference
$python_code


# Current Triton Kernel
```python
$CUDA_CODE
```

$STAGE_CONTEXT

# Nsight Compute Metrics (3 Core Metrics)
$NCU_METRICS

$BASELINE_COMPARISON

Rules:
- Return **one and only one** optimisation method — the largest expected speedup.
- Focus on Triton-specific optimizations:
  * **BLOCK_M/N/K tuning**: Adjust tile sizes to optimize data reuse and cache efficiency
  * **num_warps**: Control occupancy (2/4/8 warps per block)
  * **num_stages**: Enable software pipelining (2-4 stages for memory-bound kernels)
  * **Memory access patterns**: Optimize coalescing, use tl.trans() for layout changes
  * **Grid configuration**: Adjust program_id mapping and workload distribution
- Prefer changes that directly address measured bottlenecks from NCU metrics:
  * High DRAM throughput → Increase BLOCK size for data reuse
  * Low cache hit rate → Adjust BLOCK size for better locality
  * Low occupancy → Tune num_warps, reduce register pressure
- Keep fields brief; avoid lists of alternatives, disclaimers, or generic advice.

Output format (JSON):
```json
{
  "bottleneck": "<max 30 words>",
  "optimisation method": "<max 35 words>",
  "modification plan": "<max 35 words>"
}
Read everything and follow the Rules exactly. Return the JSON in the specified format.
"""
    )
)

# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

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
    """Return (system_prompt_str, instruction_str) for single-issue optimisation.

    Args:
        arch_path:   Path to .py that contains the PyTorch reference `class Model`
        gpu_name:    Key in GPU_SPEC_INFO (e.g., "Quadro RTX 6000")
        ncu_metrics_block: Text/Markdown block of Nsight Compute metrics
        cuda_code:   Optional current CUDA candidate source (string)
        stage_name:  Current optimization stage (e.g., "block_tiling")
        stage_description: Description of current stage (e.g., "Block Tiling (BLOCK_M/N/K)")
        baseline_metrics: NCU metrics from previous best kernel (for comparison)
    """
    gpu_info = _load_gpu_spec()
    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    )

    arch_src = Path(arch_path).read_text().strip()

    # Build stage context if provided
    stage_context = ""
    if stage_name and stage_description:
        stage_context = f"""
## Current Optimization Stage
**Stage**: {stage_description}
**Focus**: This kernel attempted to optimize **{stage_name}** but performance degraded.

**Stage-Specific Analysis Required**:
- Identify which aspect of {stage_name} optimization failed
- Suggest alternative approaches within this stage's scope
- Provide specific parameter adjustments for {stage_name}
"""

    # Build baseline comparison if provided
    baseline_comparison = ""
    if baseline_metrics:
        baseline_comparison = f"""
## Baseline Metrics (Previous Best Kernel)
{baseline_metrics}

**Task**: Compare current metrics with baseline to identify regression.
- Which metrics got worse?
- What changed during {stage_description} optimization?
- How to recover or improve upon baseline?
"""

    # system_prompt = system_prompt_tmpl.substitute()
    instruction = instruction_tmpl.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        python_code=arch_src,
        CUDA_CODE=cuda_code.strip(),
        STAGE_CONTEXT=stage_context,
        NCU_METRICS=ncu_metrics_block.strip(),
        BASELINE_COMPARISON=baseline_comparison,
    )
    return instruction