from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template
import json

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec  # Adjust import path as needed

_OPTIMIZATION_PROMPT_TEMPLATE = Template("""\
You are a Triton kernel optimization specialist. Generate the FASTEST possible kernel.

# Target GPU
GPU Name: $gpu_name
Architecture: $gpu_arch
$gpu_items

[OPTIMIZATION STAGE]
$STAGE_CONTEXT

[CURRENT CODE]
```python
$arch_src
```

[NCU PROFILING METRICS]
$NCU_METRICS

**Task**: Analyze the NCU metrics and current code, then generate optimized code that maximizes performance.

OUTPUT RULES (STRICT):
1. Follow this exact order:
   1. Imports: torch, torch.nn, triton, triton.language as tl
   2. @triton.jit decorated kernel function(s)
   3. Wrapper function(s) for grid calculation and kernel launch
   4. class ModelNew(nn.Module) that calls your kernels
2. Do NOT include: testing code, if __name__, get_inputs, get_init_inputs

```python
# <optimized Triton code>
```
""")


def build_optimization_prompt(
    arch_path: Path,
    gpu_name: Optional[str] = None,
    *,
    ncu_metrics: str = "",
    history_block: str = "",
    stage_name: str = "",
    stage_description: str = "",
    failure_analysis: str = "",
) -> str:
    """Build single-phase optimization prompt with NCU metrics.

    Args:
        arch_path: Path to kernel code to optimize
        gpu_name: Target GPU name
        ncu_metrics: NCU profiling metrics (JSON format)
        history_block: Previous kernel attempts
        stage_name: Current optimization stage
        stage_description: Stage description
        failure_analysis: Analysis of previous failures
    """
    gpu_info = _load_gpu_spec()

    if gpu_name is None:
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    gpu_items = "\n".join(f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture")

    arch_src = Path(arch_path).read_text().strip()
    hist = history_block or "(None)\n"

    # Build stage context with enhanced guidance
    stage_context = ""
    if stage_name and stage_description:
        # Universal stage optimization focus (适用于所有算子类型)
        stage_focus_map = {
    "grid_and_parallel": """
**Focus**: Optimize grid layout and parallel work distribution.

**NCU Metrics**:
• `sm__throughput.avg.pct_of_peak_sustained_elapsed` (target >60%)
• `launch__grid_size` (primarily determined by grid mapping)

**Guidelines**:
- 1D → `(cdiv(N, BLOCK))`
- 2D → `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`
- 3D (batch×M×N) → `(batch, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`
- 4D+ → flatten **independent parallel dims** to 3D
- Prefer batch / head / expert parallelism when available, before shrinking BLOCK sizes
- Change grid only if SM utilization is clearly low
- Ensure `grid = (...)` matches `tl.program_id(axis)` logic

**Safety Rules**:
- Max 3 grid dimensions; grid rank must be static
- If unsure about correctness, do not modify grid

**Autotune (safe)**:
- Autotune **either** BLOCK_* **or** (`num_warps`, `num_stages`)
- If BLOCK_* are autotuned, use `grid = lambda META: (...)` with `META["BLOCK_*"]`
- Never redefine BLOCK_* in both configs and launch
""",

    "block_tiling": """
**Focus**: Tune BLOCK_M/N/K sizes for balanced data reuse and resource usage.

**NCU Metrics to Check**:
• `sm__warps_active.avg.pct_of_peak_sustained_active`: Warp occupancy (target: >50%)

**Guidelines**:
- Tensor Cores: BLOCK_M/N multiple of 16; BLOCK_K multiple of 8.
- FP32: BLOCK_M/N in [32, 64, 128, 256], BLOCK_K in [16, 32, 64].
- **All BLOCK_SIZE/BLOCK_* must be powers of 2** (16, 32, 64, 128, 256, 512, 1024).
- Avoid oversized tiles (wasted masking); change BLOCK_* only when occupancy or reuse metrics indicate issues.
- If BLOCK sizes are unclear, keep the baseline tile as a valid candidate.

**Autotune Tip (safe)**:
Use a small autotune list (2–4 configs), all being powers of two, and **decorating the @triton.jit kernel**, never the wrapper.
""",

    "memory_access": """
**Focus**: Optimize memory access patterns and latency hiding.

**NCU Metrics to Check**:
• `dram__throughput.avg.pct_of_peak_sustained_elapsed`
• `lts__t_sector_hit_rate.pct`
• `smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct` (target: <20%)

**Guidelines**:
- Increase `num_stages` only if memory stalls are high; otherwise keep it small.
- Improve locality or coalescing only when metrics show issues; avoid unnecessary rewrites.
- Larger BLOCK_K increases reuse but can raise register pressure; adjust only when metrics justify it.

**Autotune Tip (safe)**:
If unsure, try a tiny set varying `num_stages` (e.g., {1, 2, 3}) via autotune on the **kernel**.
""",

    "advanced_memory": """
**Focus**: Final micro-optimizations (small adjustments only).

**Parameters to Tune**:
• `num_warps`: {2, 4, 8, 16}
• `num_stages`: {2, 3, 4}

**Guidelines**:
- Adjust num_warps only if occupancy suggests it; otherwise keep the original.
- Change num_stages only by ±1 from current.
- Do not modify BLOCK sizes or grid mapping here.

**Autotune Tip (safe)**:
Use 3–6 configs around the current settings; always include the original kernel config, and autotune must wrap the JIT kernel.
"""
}

        focus = stage_focus_map.get(stage_name, "")
        stage_context = f"""
## Current Optimization Stage
{focus}
"""

    # Build failure analysis
    failure_context = ""
    if failure_analysis:
        failure_context = f"""
## Previous Attempt Failed
{failure_analysis}

**Your Task**: Generate a **different** approach within {stage_description} scope.
- Do NOT repeat the failed strategy
- Try alternative parameters or implementation within this stage
- Ensure the fix addresses the root cause identified above
"""

    # Format NCU metrics
    ncu_section = ncu_metrics if ncu_metrics else "No NCU metrics available"

    return _OPTIMIZATION_PROMPT_TEMPLATE.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        arch_src=arch_src,
        history_block=hist,
        STAGE_CONTEXT=stage_context,
        NCU_METRICS=ncu_section,
        FAILURE_ANALYSIS=failure_context,
    )


# ============================================================================
# NCU Metrics Configuration for Each Stage
# ============================================================================

def get_stage_ncu_metrics(stage_name: str) -> list[str]:
    """Get NCU metrics to collect for a specific optimization stage.

    Returns list of metric names to pass to NCU profiler.
    """
    # Core metrics that should always be collected (for baseline comparison)
    core_metrics = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # SM compute utilization
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",  # Memory bandwidth
        "lts__t_sector_hit_rate.pct",  # L2 cache hit rate
        "sm__warps_active.avg.pct_of_peak_sustained_active",  # Warp occupancy
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",  # Memory coalescing
    ]

    # Stage-specific metrics
    stage_metrics = {
        "grid_and_parallel": [
            "launch__grid_size",
            "launch__block_size",
            "launch__waves_per_multiprocessor",
        ],
        "block_tiling": [
            "launch__occupancy_limit_blocks",
            "launch__occupancy_limit_registers",
            "launch__occupancy_limit_shared_mem",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_static",
        ],
        "memory_access": [
            "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
            "l1tex__t_sector_hit_rate.pct",
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
        ],
        "advanced_memory": [
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
        ],
    }

    # Combine core + stage-specific metrics
    return core_metrics + stage_metrics.get(stage_name, [])


