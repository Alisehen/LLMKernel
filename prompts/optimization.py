from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Optional

from prompts.generate_custom_cuda import (
    MODEL_FUSION,
    MODEL_NETWORK,
    MODEL_SINGLE,
    _load_gpu_spec,
)

_OPTIMIZATION_PROMPT_TEMPLATE = Template(
    """\
You are a CUDA kernel optimization specialist.

# Target GPU
$gpu_section

[CURRENT CANDIDATE]
```python
$arch_src
```

[NCU PROFILING METRICS]
$NCU_METRICS

$STAGE_CONTEXT

## Task
Generate an improved CUDA implementation that is faster on the target GPU while preserving
the original Python API and numerical behavior.

## Output Rules
1. Output a single Python code block only.
2. Use `torch.utils.cpp_extension.load_inline` for custom CUDA code.
3. Follow this order:
   - imports
   - `source` CUDA string(s)
   - `cpp_src` declaration string if needed
   - `load_inline(...)`
   - `class ModelNew(nn.Module)`
4. Do not include tests or extra prose.

## Engineering Requirements
- Maintain correctness and keep the same input/output semantics.
- Preserve CPU fallback behavior when the CUDA path is not applicable.
- Tune for real CUDA concerns: thread/block mapping, register pressure, shared memory, vectorized loads, occupancy, and memory traffic.
- Prefer targeted improvements over rewriting the whole module without evidence.

```python
# <optimized ModelNew code>
```
"""
)

NORMAL_STAGE_FOCUS_MAP = {
    "grid_and_parallel": """
## Current Optimization Stage
Focus on grid/block mapping and overall parallelism.

Prefer changes such as:
- remapping work across batch / sequence / output channels
- improving block and grid dimensions
- increasing occupancy when the GPU is underutilized
- reducing tiny launches or serial regions
""",
    "block_tiling": """
## Current Optimization Stage
Focus on tile sizes and per-block work decomposition.

Prefer changes such as:
- tuning TILE_M / TILE_N / TILE_K or similar tile shapes
- changing thread-block size
- adjusting shared-memory staging strategy
- balancing reuse against register pressure
""",
    "memory_and_tuning": """
## Current Optimization Stage
Focus on memory traffic and final launch tuning.

Prefer changes such as:
- improving coalescing and vectorized access
- reducing redundant global loads/stores
- using shared memory only when reuse justifies it
- adjusting unroll factors, launch bounds, or lightweight caching
""",
}

FUSION_STAGE_FOCUS_MAP = {
    "grid_and_parallel": """
## Current Optimization Stage
Focus on fused-kernel work mapping.

Prefer changes such as:
- sharing one launch over a longer op chain
- aligning thread indexing across fused operations
- minimizing launch overhead and intermediate tensor materialization
""",
    "block_tiling": """
## Current Optimization Stage
Focus on fusion with register pressure control.

Prefer changes such as:
- shrinking block size when fusion causes register pressure
- keeping only expensive intermediates in registers
- recomputing cheap expressions if it reduces spilling
""",
    "memory_and_tuning": """
## Current Optimization Stage
Focus on fusion-specific memory behavior.

Prefer changes such as:
- eliminating intermediate stores
- combining multiple reads into one fused pass
- keeping a single final write whenever possible
""",
}

NETWORK_STAGE_FOCUS_MAP = {
    "grid_and_parallel": """
## Current Optimization Stage
Focus on the main network hotspot rather than every layer.

Preserve module structure and optimize the dominant CUDA path.
""",
    "block_tiling": """
## Current Optimization Stage
Focus on the tile and block geometry of the dominant hotspot kernel.
""",
    "memory_and_tuning": """
## Current Optimization Stage
Focus on end-to-end bandwidth reduction in the hotspot path.
""",
}


def build_optimization_prompt(
    arch_path: Path,
    gpu_name: Optional[str] = None,
    *,
    ncu_metrics: str = "",
    stage_name: str = "",
    stage_description: str = "",
    fusion: bool = False,
    model: str = MODEL_SINGLE,
) -> str:
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
    gpu_section = "\n".join(f"- {k}: {v}" for k, v in info.items())
    arch_src = Path(arch_path).read_text().strip()

    effective_model = MODEL_FUSION if fusion and model == MODEL_SINGLE else model
    if effective_model == MODEL_NETWORK:
        stage_focus_map = NETWORK_STAGE_FOCUS_MAP
    elif effective_model == MODEL_FUSION:
        stage_focus_map = FUSION_STAGE_FOCUS_MAP
    else:
        stage_focus_map = NORMAL_STAGE_FOCUS_MAP

    stage_context = stage_focus_map.get(stage_name, "")
    if stage_description:
        stage_context = f"{stage_context}\nStage label: {stage_description}\n"

    metrics_section = ncu_metrics or "No NCU metrics available."
    return _OPTIMIZATION_PROMPT_TEMPLATE.substitute(
        gpu_section=gpu_section,
        arch_src=arch_src,
        NCU_METRICS=metrics_section,
        STAGE_CONTEXT=stage_context,
    )


def get_stage_ncu_metrics(stage_name: str) -> list[str]:
    core_metrics = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__t_sector_hit_rate.pct",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ]

    stage_metrics = {
        "grid_and_parallel": [
            "launch__grid_size",
            "launch__block_size",
            "launch__waves_per_multiprocessor",
        ],
        "block_tiling": [
            "launch__occupancy_limit_blocks",
            "launch__occupancy_limit_registers",
            "launch__registers_per_thread",
            "launch__shared_mem_per_block_static",
        ],
        "memory_and_tuning": [
            "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
            "l1tex__t_sector_hit_rate.pct",
        ],
    }

    return core_metrics + stage_metrics.get(stage_name, [])
