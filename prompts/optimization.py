from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec, MODEL_FUSION, MODEL_NETWORK, MODEL_SINGLE

_OPTIMIZATION_PROMPT_TEMPLATE = Template("""\
You are a CUDA kernel optimization specialist. Generate the fastest correct implementation.

# Target GPU: $gpu_name

[OPTIMIZATION STAGE]
$STAGE_CONTEXT

[CURRENT CODE]
```python
$arch_src
```

[NCU PROFILING METRICS]
$NCU_METRICS

Task: analyze the current code and metrics, then return optimized CUDA extension code.

## Requirements
1. Return one complete Python file.
2. Keep the code compatible with PyTorch's inline CUDA extension workflow.
3. Preserve `ModelNew` behavior and the target model's numerical semantics.
4. Use valid launch configuration, bounds checks, and explicit handling of layouts/contiguity.
5. Add `C10_CUDA_KERNEL_LAUNCH_CHECK()` after manual launches.
6. Prefer surgical changes that directly address the measured bottleneck.

## Output Format
```python
# <optimized CUDA extension code>
```

Do NOT include testing code, `if __name__ == "__main__"`, `get_inputs`, or `get_init_inputs`.
""")

NORMAL_STAGE_FOCUS_MAP = {
    "grid_and_parallel": """
Focus: launch geometry and workload distribution.

Metrics:
- sm__throughput.avg.pct_of_peak_sustained_elapsed
- launch__grid_size
- sm__warps_active.avg.pct_of_peak_sustained_active

Actions:
- Remap work to 1D/2D/3D blocks more evenly.
- Expose batch/head/group parallelism before shrinking tiles.
- Adjust blockDim/gridDim to keep more SMs busy.
- Flatten independent dimensions when output rank is higher than launch rank.
""",
    "block_tiling": """
Focus: tile sizes, shared memory, and register pressure.

Metrics:
- launch__registers_per_thread
- sm__warps_active.avg.pct_of_peak_sustained_active
- l1tex__data_bank_conflicts*

Actions:
- Tune tile shapes and per-block work.
- Use shared memory tiling when reuse is high.
- Reduce tile sizes if register pressure or spills are high.
- Consider `__launch_bounds__`, `#pragma unroll`, and vector width choices.
""",
    "memory_and_tuning": """
Focus: memory traffic and final tuning.

Metrics:
- dram__throughput.avg.pct_of_peak_sustained_elapsed
- lts__t_sector_hit_rate.pct
- smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct

Actions:
- Improve coalescing and contiguous access.
- Reduce redundant global loads/stores.
- Use shared memory or register reuse only where metrics justify it.
- Tune unrolling, vectorized loads/stores, and staging depth carefully.
""",
}

FUSION_STAGE_FOCUS_MAP = {
    "grid_and_parallel": """
Focus: mapping fused outputs to a single launch pattern.

Rules:
- All fused operations must agree on the same output indexing.
- Keep intermediate values in registers/shared memory whenever possible.
- Do not split fusion across incompatible index spaces.
""",
    "block_tiling": """
Focus: balancing fusion benefit vs register pressure.

Rules:
- Fusion raises register usage; avoid aggressive tiles that spill.
- Keep expensive intermediates, recompute cheap arithmetic if it reduces pressure.
- Prefer conservative tiles when multiple inputs are active.
""",
    "memory_and_tuning": """
Focus: eliminate intermediate traffic while keeping loads efficient.

Rules:
- Multiple input loads are fine; intermediate global stores are not.
- Keep a single final global store per output tensor when possible.
- Tune vector width, unrolling, and shared-memory use only if metrics support it.
""",
}

NETWORK_STAGE_FOCUS_MAP = FUSION_STAGE_FOCUS_MAP


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
    """Build optimization prompt with NCU metrics."""
    gpu_info = _load_gpu_spec()

    if gpu_name is None:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    arch_src = Path(arch_path).read_text().strip()

    effective_model = model
    if fusion and model == MODEL_SINGLE:
        effective_model = MODEL_FUSION

    if effective_model == MODEL_NETWORK:
        stage_focus_map = NETWORK_STAGE_FOCUS_MAP
    elif effective_model == MODEL_FUSION:
        stage_focus_map = FUSION_STAGE_FOCUS_MAP
    else:
        stage_focus_map = NORMAL_STAGE_FOCUS_MAP

    stage_context = ""
    if stage_name and stage_description:
        stage_focus = stage_focus_map.get(stage_name, "")
        stage_context = f"Stage: {stage_description}\n\n{stage_focus}".strip()

    return _OPTIMIZATION_PROMPT_TEMPLATE.substitute(
        gpu_name=gpu_name,
        STAGE_CONTEXT=stage_context or "General optimization pass.",
        arch_src=arch_src,
        NCU_METRICS=ncu_metrics.strip() or "No NCU metrics available.",
    )
