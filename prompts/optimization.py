from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template
import json

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec, MODEL_SINGLE, MODEL_FUSION, MODEL_NETWORK

_OPTIMIZATION_PROMPT_TEMPLATE = Template("""\
You are a Triton kernel optimization specialist. Generate the FASTEST possible kernel.

# Target GPU: $gpu_name

[OPTIMIZATION STAGE]
$STAGE_CONTEXT

[CURRENT CODE]
```python
$arch_src
```

[NCU PROFILING METRICS]
$NCU_METRICS

**Task**: Analyze the NCU metrics and current code, then generate optimized code that maximizes performance.

## CRITICAL — Code MUST compile and run:
1. EVERY kernel function MUST have `@triton.jit` decorator
2. Grid size MUST be > 0: use `triton.cdiv(N, BLOCK)` or `max(1, N // BLOCK)`
3. BLOCK sizes MUST be power-of-2: 16, 32, 64, 128, 256
4. `tl.program_id(axis)` only supports axis = 0, 1, 2
5. No `continue`, `break`, `return` inside loops — use masking
6. No tensor indexing with loop vars: `x[:, i]` is INVALID
7. mask shape MUST match data shape in tl.load/tl.store

## Missing Triton Functions (implement manually):
- tl.tanh, tl.sigmoid, tl.gelu, tl.silu, tl.softmax, tl.mish

## OUTPUT FORMAT (STRICT):
1. Imports: torch, torch.nn, triton, triton.language as tl
2. @triton.jit decorated kernel function(s)
3. Wrapper function(s) for grid calculation and kernel launch
4. class ModelNew(nn.Module) that calls your kernels

Do NOT include: testing code, if __name__, get_inputs, get_init_inputs

```python
# <optimized Triton code>
```
""")

# ============================================================================
# Stage Focus Maps - Normal vs Fusion
# ============================================================================

NORMAL_STAGE_FOCUS_MAP = {
    "grid_and_parallel": """
Focus: Grid layout & parallelism.

Metrics:
- sm__throughput.avg.pct_of_peak_sustained_elapsed (>60%)
- launch__grid_size

Rules:
- 1D: (cdiv(N, BLOCK))
- 2D: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
- 3D: (batch, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
- >3D: flatten ONLY independent dims
- Prefer batch / head / expert / group parallelism before shrinking BLOCK
- For grouped operations: ensure group dimension is in grid (e.g., program_id(2) for groups)
- Change grid only if SM utilization is clearly low

Safety:
- Max 3 grid dims, static rank
- grid=(G0,G1,G2) must match tl.program_id(0/1/2)
- For grouped ops: verify group indexing is correct
- If unsure about correctness, do NOT change grid

Autotune:
- Autotune either BLOCK_* OR (num_warps, num_stages)
- If autotuning BLOCK_*, use grid=lambda META: (...)
- Never redefine BLOCK_* in both kernel and launch
- Max 2-3 configs to reduce compilation time
""",

    "block_tiling": """
Focus: BLOCK_M/N/K selection.

Metrics:
- sm__warps_active.avg.pct_of_peak_sustained_active (>50%)

Rules:
- BLOCK_* must be powers of 2
- Tensor Core: BLOCK_M/N multiple of 16, BLOCK_K multiple of 8 (preference)
- FP32: M/N ∈ {32,64,128,256}, K ∈ {16,32,64}
- Avoid oversized tiles (mask waste)
- Keep baseline tile if unsure

Autotune:
- Max 2-3 configs to reduce compilation time
- Autotune ONLY on @triton.jit kernel
""",

    "memory_and_tuning": """
Focus: Memory optimization and final parameter tuning.

Metrics:
- dram__throughput.avg.pct_of_peak_sustained_elapsed
- lts__t_sector_hit_rate.pct
- smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct (<20%)
- sm__warps_active.avg.pct_of_peak_sustained_active

Parameters to tune:
- num_stages ∈ {2, 3, 4}
- num_warps ∈ {4, 8} (based on occupancy)

Rules:
- Increase num_stages only if memory stalls > 20%
- Change num_warps only if occupancy suggests it
- Larger BLOCK_K improves reuse but increases register pressure
- Do NOT modify grid or BLOCK sizes (fixed in earlier stages)
- Do not rewrite access patterns without metric evidence

Autotune:
- Max 3-4 configs combining num_stages and num_warps
- Always include original config as baseline
- Revert if gain < 2% or unstable
"""
}

FUSION_STAGE_FOCUS_MAP = {
    "grid_and_parallel": """
Focus: Grid layout & indexing for FUSED operations.

⚠️ FUSION EXCLUSIONS (do NOT apply fusion rules to these):
- Reduction ops (sum, mean, softmax along axis)
- Atomic operations
- Irregular/data-dependent access patterns
- Cross-block dependencies

Key Principle:
- All fused ops share the SAME grid AND the SAME (offsets, mask) tuple
- Grid covers OUTPUT tensor dimensions

Hard Rules:
- Every fused op MUST use identical offset calculation
- Every fused op MUST use identical boundary mask
- If broadcast needed: explicit `[None, :]` or `[:, None]`, NOT different offsets
- Element-wise: 1D grid, single `offs = pid * BLOCK + tl.arange(0, BLOCK)`
- Matmul fusion: 2D grid, `offs_m/offs_n` shared by bias add & activation

Verification:
- Check: all tl.load/tl.store use same `offsets` variable
- Check: all masks derived from same boundary condition
- If ANY op needs different indexing → do NOT fuse, split kernel
""",

    "block_tiling": """
Focus: BLOCK_SIZE with register pressure awareness.

Key Principle:
- Fusion increases register usage (intermediates stay in registers)
- Spill to local memory kills fusion benefit

Register Pressure Signals (from NCU):
- launch__registers_per_thread > 128 → likely spilling
- launch__occupancy_limit_registers < other limits → register-bound

Rules:
- Start conservative: BLOCK_SIZE ∈ {256, 512} for element-wise
- For matmul fusion: BLOCK_M/N ∈ {32, 64}, BLOCK_K ∈ {32}
- If registers > 128: reduce BLOCK_* by half
- Trade-off: recompute cheap ops (e.g., x*0.5) vs store intermediate

When to Recompute vs Keep:
- Keep: expensive ops (exp, log, div, sqrt)
- Recompute: cheap ops (add, mul, max) if register pressure high
- Example: `y = relu(x); z = y * scale` → keep y
- Example: `y = x * 0.5; z = y + bias` → can recompute y if needed

Autotune:
- 2-3 BLOCK_SIZE configs, always include smaller fallback
""",

    "memory_and_tuning": """
Focus: Memory pattern and parameter tuning for fused operations.

Key Principle:
- Fusion benefit = eliminated INTERMEDIATE stores
- Multiple input loads are OK; intermediate stores are NOT

Memory Rules:
- ✅ Multiple tl.load() for different inputs (x, weight, bias) - OK
- ❌ tl.store() for intermediate results - NEVER (this is what fusion eliminates)
- ✅ Single tl.store() for final output - required

Verification:
- Count tl.store() calls: should equal number of OUTPUT tensors (usually 1)
- Intermediate values: must stay in registers between ops
- If you see store-then-load pattern for same data → BUG, refactor

Parameters to tune:
- num_warps ∈ {4, 8}
- num_stages ∈ {2, 3}

Conditional Tuning Rules:

IF register pressure LOW (regs < 96, no spill):
  - Try num_warps=8 for compute-bound fusion
  - num_stages=3 may help hide latency

IF register pressure HIGH (regs > 128 or occupancy_limit_registers):
  - Use num_warps=4 (fewer warps = more registers per warp)
  - Keep num_stages=2 (higher stages need more registers)

IF multi-input fusion (3+ distinct loads):
  - num_stages=2 preferred (each stage buffers all inputs)
  - num_warps=4 often better than 8

Autotune:
- Max 3-4 configs combining num_stages and num_warps
- Always include conservative baseline (num_warps=4, num_stages=2)
- Revert if gain < 2%
"""
}

# Network model (level3) - use same stages as fusion
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
    """Build optimization prompt with NCU metrics.

    Args:
        arch_path: Path to kernel code to optimize
        gpu_name: Target GPU name
        ncu_metrics: NCU profiling metrics (JSON format)
        stage_name: Current optimization stage
        stage_description: Stage description
        fusion: Whether this is a fusion operator (multi-op kernel)
        model: Model type - "single" (level1), "fusion" (level2), or "network" (level3)
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

    arch_src = Path(arch_path).read_text().strip()

    # Determine effective model type (fusion param for backward compatibility)
    effective_model = model
    if fusion and model == MODEL_SINGLE:
        effective_model = MODEL_FUSION

    # Select stage focus map based on model type
    if effective_model == MODEL_NETWORK:
        stage_focus_map = NETWORK_STAGE_FOCUS_MAP
    elif effective_model == MODEL_FUSION:
        stage_focus_map = FUSION_STAGE_FOCUS_MAP
    else:
        stage_focus_map = NORMAL_STAGE_FOCUS_MAP

    # Build stage context
    stage_context = ""
    if stage_name and stage_description:
        focus = stage_focus_map.get(stage_name, "")
        stage_context = f"""
## Current Optimization Stage
{focus}
"""

    # Format NCU metrics - only show GPU hardware info when NCU metrics unavailable
    if ncu_metrics:
        ncu_section = ncu_metrics
    else:
        gpu_items = "\n".join(f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture")
        ncu_section = f"No NCU metrics available.\n\nGPU Hardware Info:\n{gpu_items}"

    return _OPTIMIZATION_PROMPT_TEMPLATE.substitute(
        gpu_name=gpu_name,
        arch_src=arch_src,
        STAGE_CONTEXT=stage_context,
        NCU_METRICS=ncu_section,
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
        "memory_and_tuning": [
            "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
            "l1tex__t_sector_hit_rate.pct",
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
            "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
        ],
    }

    # Combine core + stage-specific metrics
    return core_metrics + stage_metrics.get(stage_name, [])


