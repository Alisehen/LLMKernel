from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template
import json
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec  # Adjust import path as needed

_OPTIMIZATION_PROMPT_TEMPLATE = Template("""\
You are a CUDA kernel optimization specialist. Generate the FASTEST possible kernel.

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

OUTPUT RULES (STRICT) ────────────────────────────────────────────────────────────────
1. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple-quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
2. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.

```python
# <optimized CUDA code>
```
""")

def _escape_template(s: str) -> str:
    return s.replace("$", "$$")

def _sanitize_text(s: str) -> str:
    return s.replace("```", "`")

def _format_problem(problem: Optional[Any]) -> str:
    if problem is None or problem == "":
        return "No prior critical problem provided."
    if isinstance(problem, Mapping):
        # Prefer extracting bottleneck / optimisation method / modification plan
        bottleneck = str(problem.get("bottleneck", "")).strip()
        opt_method = str(problem.get("optimisation method", "")).strip()
        mod_plan   = str(problem.get("modification plan", "")).strip()
        if bottleneck or opt_method or mod_plan:
            return (
                "{\n"
                f'  "bottleneck": "{bottleneck}",\n'
                f'  "optimisation method": "{opt_method}",\n'
                f'  "modification plan": "{mod_plan}"\n'
                "}"
            )
        # fallback to JSON dump
        return json.dumps(problem, ensure_ascii=False, indent=2)
    # For other types, convert to string directly
    return str(problem)

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

    # Build stage context
    stage_context = ""
    if stage_name and stage_description:
        # Map stage names to specific optimization focus
        # Suggestions are based on NCU metrics - LLM should evaluate applicability
        stage_focus_map = {
            "grid_and_parallel": """
**NCU Metric**: `sm__maximum_warps_per_active_cycle_pct` (Target: >80%)

**Core Principle**: Maximize SM utilization through proper grid/block configuration.

**Key Actions**:
• Tune block size (64/128/256 threads, must be multiple of 32)
• For 2D problems, use 2D blocks for better memory patterns
• Balance: too small wastes SMs, too large hits resource limits

""",

            "memory_coalescing": """
**NCU Metric**: `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained_elapsed` (Target: >80%)

**Core Principle**: Ensure consecutive threads access consecutive memory addresses.

**CRITICAL RULE**: Within a warp, threadIdx.x must map to contiguous memory offsets.

**Key Patterns to Fix**:
• Strided access → Change to unit-stride access
• Column-major for row-major data → Swap loop order or use threadIdx mapping
• Transpose operations → Use shared memory as staging buffer (pad to avoid bank conflicts)

""",

            "block_tiling": """
**CRITICAL**: Tile size directly affects performance by orders of magnitude!

**⚠️ NCU Metrics Can Be Misleading**:
- If tile is too small (e.g., 16×16), NCU may show:
  ✓ Occupancy 100% (because many small blocks)
  ✓ L2 hit 99% (because all data fits in cache)
  ✗ BUT actual performance is 5-10x SLOWER due to:
    • Excessive kernel launch overhead
    • Each thread does too little work (low compute intensity)

**Correct Decision Process**:
1. Calculate actual TFLOPS from benchmark time
   - For matmul: TFLOPS = 2×M×N×K / (time_ms / 1000) / 1e12
2. If TFLOPS < 20% of GPU peak → Tile size is TOO SMALL
3. Increase tile size even if it lowers occupancy (this is correct!)

**Recommended Tile Sizes** (for matmul):
• Small matrices (N<1024): 32×32
• Medium matrices (1024≤N<4096): 64×64
• Large matrices (N≥4096): 128×128

**Key Insight**: Occupancy 50% with 128×128 tiles is MUCH faster than occupancy 100% with 16×16 tiles!

""",

            "memory_hierarchy": """
**NCU Metrics** (optimize in order):
1. `l1tex__data_bank_conflicts` (Target: <100)
2. `lts__t_sector_hit_rate` (Target: >70%)
3. `smsp__warp_issue_stalled_memory_dependency` (Target: <20%)

**Optimization Priority** (based on NCU):

**IF bank conflicts > 100**: Pad shared memory arrays (+1 dimension) to avoid bank conflicts

**IF L2 hit < 70%**: Use shared memory to increase data reuse and reduce global memory access

**IF memory stalls > 20%**: Add double buffering to hide memory latency (prefetch next tile while computing current)

**IF DRAM > 80% (memory-bound)**: Use vectorized loads/stores (float4) for higher bandwidth

""",
        }

        focus = stage_focus_map.get(stage_name, "")
        stage_context = f"""
## Current Optimization Stage
**Stage**: {stage_description}
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
