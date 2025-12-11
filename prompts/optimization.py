from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template
import json
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec  # Adjust import path as needed
from config.operator_categories_v3 import build_stage_prompt_section

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
    category: str = "Memory-Intensive",
    stage_id: int = 0,
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
        category: Operator category (Compute-Intensive, Memory-Intensive, Fusion-Compute, Fusion-Memory)
        stage_id: Stage index (0-based)
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

    # Build stage context with category-specific guidance
    stage_context = ""
    if stage_name and stage_description:
        # Use category-specific guidance
        try:
            category_guidance = build_stage_prompt_section(category, stage_id)
            if category_guidance:
                stage_context = category_guidance
            else:
                # Simple fallback if no category guidance available
                stage_context = f"""
## Current Optimization Stage
**Stage**: {stage_description}
**Focus**: Optimize based on NCU metrics analysis.
"""
        except Exception as e:
            # Fallback to simple stage description
            print(f"[Warning] Failed to get category guidance: {e}")
            stage_context = f"""
## Current Optimization Stage
**Stage**: {stage_description}
**Focus**: Optimize based on NCU metrics analysis.
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
