# prompts/error.py
"""
Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
Adds GPU hardware context and architecture source for better fixes.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template

# Project roots (adjust if your tree differs)
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Reuse your existing GPU spec loader
from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402

COMPILE_ERROR = Template(
    """Fix the Triton kernel errors. Generate correct, high-performance code.

Current Error Log:
$ERROR_LOG

Main Critical Problem Analysis:
$PROBLEM_ANALYSIS

Broken Code:
```python
$OLD_CODE
```

OUTPUT RULES (STRICT):
1. Follow this exact order:
   1. Imports: torch, torch.nn, triton, triton.language as tl, AND any other modules used (e.g., import math if using math.sqrt)
   2. @triton.jit decorated kernel function(s) — NO continue/break/return inside loops (use masking)
   3. Wrapper function(s) for grid calculation and kernel launch
   4. class ModelNew(nn.Module) that calls your kernels — THIS CLASS IS REQUIRED
2. Do NOT include: testing code, if __name__, get_inputs, get_init_inputs
3. Learn from previous repair attempts to avoid repeating the same mistakes
4. Ensure ALL imports are included at the top (common mistake: forgetting `import math`)

```python
# <corrected code>
```
"""
)

def _escape_template(s: str) -> str:
    return s.replace("$", "$$")

def _sanitize_text(s: str) -> str:
    return s.replace("```", "`")

def _format_problem(problem: Optional[Any]) -> str:
    import json
    if problem is None or problem == "":
        return "No prior critical problem provided."
    if isinstance(problem, Mapping):
        # Prefer to concatenate the three key fields into a concise description; otherwise fall back to JSON
        ci  = str(problem.get("critical_issue", "")).strip()
        wim = str(problem.get("why_it_matters", "")).strip()
        mfh = str(problem.get("minimal_fix_hint", "")).strip()
        if ci or wim or mfh:
            return f"critical_issue: {ci}\nwhy_it_matters: {wim}\nminimal_fix_hint: {mfh}"
        return json.dumps(problem, ensure_ascii=False, indent=2)
    # For other types, simply convert to string
    return str(problem)

def build_error_prompt(
    *,
    old_code: str,
    error_log: str,
    problem: Optional[Any] = None,
    gpu_name: Optional[str] = None,
    error_history: str = "",
    arch_path: Optional[Path] = None,
) -> str:
    """
    Build the error-repair prompt with GPU context + architecture source.

    Parameters
    ----------
    old_code : str
        The broken Python script content to show under OLD CODE.
    error_log : str
        The compiler/runtime error text to show under ERROR LOG.
    problem : Optional[Any]
        The problem analysis from previous step.
    gpu_name : Optional[str]
        Human-readable GPU name key to lookup in gpu_specs.
        If None, attempts torch.cuda.get_device_name(0).
    error_history : str
        History of previous repair attempts and their outcomes.
    arch_path : Optional[Path]
        Path to the reference PyTorch implementation file to display.

    Returns
    -------
    str
        The final prompt string to send to the LLM.
    """
    # Load the GPU spec dictionary
    gpu_info = _load_gpu_spec()

    # Resolve GPU name
    if gpu_name is None:
        try:
            import torch  # local import to avoid hard dependency if CPU-only
            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO (file: {HW_FILE})")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")

    # Detect Triton-specific error patterns and provide targeted guidance

    # Load PyTorch reference code if provided
    pytorch_code = ""
    if arch_path and arch_path.exists():
        pytorch_code = arch_path.read_text(encoding="utf-8").strip()
    else:
        pytorch_code = "# PyTorch reference code not provided"

    # Format error history if provided
    history_section = ""
    if error_history and error_history.strip():
        history_section = f"""Previous Repair Attempts (avoid repeating these errors):
{error_history.strip()}

"""
    else:
        history_section = "None\n"

    # Format problem analysis if provided
    problem_section = ""
    if problem is not None and problem != "":
        formatted_problem = _format_problem(problem)
        problem_section = f"""Problem Analysis (from expert diagnosis):
{formatted_problem}

Focus your fix on addressing the identified critical issue.
"""
    else:
        problem_section = ""

    # Substitute all fields
    return COMPILE_ERROR.substitute(
        ERROR_HISTORY=history_section,
        PROBLEM_ANALYSIS=problem_section,
        PYTORCH_CODE=pytorch_code,
        ERROR_LOG=error_log.strip(),
        OLD_CODE=old_code.strip(),
    )
