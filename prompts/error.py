# prompts/error.py
"""
Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
Adds GPU hardware context and architecture source for better fixes.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
from string import Template

# Project roots (adjust if your tree differs)
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Reuse your existing GPU spec loader
from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402

COMPILE_ERROR = Template(
    """Fix the Triton kernel errors. Generate correct code.

## ERROR LOG
```
$ERROR_LOG
```
$ERROR_HISTORY
## Broken Code
```python
$OLD_CODE
```

## CRITICAL — These cause 60%+ of failures:
1. EVERY kernel function MUST have `@triton.jit` decorator — MANDATORY
2. Grid size MUST be > 0: use `triton.cdiv(N, BLOCK)` or `max(1, N // BLOCK)`
3. BLOCK sizes MUST be power-of-2: 16, 32, 64, 128, 256
4. `tl.program_id(axis)` only supports axis = 0, 1, 2
5. No `continue`, `break`, `return` inside loops — use masking
6. No tensor indexing with loop vars: `x[:, i]` is INVALID
7. mask shape MUST match data shape in tl.load/tl.store

## Missing Triton Functions (implement manually):
- tl.tanh, tl.sigmoid, tl.gelu, tl.silu, tl.softmax, tl.mish

## OUTPUT FORMAT (STRICT):
1. Imports: torch, torch.nn, triton, triton.language as tl (and math if needed)
2. @triton.jit decorated kernel function(s)
3. Wrapper function(s) for grid calculation and kernel launch
4. class ModelNew(nn.Module) — REQUIRED

Do NOT include: testing code, if __name__, get_inputs, get_init_inputs

```python
# <corrected code>
```
"""
)


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
    Build the error-repair prompt with error history.

    Parameters
    ----------
    old_code : str
        The broken Python script content to show under OLD CODE.
    error_log : str
        The compiler/runtime error text to show under ERROR LOG.
    problem : Optional[Any]
        Deprecated, kept for backward compatibility. Ignored.
    gpu_name : Optional[str]
        Human-readable GPU name key to lookup in gpu_specs.
    error_history : str
        History of previous repair attempts (concise summaries).
    arch_path : Optional[Path]
        Path to reference PyTorch implementation (currently unused).

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

    # Format error history if provided (concise format to avoid repeating mistakes)
    history_section = ""
    if error_history and error_history.strip():
        history_section = f"""
## Previous Failed Attempts (DO NOT repeat these mistakes):
{error_history.strip()}

"""

    # Substitute all fields
    return COMPILE_ERROR.substitute(
        ERROR_HISTORY=history_section,
        ERROR_LOG=error_log.strip(),
        OLD_CODE=old_code.strip(),
    )
