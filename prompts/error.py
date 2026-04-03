# prompts/error.py
"""Prompt template for automatic CUDA kernel repair."""
from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402

COMPILE_ERROR = Template(
    """Fix the CUDA kernel or inline extension errors. Generate correct code.

## ERROR LOG
```
$ERROR_LOG
```
$ERROR_HISTORY
## Broken Code
```python
$OLD_CODE
```

## Requirements
1. Return one complete Python module.
2. Use PyTorch's inline CUDA extension flow (`load_inline`) or equivalent valid CUDA extension code.
3. Keep `class ModelNew(nn.Module)` and preserve the target behavior.
4. Every CUDA kernel launch must use valid grid/block dimensions and bounds checks.
5. Use contiguous inputs or make explicit contiguous copies before launching when needed.
6. Check launch failures with `C10_CUDA_KERNEL_LAUNCH_CHECK()` when manually launching kernels.
7. Fix the root cause shown in the error log instead of rewriting unrelated parts.

## Common CUDA Failure Sources
- Wrong tensor dtype/device assumptions in `data_ptr<T>()`
- Shape/stride mismatch between Python wrapper and CUDA kernel
- Missing bounds guards
- Invalid grid/block/shared-memory configuration
- Missing declarations or exported functions in `load_inline`
- Incorrect use of contiguous tensors or transposed layouts

## OUTPUT FORMAT (STRICT)
```python
# <corrected CUDA extension code>
```

Do NOT include testing code, `if __name__ == "__main__"`, `get_inputs`, or `get_init_inputs`.
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
    """Build the error-repair prompt with error history."""
    gpu_info = _load_gpu_spec()

    if gpu_name is None:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO (file: {HW_FILE})")

    history_section = ""
    if error_history and error_history.strip():
        history_section = f"""
## Previous Failed Attempts (DO NOT repeat these mistakes)
{error_history.strip()}

"""

    return COMPILE_ERROR.substitute(
        ERROR_HISTORY=history_section,
        ERROR_LOG=error_log.strip(),
        OLD_CODE=old_code.strip(),
    )
