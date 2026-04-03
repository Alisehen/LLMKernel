"""Prompt builder for CUDA compile/runtime repair."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any, Optional

from prompts.generate_custom_cuda import _load_gpu_spec

COMPILE_ERROR = Template(
    """You are a senior CUDA extension engineer.

## Target GPU
$GPU_SECTION

## Error Log
```text
$ERROR_LOG
```

$ERROR_HISTORY
## Broken Candidate
```python
$OLD_CODE
```

## Repair Task
Fix the compilation or runtime error and return a corrected CUDA implementation.

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

## Requirements
- Keep the same public behavior as the original candidate.
- Fix the specific failure shown in the error log.
- Validate tensor assumptions where needed.
- Keep CPU fallback behavior when useful.
- If the failure comes from invalid CUDA launch parameters, correct the launch geometry explicitly.

```python
# <corrected ModelNew code>
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
    del problem, arch_path

    gpu_info = _load_gpu_spec()
    if gpu_name is None:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    gpu_section = "\n".join(f"- {k}: {v}" for k, v in gpu_info[gpu_name].items())

    history_block = ""
    if error_history.strip():
        history_block = (
            "## Previous Failed Attempts\n"
            f"{error_history.strip()}\n\n"
        )

    return COMPILE_ERROR.substitute(
        GPU_SECTION=gpu_section,
        ERROR_LOG=error_log.strip(),
        ERROR_HISTORY=history_block,
        OLD_CODE=old_code.strip(),
    )
