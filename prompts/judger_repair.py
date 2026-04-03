# prompts/judger_repair.py
"""Prompt template for CUDA kernel correctness analysis."""
from __future__ import annotations

from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402

unified_prompt_tmpl = Template("""You are a CUDA kernel debugging expert. Analyze the error and identify the root cause.

## ERROR LOG
```
$ERROR_LOG
```

## Expected Behavior (PyTorch Reference)
```python
$PYTORCH_CODE
```

## Current Implementation (Broken CUDA Kernel)
```python
$CUDA_CODE
```

## Your Task
Identify the single most critical issue that causes the error above.

### Analysis Guidelines
1. Focus on root cause, not symptoms.
2. Be specific about what is wrong and where it happens.
3. Prioritize correctness issues over performance issues.

### Output Format
Return only valid JSON:

```json
{
  "critical_issue": "<max 30 words>",
  "why_it_matters": "<max 35 words>",
  "minimal_fix_hint": "<max 30 words>"
}
```
""")


def build_correctness_prompts(
    *,
    error_log: str,
    arch_path: Path,
    cuda_code: str,
) -> str:
    """Build unified prompt for kernel correctness analysis."""
    pytorch_code = Path(arch_path).read_text(encoding="utf-8").strip()

    return unified_prompt_tmpl.substitute(
        ERROR_LOG=error_log.strip(),
        PYTORCH_CODE=pytorch_code,
        CUDA_CODE=cuda_code.strip(),
    )
