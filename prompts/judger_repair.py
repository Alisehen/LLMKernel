# prompts/judger_repair.py
"""
Prompt template for kernel correctness analysis (repair phase 1).
Analyzes errors and provides structured problem diagnosis.
"""
from __future__ import annotations
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402

# Unified prompt template
unified_prompt_tmpl = Template("""You are a Triton kernel debugging expert. Analyze the error and identify the root cause.

## ERROR LOG
```
$ERROR_LOG
```

## Expected Behavior (PyTorch Reference)
```python
$PYTORCH_CODE
```

## Current Implementation (Broken Triton Kernel)
```python
$CUDA_CODE
```

---

## Your Task

Identify the **single most critical issue** that causes the error above.

### Analysis Guidelines

1. **Focus on root cause**, not symptoms
   - Bad: "Output is wrong"
   - Good: "BLOCK_K loop missing, only processes first 32 elements of K dimension"

2. **Be specific about WHAT and WHERE**
   - Bad: "Memory access issue"
   - Good: "Line 45: tl.atomic_add(c_block_ptr, acc) - atomic_add requires scalar pointer, not block_ptr"

3. **Prioritize by impact**
   - Correctness bugs > Performance issues > Style problems
   - Algorithm errors > Implementation details

### Output Format

**CRITICAL: You MUST output ONLY valid JSON. No other text allowed.**

```json
{
  "critical_issue": "<Concise description of THE root cause, max 30 words>",
  "why_it_matters": "<Why this causes the observed error, max 35 words>",
  "minimal_fix_hint": "<What needs to change (not how), max 30 words>"
}
```

**Remember**: Output ONLY the JSON block. No explanations, no commentary, no additional text.
""")

def build_correctness_prompts(
    *,
    error_log: str,
    arch_path: Path,
    cuda_code: str,
) -> str:
    """
    Build unified prompt for kernel correctness analysis.

    Parameters
    ----------
    error_log : str
        Error message from compilation/runtime
    arch_path : Path
        Path to PyTorch reference implementation
    cuda_code : str
        Broken Triton kernel code to analyze

    Returns
    -------
    str
        Complete prompt ready for LLM
    """
    pytorch_code = Path(arch_path).read_text(encoding="utf-8").strip()

    unified_prompt = unified_prompt_tmpl.substitute(
        ERROR_LOG=error_log.strip(),
        PYTORCH_CODE=pytorch_code,
        CUDA_CODE=cuda_code.strip(),
    )

    return unified_prompt
