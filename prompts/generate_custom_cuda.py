from __future__ import annotations
"""Prompt builder for CUDA kernel generation."""

import argparse
import importlib.util
import sys
from pathlib import Path
from string import Template
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Few-shot examples
FEWSHOT_BASE = ROOT / "prompts/few_shot/model_ex_add.py"
FEWSHOT_NEW = ROOT / "prompts/few_shot/model_new_ex_add.py"

FEWSHOT_FUSION_BASE = ROOT / "prompts/few_shot/model_ex_fuse_gelu.py"
FEWSHOT_FUSION_NEW = ROOT / "prompts/few_shot/model_new_ex_fuse_gelu.py"

FEWSHOT_MATMUL_BASE = ROOT / "prompts/few_shot/model_ex_tiled_matmul.py"
FEWSHOT_MATMUL_NEW = ROOT / "prompts/few_shot/model_new_ex_tiled_matmul.py"

FEWSHOT_NETWORK_BASE = ROOT / "prompts/few_shot/model_ex_mnist2.py"
FEWSHOT_NETWORK_NEW = ROOT / "prompts/few_shot/model_new_ex_mnist2.py"

MODEL_SINGLE = "single"
MODEL_FUSION = "fusion"
MODEL_NETWORK = "network"

FUSION_GUIDANCE = """
## Fusion Guidance
- Prefer fusing elementwise chains, reductions followed by pointwise ops, or linear + activation style patterns.
- For large convolutions, do not force a full custom kernel unless the mapping is clear and profitable.
- It is acceptable to keep heavy library ops in PyTorch and fuse only the surrounding post-processing in CUDA.
"""

NETWORK_GUIDANCE = """
## Full-Network Guidance
- Preserve the nn.Module structure and forward() control flow.
- Optimize the main hotspots, not necessarily every operator.
- Do not fuse across branches or residual paths unless the dependency is simple and explicit.
- Keep parameter names and public behavior compatible with the reference Model.
"""

PROMPT = Template(
    dedent(
        """
You write high-performance custom CUDA extensions for PyTorch.
Generate a complete Python implementation that uses `torch.utils.cpp_extension.load_inline`
to build and launch hand-written CUDA kernels.

## Target GPU
$gpu_spec

## Required Strategy
- Replace the hottest PyTorch operator(s) with custom CUDA kernels.
- You may fuse multiple operators into one kernel when it clearly reduces memory traffic or launch overhead.
- You may leave non-critical parts in PyTorch if that is the better engineering choice.
- Prioritize correctness, compilability, and measurable speedup.

## Output Rules
1. Output a single Python code block only.
2. Inside the code, follow this order:
   - imports
   - `source` CUDA string(s)
   - `cpp_src` declaration string if needed
   - `load_inline(...)`
   - `class ModelNew(nn.Module)`
3. Use `from torch.utils.cpp_extension import load_inline`.
4. Build real CUDA code, not pseudocode.
5. Do not include tests, `get_inputs`, `get_init_inputs`, or `if __name__ == "__main__"`.

## CUDA Requirements
- The generated CUDA entrypoints must validate tensor device/dtype/contiguity when needed.
- Use descriptive exported function names such as `foo_cuda`.
- Launch parameters must be explicit and valid.
- Check kernel launch errors when appropriate.
- Keep CPU fallback paths when the custom CUDA path is not applicable.

$extra_guidance

## Few-shot Example
Reference PyTorch:
```python
$few_base
```

Reference CUDA rewrite:
```python
$few_new
```

## Target Model
```python
$kernel_src
```
"""
    )
)

default_system_prompt = """\
You are a senior CUDA kernel optimization specialist.

Return only runnable Python code that builds and uses hand-written CUDA kernels through
`torch.utils.cpp_extension.load_inline`.
"""


def _load_gpu_spec() -> dict:
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]


def _pick_fewshot(model: str, kernel_src: str) -> tuple[str, str, str]:
    if model == MODEL_NETWORK:
        return (
            FEWSHOT_NETWORK_BASE.read_text().strip(),
            FEWSHOT_NETWORK_NEW.read_text().strip(),
            NETWORK_GUIDANCE,
        )

    if model == MODEL_FUSION:
        if "matmul" in kernel_src.lower() or "linear" in kernel_src.lower():
            return (
                FEWSHOT_MATMUL_BASE.read_text().strip(),
                FEWSHOT_MATMUL_NEW.read_text().strip(),
                FUSION_GUIDANCE,
            )
        return (
            FEWSHOT_FUSION_BASE.read_text().strip(),
            FEWSHOT_FUSION_NEW.read_text().strip(),
            FUSION_GUIDANCE,
        )

    if "matmul" in kernel_src.lower():
        return (
            FEWSHOT_MATMUL_BASE.read_text().strip(),
            FEWSHOT_MATMUL_NEW.read_text().strip(),
            "",
        )

    return (
        FEWSHOT_BASE.read_text().strip(),
        FEWSHOT_NEW.read_text().strip(),
        "",
    )


def build_seed_prompt(
    arch_path: Path,
    gpu_name: str | None = None,
    fusion: bool = False,
    model: str = MODEL_SINGLE,
) -> str:
    gpu_info = _load_gpu_spec()

    if gpu_name is None:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    kernel_src = Path(arch_path).read_text().strip()
    effective_model = MODEL_FUSION if fusion and model == MODEL_SINGLE else model
    few_base, few_new, extra_guidance = _pick_fewshot(effective_model, kernel_src)

    info = gpu_info[gpu_name]
    gpu_spec = "\n".join(f"- {k}: {v}" for k, v in info.items())

    return PROMPT.substitute(
        gpu_spec=gpu_spec,
        extra_guidance=extra_guidance,
        few_base=few_base,
        few_new=few_new,
        kernel_src=kernel_src,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build LLM prompt for CUDA kernel generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_py", help="Path to .py containing class Model")
    parser.add_argument("--gpu", default=None, help="GPU name key in gpu_specs.py")
    parser.add_argument(
        "--model",
        default=MODEL_SINGLE,
        choices=[MODEL_SINGLE, MODEL_FUSION, MODEL_NETWORK],
        help="Model type: single (level1), fusion (level2), network (level3)",
    )
    parser.add_argument("-o", "--out", help="Save prompt to file")
    args = parser.parse_args()

    prompt = build_seed_prompt(Path(args.model_py), args.gpu, model=args.model)

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"[✓] Prompt saved to {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":  # pragma: no cover
    _cli()
