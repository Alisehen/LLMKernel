from __future__ import annotations
"""Prompt builder for CUDA kernel generation and optimization."""

import argparse
import importlib.util
import sys
from pathlib import Path
from string import Template
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Few-shot examples
FEWSHOT_SINGLE_BASE = ROOT / "prompts/few_shot/model_ex_add.py"
FEWSHOT_SINGLE_NEW_CUDA = ROOT / "prompts/few_shot/model_new_ex_add.py"

FEWSHOT_MATMUL_BASE = ROOT / "prompts/few_shot/model_ex_tiled_matmul.py"
FEWSHOT_MATMUL_NEW_CUDA = ROOT / "prompts/few_shot/model_new_ex_tiled_matmul.py"

FEWSHOT_FUSION_BASE = ROOT / "prompts/few_shot/model_ex_fuse_gelu.py"
FEWSHOT_FUSION_NEW_CUDA = ROOT / "prompts/few_shot/model_new_ex_fuse_gelu.py"

FEWSHOT_NETWORK_BASE = ROOT / "prompts/few_shot/model_ex_mnist2.py"
FEWSHOT_NETWORK_NEW_CUDA = ROOT / "prompts/few_shot/model_new_ex_mnist2.py"

FEWSHOT_FLASH_BASE = ROOT / "prompts/few_shot/model_ex_flash_attn.py"
FEWSHOT_FLASH_NEW_CUDA = ROOT / "prompts/few_shot/model_new_ex_flash_attn.py"

# Model type constants
MODEL_SINGLE = "single"
MODEL_FUSION = "fusion"
MODEL_NETWORK = "network"

FUSION_GUIDANCE = """
## Fusion Guidance
- You may keep part of the graph in PyTorch and replace only the hot path with custom CUDA.
- For fused kernels, intermediate values should stay in registers or shared memory when possible.
- Use a single final global-memory write for each output tensor.
"""

NETWORK_GUIDANCE = """
## Full-Network Guidance
- Preserve module structure, constructor arguments, and forward() semantics.
- It is acceptable to accelerate only the dominant operator(s) and keep the rest in PyTorch.
- Reuse PyTorch ops for parts that are not worth implementing as custom CUDA.
"""

PROMPT_TEMPLATE = Template(
    dedent(
        """Write high-performance custom CUDA code to replace PyTorch operators.
Generate the fastest correct implementation for the target GPU.

## Target GPU
- Name: $gpu_name
- Architecture: $gpu_arch
$arch_src

## Required Output
- Return one complete Python file.
- Use `from torch.utils.cpp_extension import load_inline`.
- Put CUDA/C++ code in `source` and declarations in `cpp_src`.
- Export callable functions via `load_inline(..., cuda_sources=source, functions=[...])`.
- Define `class ModelNew(nn.Module)` with matching behavior.

## CUDA Rules
1. Preserve the target model's constructor and forward semantics.
2. The generated code must compile with nvcc through PyTorch's inline extension flow.
3. Every CUDA kernel must use valid launch bounds and explicit out-of-bounds guards.
4. Call `C10_CUDA_KERNEL_LAUNCH_CHECK()` after kernel launches when launching manually.
5. Use contiguous tensors or make explicit contiguous copies before launching if needed.
6. Prefer coalesced global memory access, shared-memory tiling, loop unrolling, and vectorized access when justified.
7. Use `__restrict__`, `constexpr` tile sizes, and shared memory when they improve performance.
8. If a full rewrite is too risky, keep non-critical logic in PyTorch and accelerate only the bottleneck.

Do NOT include:
- testing code
- `if __name__ == "__main__"`
- `get_inputs`
- `get_init_inputs`

$fusion_guidance

Example PyTorch:
```python
$few_base
```

Example CUDA:
```python
$few_new
```

Target:
```python
$kernel_src
```"""
    )
)

default_system_prompt = """\
You are an expert in high-performance CUDA kernel optimization.

Generate the fastest possible custom CUDA implementation that can be compiled
through PyTorch's inline extension workflow while preserving correctness.

Output format:
```python
# <complete ModelNew code with CUDA/C++ extension source>
```
"""


def _load_gpu_spec() -> dict:
    """Import `gpu_specs.py` and return the GPU_SPEC_INFO dict."""
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]


def _select_few_shot(
    *,
    effective_model: str,
    kernel_src: str,
) -> tuple[str, str, str]:
    src_lower = kernel_src.lower()

    if effective_model == MODEL_NETWORK:
        return (
            FEWSHOT_NETWORK_BASE.read_text().strip(),
            FEWSHOT_NETWORK_NEW_CUDA.read_text().strip(),
            NETWORK_GUIDANCE,
        )

    if effective_model == MODEL_FUSION:
        return (
            FEWSHOT_FUSION_BASE.read_text().strip(),
            FEWSHOT_FUSION_NEW_CUDA.read_text().strip(),
            FUSION_GUIDANCE,
        )

    if any(token in src_lower for token in ("matmul", "gemm", "linear")):
        return (
            FEWSHOT_MATMUL_BASE.read_text().strip(),
            FEWSHOT_MATMUL_NEW_CUDA.read_text().strip(),
            "",
        )

    if "attention" in src_lower:
        return (
            FEWSHOT_FLASH_BASE.read_text().strip(),
            FEWSHOT_FLASH_NEW_CUDA.read_text().strip(),
            "",
        )

    return (
        FEWSHOT_SINGLE_BASE.read_text().strip(),
        FEWSHOT_SINGLE_NEW_CUDA.read_text().strip(),
        "",
    )


def build_seed_prompt(
    arch_path: Path,
    gpu_name: str | None = None,
    fusion: bool = False,
    model: str = MODEL_SINGLE,
) -> str:
    """Build LLM prompt for CUDA kernel seed generation."""
    gpu_info = _load_gpu_spec()

    if gpu_name is None:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("CUDA device not found – pass --gpu <name>.") from exc

    if gpu_name not in gpu_info:
        raise KeyError(f"{gpu_name} not present in GPU_SPEC_INFO")

    info = gpu_info[gpu_name]
    gpu_arch = info.get("GPU Architecture", "Unknown")
    arch_src = "\n".join(
        f"- {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    ) if gpu_arch != "Unknown" else "- Not Specified"

    kernel_src = Path(arch_path).read_text().strip()

    effective_model = model
    if fusion and model == MODEL_SINGLE:
        effective_model = MODEL_FUSION

    few_base, few_new, fusion_guidance = _select_few_shot(
        effective_model=effective_model,
        kernel_src=kernel_src,
    )

    return PROMPT_TEMPLATE.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        arch_src=arch_src,
        few_base=few_base,
        few_new=few_new,
        kernel_src=kernel_src,
        fusion_guidance=fusion_guidance,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build LLM prompt for CUDA kernel optimisation (seed generation)",
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
