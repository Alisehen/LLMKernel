from __future__ import annotations
"""Prompt builder for Mind‑Evolution CUDA‑kernel search (seed‑kernel version).

Generates a **single prompt** that contains:
1. Target GPU spec (from `prompts/hardware/gpu_specs.py`)
2. **Few‑shot pair** – original *and* optimised model code blocks
3. Source architecture (`class Model`) that needs to be optimised
4. Existing kernel summaries (optional, for diversity context)
5. A **diversity requirement** section ensuring the new kernel differs from all previous ones
6. Output requirements

CLI usage
---------
```bash
python -m prompts.build_prompt KernelBench/level1/19_ReLU.py \
       --gpu "Quadro RTX 6000" -o prompt.txt
```
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from string import Template
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"  # GPU spec table

# --------------------------------------------------
# Few‑shot pair  (before / after)
# --------------------------------------------------
FEWSHOT_BASE = ROOT / "prompts/few_shot/model_ex_add.py"   # original Model
# Choose between CUDA and Triton implementations
FEWSHOT_NEW_CUDA = ROOT / "prompts/few_shot/model_new_ex_add.py"  # optimised with CUDA
FEWSHOT_NEW_TRITON = ROOT / "prompts/few_shot/model_new_ex_add_triton.py"  # optimised with Triton
FEWSHOT_NEW = FEWSHOT_NEW_TRITON  # Default to Triton

# Matrix multiplication examples
FEWSHOT_MATMUL_BASE = ROOT / "prompts/few_shot/model_ex_tiled_matmul.py"
FEWSHOT_MATMUL_NEW_TRITON = ROOT / "prompts/few_shot/model_new_ex_matmul_triton.py"

# Fusion operator examples
FEWSHOT_FUSION_BASE = ROOT / "prompts/few_shot/model_ex_gemm_add_relu.py"
FEWSHOT_FUSION_NEW_TRITON = ROOT / "prompts/few_shot/model_new_ex_gemm_add_relu_triton.py"

# ---------------------------------------------------------------------------
# Prompt template (with diversity requirement)
# ---------------------------------------------------------------------------
test = Template(
    dedent(
        """
Write a correct and reasonably fast Triton kernel to replace the given PyTorch operator.
This is a SEED implementation: prioritize correctness and stable compilation.

Rules:
- Use `tl.program_id(axis)` (axis=0/1/2 only)
- Use `tl.arange()` for block indices
- Operate on blocks (no CUDA thread model)
- No manual shared memory or synchronization

Hard Constraints:
- All BLOCK_* are `tl.constexpr` and powers of 2
- `tl.arange(0, BLOCK)` requires BLOCK to be power-of-2
- No dynamic `tl.reshape()` or view
- `tl.load` / `tl.store`: scalar ptr → scalar, block ptr → block
- No Python control flow on `tl.tensor` or BLOCK_*
- Triton does NOT support `continue`, `break`, or `return` inside loops — use masking instead
- Import ALL modules you use (e.g., `import math` if using `math.sqrt`)
- Do NOT index tensors with loop variables: `tensor[:, i]` or `tensor[i, :]` where i is a loop var is INVALID
- Shared memory limit ~100KB: for matmul, BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N < 25000 floats

Output Format (STRICT):
1. Imports (torch, torch.nn, triton, triton.language, and any other needed modules like math)
2. `@triton.jit` kernel(s)
3. Wrapper function(s)
4. `class ModelNew(nn.Module)` — this class is REQUIRED

Do NOT include testing code or `if __name__ == "__main__"`.

Example PyTorch:
'''
$few_base
'''

Example Triton:
'''
$few_new
'''

Hardware:
$arch_src

Target:
```python
$kernel_src
"""
    )
)
TEMPLATE = Template(
    dedent(
        """
Task
----
Generate **hand‑written CUDA kernels** that replace *all* PyTorch operator(s)
inside the original `class Model` (shown later).  You may fuse multiple
operators into a single kernel if that yields better performance.  Leave any
non‑replaced parts of the model unchanged.

OUTPUT RULES (STRICT) ────────────────────────────────────────────────
1. Inside the block, follow **exactly** this order:
   1. Imports – `torch`, `torch.nn`, `load_inline`.
   2. `source` – triple‑quoted CUDA string(s) (kernel + host wrapper).
   3. `cpp_src` – prototypes for *all* kernels you expose.
   4. **One** `load_inline` call per kernel group.
   5. `class ModelNew(nn.Module)` – mirrors original inputs/outputs but calls
      your CUDA kernels.
2. **Do NOT include** testing code, `if __name__ == "__main__"`, or extra prose.


Few‑shot example (reference only – do **not** echo):
**Original**
```python
$few_base
```
**Optimised**
```python
$few_new
```

Target architecture (to optimise):
```python
$arch_src
```

Optimize the architecture named Model with custom CUDA operators! Name your optimized
output architecture ModelNew. Output the new code in codeblocks. Please generate real
code, NOT pseudocode, make sure the code compiles and is fully functional. Just output
the new model code, no other text, and NO testing code!

Example:
```python
# <complete ModelNew code>
```
# ==========================================================
"""
    )
)
default_system_prompt = """\
You are an expert in high-performance GPU kernel optimization with Triton.

Generate the fastest possible Triton kernels. Optimize aggressively for performance while ensuring correctness.

Output format:
```python
# <complete ModelNew code with optimized Triton kernels>
```
"""

# ---------------------------------------------------------------------------
# GPU spec loader
# ---------------------------------------------------------------------------


def _load_gpu_spec() -> dict:  # noqa: D401
    """Import `gpu_specs.py` and return the GPU_SPEC_INFO dict (robust across Python versions)."""
    spec = importlib.util.spec_from_file_location("gpu_specs", HW_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {HW_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["gpu_specs"] = module  # avoid re‑import
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "GPU_SPEC_INFO"):
        raise AttributeError("GPU_SPEC_INFO not defined in gpu_specs.py")
    return module.GPU_SPEC_INFO  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Prompt builder core
# ---------------------------------------------------------------------------

def build_seed_prompt(
    arch_path: Path,
    gpu_name: str | None = None,
    fusion: bool = False,
) -> str:
    """Build LLM prompt for CUDA‑kernel optimisation (seed generation).

    Args:
        arch_path: Path to the kernel architecture file
        gpu_name: Target GPU name (auto-detect if None)
        fusion: Whether this is a fusion operator (multi-op kernel)
    """
    gpu_info = _load_gpu_spec()

    # Auto‑detect GPU if not provided
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
        f"• {k}: {v}" for k, v in info.items() if k != "GPU Architecture"
    ) if gpu_arch != "Unknown" else "Not Specified"

    # Read the kernel source
    kernel_src = Path(arch_path).read_text().strip()

    # Load few-shot examples (use fusion examples if fusion mode enabled)
    if fusion:
        few_base = FEWSHOT_FUSION_BASE.read_text().strip()
        few_new = FEWSHOT_FUSION_NEW_TRITON.read_text().strip()
    else:
        few_base = FEWSHOT_BASE.read_text().strip()
        few_new = FEWSHOT_NEW.read_text().strip()

    return test.substitute(
        few_base=few_base,
        few_new=few_new,
        arch_src=arch_src,
        kernel_src=kernel_src,
    )


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _cli() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(
        description="Build LLM prompt for CUDA‑kernel optimisation (seed generation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_py", help="Path to .py containing class Model")
    parser.add_argument("--gpu", default=None, help="GPU name key in gpu_specs.py")
    parser.add_argument("-o", "--out", help="Save prompt to file")
    args = parser.parse_args()

    prompt = build_seed_prompt(Path(args.model_py), args.gpu)

    if args.out:
        Path(args.out).write_text(prompt)
        print(f"[✓] Prompt saved to {args.out}")
    else:
        print(prompt)


if __name__ == "__main__":  # pragma: no cover
    _cli()
