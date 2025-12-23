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

# ConvTranspose examples (use PyTorch native + fuse post-ops)
FEWSHOT_CONVTRANSPOSE_BASE = ROOT / "prompts/few_shot/model_ex_convtranspose_pool_bias.py"
FEWSHOT_CONVTRANSPOSE_NEW_TRITON = ROOT / "prompts/few_shot/model_new_ex_convtranspose_pool_bias_triton.py"

# Model type constants
MODEL_SINGLE = "single"      # level1: single operator
MODEL_FUSION = "fusion"      # level2: fused operators
MODEL_NETWORK = "network"    # level3: full network architecture

# ---------------------------------------------------------------------------
# Fusion-specific guidance for Conv operations
# ---------------------------------------------------------------------------
FUSION_CONV_GUIDANCE = """
## Conv Fusion Pattern (CRITICAL for Conv2d/Conv3d)

Triton grid is limited to 3 dimensions. For Conv with 4D+ output, you MUST flatten:

**Conv2d: Output [N, OC, H_out, W_out] → Flatten to 2D grid**
```python
P = N * H_out * W_out  # flatten spatial + batch
grid = (cdiv(P, BLOCK_M), cdiv(OC, BLOCK_N))

# In kernel: decode flat index back to (n, oh, ow)
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
n_idx = offs_m // (H_out * W_out)
rem = offs_m % (H_out * W_out)
oh_idx = rem // W_out
ow_idx = rem % W_out

# Accumulator: simple 2D
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

# Conv loop: broadcast 1D vectors to 2D
for ic in range(C_in):
    for kh in range(K_H):
        for kw in range(K_W):
            x_vals = tl.load(...)  # [BLOCK_M]
            w_vals = tl.load(...)  # [BLOCK_N]
            acc += x_vals[:, None] * w_vals[None, :]  # [BLOCK_M, BLOCK_N]
```

**Conv3d: Output [N, OC, D_out, H_out, W_out] → Same pattern**
```python
P = N * D_out * H_out * W_out  # flatten spatial + batch
grid = (cdiv(P, BLOCK_M), cdiv(OC, BLOCK_N))

# Decode: n, od, oh, ow from flat index
DHW = D_out * H_out * W_out
HW = H_out * W_out
n_idx = offs_m // DHW
rem = offs_m % DHW
od_idx = rem // HW
rem2 = rem % HW
oh_idx = rem2 // W_out
ow_idx = rem2 % W_out
```

**Key Rules:**
- NEVER use 4D grid (program_id only supports 0,1,2)
- ALWAYS flatten output positions to 1 dimension
- Use simple 2D accumulator [BLOCK_M, BLOCK_N]
- Broadcast: x[:, None] * w[None, :] for clean 2D matmul-like pattern
"""

# ---------------------------------------------------------------------------
# Network model guidance (level3) - selective optimization
# ---------------------------------------------------------------------------
NETWORK_GUIDANCE = """
## Full-Network Optimization Guidance
- Target is a **complete nn.Module**, not a single operator.
- Preserve module structure and forward() control flow.
- Allow fusion **within a branch or module**, but NOT across branches.
"""

# ---------------------------------------------------------------------------
# Prompt template (with diversity requirement)
# ---------------------------------------------------------------------------
test = Template(
    dedent(
        """
Write high-performance Triton kernels to replace PyTorch operators.
Generate the FASTEST kernel while maintaining correctness.

## CRITICAL — These cause 60%+ of failures:
1. EVERY kernel function MUST have `@triton.jit` decorator — MANDATORY
2. Grid size MUST be > 0: use `triton.cdiv(N, BLOCK)` or `max(1, N // BLOCK)`
3. BLOCK sizes MUST be power-of-2 constexpr: 16, 32, 64, 128, 256
4. `tl.program_id(axis)` only supports axis = 0, 1, 2 (max 3D grid)

## Triton Syntax Rules:
- For matmul/conv/linear ops, prefer `tl.dot(a, b, allow_tf32=True)` over element-wise multiply-add
- No `continue`, `break`, `return` inside loops — use masking instead
- No tensor indexing with loop vars: `x[:, i]` or `x[i, :]` is INVALID
- No tuple unpacking inside kernel: `a, b = tl.load(...)` is INVALID
- No nested functions inside @triton.jit
- No Python control flow on tl.tensor or BLOCK_* values
- No dynamic `tl.reshape()` or view operations

## Missing Triton Functions (implement manually):
- tl.tanh → `(tl.exp(2*x) - 1) / (tl.exp(2*x) + 1)`
- tl.sigmoid → `1 / (1 + tl.exp(-x))`
- tl.gelu, tl.silu, tl.softmax, tl.mish → implement from definition

## Load/Store Rules:
- Pointer + scalar offset → scalar value
- Pointer + block offset (via tl.arange) → block of values
- mask shape MUST match data shape exactly

## Output Format (STRICT):
1. Imports: `import torch, torch.nn as nn, triton, triton.language as tl` (and math if needed)
2. `@triton.jit` kernel(s) — MUST have this decorator
3. Wrapper function with grid calculation
4. `class ModelNew(nn.Module)` — REQUIRED

Do NOT include: testing code, `if __name__ == "__main__"`, get_inputs, get_init_inputs

$fusion_guidance

Example PyTorch:
'''
$few_base
'''

Example Triton:
'''
$few_new
'''

Target:
```python
$kernel_src
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
    model: str = MODEL_SINGLE,
) -> str:
    """Build LLM prompt for CUDA‑kernel optimisation (seed generation).

    Args:
        arch_path: Path to the kernel architecture file
        gpu_name: Target GPU name (auto-detect if None)
        fusion: Whether this is a fusion operator (multi-op kernel)
        model: Model type - "single" (level1), "fusion" (level2), or "network" (level3)
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

    # Check if source contains ConvTranspose
    has_conv_transpose = "convtranspose" in kernel_src.lower() or "conv_transpose" in kernel_src.lower()

    # Determine effective model type (fusion param for backward compatibility)
    effective_model = model
    if fusion and model == MODEL_SINGLE:
        effective_model = MODEL_FUSION

    # Load few-shot examples based on model type
    if effective_model == MODEL_NETWORK:
        # Level 3: Full network - use fusion examples + network guidance
        few_base = FEWSHOT_FUSION_BASE.read_text().strip()
        few_new = FEWSHOT_FUSION_NEW_TRITON.read_text().strip()
        fusion_guidance = NETWORK_GUIDANCE
    elif effective_model == MODEL_FUSION:
        # Level 2: Fused operators
        if "conv" in str(arch_path).lower():
            few_base = FEWSHOT_FUSION_BASE.read_text().strip()
            few_new = FEWSHOT_FUSION_NEW_TRITON.read_text().strip()
            fusion_guidance = FUSION_CONV_GUIDANCE
        else:
            few_base = FEWSHOT_FUSION_BASE.read_text().strip()
            few_new = FEWSHOT_FUSION_NEW_TRITON.read_text().strip()
            fusion_guidance = ""
    else:
        # Level 1: Single operator
        few_base = FEWSHOT_BASE.read_text().strip()
        few_new = FEWSHOT_NEW.read_text().strip()
        fusion_guidance = ""

    return test.substitute(
        few_base=few_base,
        few_new=few_new,
        arch_src=arch_src,
        kernel_src=kernel_src,
        fusion_guidance=fusion_guidance,
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
    parser.add_argument("--model", default=MODEL_SINGLE,
                        choices=[MODEL_SINGLE, MODEL_FUSION, MODEL_NETWORK],
                        help="Model type: single (level1), fusion (level2), network (level3)")
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
