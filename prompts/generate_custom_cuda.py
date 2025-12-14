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
# Operator-specific guidance
# ---------------------------------------------------------------------------

OPERATOR_GUIDANCE = {
    "conv_transpose": """
Performance Guidelines for Transposed Convolution (ConvTranspose):
⚠️ CRITICAL DIFFERENCES from standard convolution:
- Output size GROWS: out_size = (in_size - 1) * stride + kernel_size - 2*padding
- Each INPUT element writes to MULTIPLE OUTPUT positions (scatter pattern)
- Groups require: in_channels % groups == 0 AND out_channels % groups == 0

RECOMMENDED APPROACH - Direct Implementation (NOT implicit GEMM):
1. **Reverse Index Mapping**:
   For each output position (od, oh, ow), find ALL contributing input positions:
   ```python
   # For each kernel position (kd, kh, kw):
   id = (od + padding - kd) // stride  # Must check if divisible!
   ih = (oh + padding - kh) // stride
   iw = (ow + padding - kw) // stride

   # Valid only if:
   valid = ((od + padding - kd) % stride == 0) & (id >= 0) & (id < ID) & ...
   ```

2. **Grid Layout for 3D Transposed Conv**:
   ```python
   # Parallelize over output spatial dimensions
   N, C_out, OD, OH, OW = output.shape
   grid = (triton.cdiv(N * OD * OH * OW, BLOCK_OUT), C_out // groups, groups)

   # In kernel - decode flat index:
   pid = tl.program_id(0)
   out_idx = pid * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
   n = out_idx // (OD * OH * OW)
   remainder = out_idx % (OD * OH * OW)
   od = remainder // (OH * OW)
   oh = (remainder % (OH * OW)) // OW
   ow = remainder % OW
   ```

3. **Grouped Convolution Handling**:
   ```python
   group_id = tl.program_id(2)
   C_in_per_group = C_in // groups
   C_out_per_group = C_out // groups

   c_in_start = group_id * C_in_per_group
   c_out_start = group_id * C_out_per_group

   # Weight indexing: weight[c_out_in_group, c_in_in_group, kd, kh, kw]
   # where c_out_in_group ∈ [0, C_out_per_group)
   #       c_in_in_group ∈ [0, C_in_per_group)
   ```

4. **Accumulation Pattern**:
   ```python
   acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

   # Loop over kernel dimensions
   for kd in range(KD):
       for kh in range(KH):
           for kw in range(KW):
               # Compute input indices
               id = (od + padding - kd) // stride
               ih = (oh + padding - kh) // stride
               iw = (ow + padding - kw) // stride

               # Check validity (must be divisible AND in bounds)
               valid_d = ((od + padding - kd) % stride == 0) & (id >= 0) & (id < ID)
               valid_h = ((oh + padding - kh) % stride == 0) & (ih >= 0) & (ih < IH)
               valid_w = ((ow + padding - kw) % stride == 0) & (iw >= 0) & (iw < IW)
               valid = valid_d & valid_h & valid_w

               # Loop over input channels in this group
               for c_in_offset in range(C_in_per_group):
                   c_in = c_in_start + c_in_offset
                   input_val = tl.load(input_ptr + ..., mask=valid, other=0.0)
                   weight_val = tl.load(weight_ptr + ...)
                   acc += input_val * weight_val
   ```

5. **AVOID Common Errors**:
   ❌ Do NOT use implicit GEMM (access pattern is fundamentally different)
   ❌ Do NOT forget modulo check: (od + pad - kd) % stride == 0
   ❌ Do NOT ignore groups in weight/channel indexing
   ❌ Do NOT use atomics unless truly necessary (prefer output parallelism)

6. **Numerical Precision**:
   - Use fp32 accumulation even for fp16 input
   - Test stride > 1 and groups > 1 carefully
   - Verify output shape formula matches PyTorch exactly

7. **Performance Tips**:
   - For small kernels (3x3x3), unroll loops manually
   - Use vectorized loads when possible (BLOCK_OUT should be power of 2)
   - Consider tiling over channels if C_in_per_group is large
""",

    "conv": """
Performance Guidelines for STANDARD Convolution (Conv2d/Conv3d):
⚠️ For TRANSPOSED convolution (ConvTranspose), see conv_transpose guidance instead!

For STANDARD convolution only:
- Use **implicit GEMM** approach — treat convolution as matrix multiplication:
  * Output shape: (N*OH*OW, OC) for 2D, (N*OD*OH*OW, OC) for 3D
  * Weight shape: (OC, IC*KH*KW) for 2D, (OC, IC*KD*KH*KW) for 3D
  * Compute input indices on-the-fly from output position
  * Use `tl.dot()` for the matrix multiply to leverage Tensor Cores

- **Groups handling**:
  * Split channels: each group processes (C_in/groups) → (C_out/groups)
  * Use program_id for group parallelism
  * Weight indexing: weight[group_oc, group_ic*K*H*W]

- **Index Calculation** (2D example with proper broadcasting):
  ```python
  # Each thread block handles BLOCK_M output positions
  pid_m = tl.program_id(0)
  offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

  # Decode to (n, oh, ow)
  n = offs_m // (OH * OW)
  oh = (offs_m % (OH * OW)) // OW
  ow = offs_m % OW

  # Loop over kernel with broadcasting
  for kh in range(KH):
      for kw in range(KW):
          # Compute input positions (vectorized)
          ih = oh * stride - padding + kh  # Shape: (BLOCK_M,)
          iw = ow * stride - padding + kw

          # Boundary check
          mask = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

          # Load input: shape (BLOCK_M, C_in)
          # Load weight: shape (C_out, C_in)
          # Accumulate: (BLOCK_M, C_out) using tl.dot or manual multiply-add
  ```

- **For depthwise/separable convolution**:
  * Depthwise: groups = in_channels = out_channels
  * Each channel has its own kernel
  * Much simpler than grouped conv

- **For pointwise/1x1 convolution** (kernel_size=1):
  * This is just a GEMM: (N*H*W, C_in) @ (C_in, C_out) → (N*H*W, C_out)
  * Use tl.dot() with proper tiling
  * Keep autotune configs minimal (2-3 max) - autotuning is SLOW!
  * PyTorch 1x1 conv is already highly optimized (uses cuBLAS)
  * Example:
    ```python
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def pointwise_kernel(...): ...
    ```

- AVOID naive nested loops over kernel dimensions without vectorization
- Use proper tiling (BLOCK_M, BLOCK_N, BLOCK_K) and iterate over K dimension in tiles
""",

    "matmul_large_k": """
Performance Guidelines for Large-K Matrix Multiplication:
For K > 100k, numerical precision is the PRIMARY challenge (more than performance).

CRITICAL CONSTRAINTS:
1. Shared memory limit ~100KB: BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N must fit
   - BLOCK_M=128, BLOCK_N=128, BLOCK_K=128 needs 128KB → TOO LARGE!
   - Safe config: BLOCK_M=64, BLOCK_N=64, BLOCK_K=64 or BLOCK_M=128, BLOCK_N=64, BLOCK_K=32
2. Use larger BLOCK_K (64+) to reduce accumulation steps, but stay within shared memory

RECOMMENDED APPROACH for extreme K (>500k):
- Consider splitting K into chunks and accumulating in PyTorch:
```python
def large_k_matmul(A, B, chunk_size=65536):
    M, K = A.shape
    K_, N = B.shape
    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
    for k_start in range(0, K, chunk_size):
        k_end = min(k_start + chunk_size, K)
        C += triton_matmul(A[:, k_start:k_end], B[k_start:k_end, :])
    return C
```
This reduces accumulation error by chunking.

BLOCK SIZE GUIDE (for shared memory ~100KB):
- BLOCK_M=64, BLOCK_N=64, BLOCK_K=64: uses ~32KB (safe)
- BLOCK_M=128, BLOCK_N=64, BLOCK_K=32: uses ~24KB (safe)
- BLOCK_M=128, BLOCK_N=128, BLOCK_K=32: uses ~32KB (safe)
""",

    "pooling": """
Performance Guidelines for Pooling Operations:
- Do NOT use `tl.static_range(hardcoded_value)` — it unrolls loops at compile time
- Do NOT hardcode kernel dimensions in loops — use dynamic range or treat as reduction
- Correct index calculation pattern:
  ```python
  # Each thread handles one output element
  pid = tl.program_id(0)
  # Decode to (n, c, oh, ow)
  n, c, oh, ow = decode_position(pid, ...)
  # Loop over kernel with proper bounds
  for kh in range(kernel_size):  # NOT tl.static_range!
      for kw in range(kernel_size):
          ih = oh * stride + kh - padding
          iw = ow * stride + kw - padding
          # Careful: all variables are SCALARS here in this pattern
  ```
- Alternative: Use 2D tiling like Conv (implicit GEMM style) for better performance
""",

    "scan": """
Performance Guidelines for Scan Operations (cumsum, cumprod):
- Use `tl.cumsum(x, axis=0)` for within-block cumulative sum — do NOT manually loop
- Do NOT index tensors with constants like `tensor[0]` — Triton does not support this
- For cross-block scan (when dim_size > BLOCK_SIZE), use multi-pass algorithm:
  1. Pass 1: Compute per-block reductions (block_sums)
  2. Pass 2: Compute prefix sum of block_sums
  3. Pass 3: Add prefix to each block's local scan result

MANDATORY for reverse/exclusive cumsum — YOU MUST FOLLOW THIS:
- Do NOT implement reverse/shift logic inside Triton kernel — it's error-prone!
- Triton kernel should ONLY do standard inclusive cumsum
- Use PyTorch ops for pre/post processing (flip, shift, pad)

REQUIRED PATTERN for exclusive_cumsum:
```python
def triton_exclusive_cumsum(x, dim):
    # Step 1: Use Triton for inclusive cumsum only
    inclusive = triton_inclusive_cumsum(x, dim)  # Your Triton kernel here

    # Step 2: Use PyTorch to shift (convert inclusive -> exclusive)
    # Exclusive means output[i] = sum of x[0:i], so shift right and pad 0
    zeros = torch.zeros_like(x.select(dim, 0).unsqueeze(dim))
    exclusive = torch.cat([zeros, inclusive.narrow(dim, 0, x.size(dim)-1)], dim=dim)
    return exclusive
```

REQUIRED PATTERN for reverse_cumsum:
```python
def triton_reverse_cumsum(x, dim):
    x_flipped = x.flip(dim)                    # PyTorch flip
    result = triton_inclusive_cumsum(x_flipped, dim)  # Triton cumsum
    return result.flip(dim)                    # PyTorch flip back
```

DO NOT try to implement shift/flip inside the Triton kernel — use the patterns above!
""",
}


def _detect_operator_type(file_path: Path) -> list[str]:
    """Detect operator type from filename and return list of relevant guidance keys.

    Args:
        file_path: Path to the kernel file

    Returns:
        List of keys for OPERATOR_GUIDANCE to include
    """
    filename = file_path.name.lower()
    guidance_keys = []

    # Conv operations - CHECK TRANSPOSED FIRST before standard conv!
    if "conv_transpose" in filename or "convtranspose" in filename or "transposed" in filename:
        guidance_keys.append("conv_transpose")
    elif "conv" in filename:
        guidance_keys.append("conv")

    # Pooling operations
    if "pool" in filename:
        guidance_keys.append("pooling")

    # Scan operations
    if any(op in filename for op in ["cumsum", "cumprod", "cummax", "cummin", "scan"]):
        guidance_keys.append("scan")

    # Large K matmul - detect from file content if possible, or special naming
    if "matmul" in filename and "large_k" in filename:
        guidance_keys.append("matmul_large_k")

    return guidance_keys


def _build_operator_guidance(guidance_keys: list[str]) -> str:
    """Build operator-specific guidance section from keys.

    Args:
        guidance_keys: List of keys for OPERATOR_GUIDANCE

    Returns:
        Formatted guidance string, or empty if no keys
    """
    if not guidance_keys:
        return ""

    sections = [OPERATOR_GUIDANCE[key] for key in guidance_keys if key in OPERATOR_GUIDANCE]

    if not sections:
        return ""

    return "\n".join(sections)


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
) -> str:
    """Build LLM prompt for CUDA‑kernel optimisation (seed generation)."""
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

    # Load few-shot examples
    few_base = FEWSHOT_BASE.read_text().strip()
    few_new = FEWSHOT_NEW.read_text().strip()

    # Detect operator type and build guidance
    guidance_keys = _detect_operator_type(arch_path)
    operator_guidance = _build_operator_guidance(guidance_keys)

    return test.substitute(
        few_base=few_base,
        few_new=few_new,
        arch_src=arch_src,
        kernel_src=kernel_src,
        OPERATOR_SPECIFIC_GUIDANCE=operator_guidance
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
