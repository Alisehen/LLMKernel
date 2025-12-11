# prompts/error.py
"""
Prompt template for automatic kernel repair.
Uses `string.Template` to avoid `{}` brace conflicts with C/CUDA code.
Adds GPU hardware context and architecture source for better fixes.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Any
from string import Template

# Project roots (adjust if your tree differs)
ROOT = Path(__file__).resolve().parents[1]  # project root
HW_FILE = ROOT / "prompts/hardware/gpu_specs.py"

# Reuse your existing GPU spec loader
from prompts.generate_custom_cuda import _load_gpu_spec  # noqa: E402

COMPILE_ERROR = Template(
    """Fix the Triton kernel errors. Generate correct, high-performance code.

Current Error Log:
$ERROR_LOG

History Error:
$ERROR_HISTORY

PyTorch Reference:
```python
$PYTORCH_CODE
```

Broken Code:
```python
$OLD_CODE
```

OUTPUT RULES (STRICT):
1. Follow this exact order:
   1. Imports: torch, torch.nn, triton, triton.language as tl
   2. @triton.jit decorated kernel function(s)
   3. Wrapper function(s) for grid calculation and kernel launch
   4. class ModelNew(nn.Module) that calls your kernels
2. Do NOT include: testing code, if __name__, get_inputs, get_init_inputs
3. Learn from previous repair attempts to avoid repeating the same mistakes

```python
# <corrected code>
```
"""
)

def _escape_template(s: str) -> str:
    return s.replace("$", "$$")

def _sanitize_text(s: str) -> str:
    return s.replace("```", "`")

def _detect_triton_error_patterns(error_log: str, old_code: str) -> str:
    """
    Detect common Triton-specific error patterns and provide targeted guidance.

    Returns a guidance string if a pattern is detected, empty string otherwise.
    """
    guidance_parts = []

    # Pattern 0: tl.reshape with non-constexpr shape (CRITICAL!)
    if "Shape element" in error_log and "must have type `constexpr[int]`" in error_log:
        guidance_parts.append(
            "❌ CRITICAL: tl.reshape() requires compile-time constant shape!\n"
            "\n"
            "WRONG:\n"
            "```python\n"
            "BLOCK_HW = BLOCK_H * BLOCK_W  # This is a runtime value!\n"
            "flat = tl.reshape(tensor_2d, (BLOCK_HW,))  # ERROR: BLOCK_HW not constexpr\n"
            "```\n"
            "\n"
            "CORRECT - Use .view() with runtime shapes:\n"
            "```python\n"
            "# For flattening [BLOCK_H, BLOCK_W] -> [BLOCK_H*BLOCK_W]\n"
            "flat = tensor_2d.reshape(-1)  # Auto-infer size\n"
            "# OR directly compute flat indices without reshape:\n"
            "h_idx = tl.arange(0, BLOCK_H)[:, None]\n"
            "w_idx = tl.arange(0, BLOCK_W)[None, :]\n"
            "flat_idx = h_idx * W + w_idx  # [BLOCK_H, BLOCK_W], no flatten needed\n"
            "```\n"
        )

    # Pattern 1: thread_idx errors
    if ("thread_idx" in error_log.lower() or "thread_idx" in old_code.lower() or
        "threadidx" in error_log.lower() or "threadidx" in old_code.lower()):
        guidance_parts.append(
            "❌ CRITICAL ERROR: Triton does NOT have thread_idx (unlike CUDA)!\n"
            "\n"
            "WRONG (CUDA thinking):\n"
            "```python\n"
            "thread_idx = tl.thread_idx_x   # Does NOT exist!\n"
            "thread_idx = tl.thread_idx()   # Does NOT exist!\n"
            "h = h_start + thread_idx // W  # Wrong approach\n"
            "```\n"
            "\n"
            "CORRECT (Triton way):\n"
            "```python\n"
            "# Use tl.arange() to generate indices for the entire block\n"
            "offsets = tl.arange(0, BLOCK_SIZE)  # Creates [0, 1, 2, ..., BLOCK_SIZE-1]\n"
            "h_idx = h_start + offsets // W\n"
            "w_idx = w_start + offsets % W\n"
            "\n"
            "# For 2D indexing:\n"
            "h_offsets = tl.arange(0, BLOCK_H)[:, None]  # Shape: [BLOCK_H, 1]\n"
            "w_offsets = tl.arange(0, BLOCK_W)[None, :]  # Shape: [1, BLOCK_W]\n"
            "```\n"
            "\n"
            "KEY INSIGHT: Triton operates on BLOCKS of data, not individual threads!"
        )

    # Pattern 1: atomic_add on block_ptr
    if "atomic_add" in error_log and "block_type" in error_log:
        guidance_parts.append(
            "⚠️ TRITON ERROR DETECTED: atomic_add on block_ptr\n"
            "PROBLEM: tl.atomic_add() only accepts SCALAR pointers, not block_ptr.\n"
            "CORRECT FIX:\n"
            "```python\n"
            "# WRONG:\n"
            "# tl.atomic_add(c_block_ptr, acc)\n"
            "\n"
            "# CORRECT - Method 1 (manual iteration):\n"
            "for i in range(BLOCK_M):\n"
            "    for j in range(BLOCK_N):\n"
            "        row = base_m + i\n"
            "        col = base_n + j\n"
            "        if row < M and col < N:\n"
            "            c_scalar_ptr = c_ptr + row * stride_cm + col * stride_cn\n"
            "            tl.atomic_add(c_scalar_ptr, acc[i, j])\n"
            "\n"
            "# CORRECT - Method 2 (if no atomics needed):\n"
            "# Use regular tl.store() instead and reorganize algorithm\n"
            "```"
        )

    # Pattern 2: unsupported tensor indexing
    if "unsupported tensor index" in error_log or "int32[]" in error_log:
        guidance_parts.append(
            "⚠️ TRITON ERROR DETECTED: Invalid tensor indexing\n"
            "PROBLEM: Cannot use Python loop variables to index Triton tensors.\n"
            "REASON: Triton tensors are compile-time shapes, indices must be constexpr.\n"
            "CORRECT FIX:\n"
            "```python\n"
            "# WRONG:\n"
            "# for i in range(BLOCK_M):\n"
            "#     val = acc[i, j]  # i and j are runtime variables!\n"
            "\n"
            "# CORRECT - Option 1: Use tl.load/tl.store with offsets\n"
            "offs_m = tl.arange(0, BLOCK_M)\n"
            "offs_n = tl.arange(0, BLOCK_N)\n"
            "# Work with offset arrays, not individual indices\n"
            "\n"
            "# CORRECT - Option 2: If you must iterate, use tl.static_range\n"
            "# But note: this unrolls the loop at compile time\n"
            "```"
        )

    # Pattern 3: simultaneous multiple comparison
    if "simultaneous multiple comparison" in error_log:
        guidance_parts.append(
            "⚠️ TRITON ERROR DETECTED: Incorrect mask syntax\n"
            "PROBLEM: Missing parentheses around comparisons in mask expressions.\n"
            "CORRECT FIX:\n"
            "```python\n"
            "# WRONG:\n"
            "# mask = (rm + row_offsets)[:, None] < M & (k + col_offsets)[None, :] < K\n"
            "#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^   (lacks parentheses)\n"
            "\n"
            "# CORRECT:\n"
            "mask = ((rm + row_offsets) < M) & ((k + col_offsets) < K)\n"
            "#      ^                     ^   ^                      ^\n"
            "#      Parentheses around EACH comparison expression\n"
            "\n"
            "# For 2D masks:\n"
            "a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)\n"
            "b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)\n"
            "```"
        )

    # Pattern 4: BLOCK_M/BLOCK_N confusion
    if "threads per block" in old_code.lower() or "4096 threads" in error_log:
        guidance_parts.append(
            "⚠️ COMMON MISCONCEPTION: BLOCK_M/BLOCK_N are NOT thread counts!\n"
            "CLARIFICATION:\n"
            "- BLOCK_M, BLOCK_N, BLOCK_K are DATA dimensions (elements processed)\n"
            "- They do NOT directly specify the number of threads\n"
            "- Triton automatically determines thread layout based on operations\n"
            "- The num_warps parameter controls parallelism, not BLOCK sizes\n"
            "\n"
            "EXAMPLE:\n"
            "BLOCK_M=64, BLOCK_N=64 means:\n"
            "  → Process 64×64 = 4096 ELEMENTS per thread block\n"
            "  → Does NOT mean 4096 threads (which would exceed GPU limits)\n"
            "  → Triton maps this to efficient thread/warp layout automatically\n"
        )

    # Pattern 5: make_block_ptr issues
    if "make_block_ptr" in error_log and ("shape" in error_log or "stride" in error_log):
        guidance_parts.append(
            "⚠️ TRITON ERROR DETECTED: Incorrect make_block_ptr usage\n"
            "COMMON ISSUES:\n"
            "1. Offsets out of bounds\n"
            "2. Incorrect stride calculation\n"
            "3. Wrong order parameter\n"
            "\n"
            "CORRECT USAGE:\n"
            "```python\n"
            "# For row-major matrix A (M×K)\n"
            "a_block_ptr = tl.make_block_ptr(\n"
            "    base=a_ptr,\n"
            "    shape=(M, K),           # Full tensor shape\n"
            "    strides=(K, 1),         # Row-major: (stride_row, stride_col)\n"
            "    offsets=(row_start, col_start),  # Must be in bounds\n"
            "    block_shape=(BLOCK_M, BLOCK_K),  # Block size to load\n"
            "    order=(1, 0)            # (1,0) for row-major, (0,1) for col-major\n"
            ")\n"
            "\n"
            "# Key: shape and strides refer to FULL tensor, not the block!\n"
            "```"
        )

    # Pattern 6: Conv2D/Pooling indexing errors (correctness issues)
    # Detect by: (1) "conv" or "pool" in code, (2) outputs not close or shape errors
    is_conv_or_pool = ("conv" in old_code.lower() or "pool" in old_code.lower())
    is_correctness_error = (
        "outputs are not close" in error_log.lower() or
        "make_shape_compatible" in error_log or
        "broadcast" in error_log.lower()
    )
    if is_conv_or_pool and is_correctness_error:
        guidance_parts.append(
            "⚠️ CONV2D/POOLING INDEXING ERROR DETECTED\n"
            "\n"
            "PROBLEM: 2D spatial indexing is TRICKY in Triton - common mistakes:\n"
            "1. Mixing 2D tensor shapes with 1D flattened memory offsets\n"
            "2. Incorrectly calculating output spatial positions\n"
            "3. Wrong reshape/flatten operations\n"
            "\n"
            "CORRECT PATTERN for Conv2D:\n"
            "```python\n"
            "# Step 1: Calculate output spatial block (OH, OW dimensions)\n"
            "pid = tl.program_id(0)\n"
            "num_ow_blocks = (OW + BLOCK_W - 1) // BLOCK_W\n"
            "oh_block = pid // num_ow_blocks\n"
            "ow_block = pid % num_ow_blocks\n"
            "\n"
            "oh_start = oh_block * BLOCK_H\n"
            "ow_start = ow_block * BLOCK_W\n"
            "\n"
            "# Step 2: Create output position offsets (use 1D flattening!)\n"
            "oh_offs = oh_start + tl.arange(0, BLOCK_H)  # Shape: [BLOCK_H]\n"
            "ow_offs = ow_start + tl.arange(0, BLOCK_W)  # Shape: [BLOCK_W]\n"
            "\n"
            "# For loading: flatten 2D -> 1D\n"
            "out_idx_1d = oh_offs[:, None] * OW + ow_offs[None, :]  # Shape: [BLOCK_H, BLOCK_W]\n"
            "out_idx_flat = tl.reshape(out_idx_1d, (BLOCK_H * BLOCK_W,))\n"
            "\n"
            "# Step 3: Map output positions to input positions (consider stride, padding, kernel)\n"
            "for kh in range(KH):\n"
            "    for kw in range(KW):\n"
            "        # Input position = output_pos * stride - padding + kernel_offset\n"
            "        ih = oh_offs[:, None] * stride - padding + kh  # [BLOCK_H, 1]\n"
            "        iw = ow_offs[None, :] * stride - padding + kw  # [1, BLOCK_W]\n"
            "        \n"
            "        # Mask for boundary check\n"
            "        mask = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)  # [BLOCK_H, BLOCK_W]\n"
            "        \n"
            "        # Flatten to 1D for tl.load\n"
            "        input_offset = ih * W + iw  # [BLOCK_H, BLOCK_W]\n"
            "        input_offset_flat = tl.reshape(input_offset, (BLOCK_H * BLOCK_W,))\n"
            "        mask_flat = tl.reshape(mask, (BLOCK_H * BLOCK_W,))\n"
            "        \n"
            "        # Load with 1D offset\n"
            "        input_val = tl.load(input_ptr + input_offset_flat, mask=mask_flat, other=0.0)\n"
            "        # ...\n"
            "\n"
            "# KEY INSIGHTS:\n"
            "# - Use 2D indexing for spatial calculations (oh, ow, ih, iw)\n"
            "# - Flatten to 1D ONLY when passing to tl.load/tl.store\n"
            "# - Output position → Input position mapping: in = out * stride - padding + k_offset\n"
            "```"
        )

    if guidance_parts:
        return "\n\n" + "\n\n".join(guidance_parts) + "\n"
    return ""

def _format_problem(problem: Optional[Any]) -> str:
    import json
    if problem is None or problem == "":
        return "No prior critical problem provided."
    if isinstance(problem, Mapping):
        # Prefer to concatenate the three key fields into a concise description; otherwise fall back to JSON
        ci  = str(problem.get("critical_issue", "")).strip()
        wim = str(problem.get("why_it_matters", "")).strip()
        mfh = str(problem.get("minimal_fix_hint", "")).strip()
        if ci or wim or mfh:
            return f"critical_issue: {ci}\nwhy_it_matters: {wim}\nminimal_fix_hint: {mfh}"
        return json.dumps(problem, ensure_ascii=False, indent=2)
    # For other types, simply convert to string
    return str(problem)

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
    Build the error-repair prompt with GPU context + architecture source.

    Parameters
    ----------
    old_code : str
        The broken Python script content to show under OLD CODE.
    error_log : str
        The compiler/runtime error text to show under ERROR LOG.
    problem : Optional[Any]
        The problem analysis from previous step.
    gpu_name : Optional[str]
        Human-readable GPU name key to lookup in gpu_specs.
        If None, attempts torch.cuda.get_device_name(0).
    error_history : str
        History of previous repair attempts and their outcomes.
    arch_path : Optional[Path]
        Path to the reference PyTorch implementation file to display.

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
    triton_guidance = _detect_triton_error_patterns(error_log, old_code)
    if triton_guidance:
        triton_guidance = "\n" + triton_guidance + "\n"
    else:
        triton_guidance = ""

    # Load PyTorch reference code if provided
    pytorch_code = ""
    if arch_path and arch_path.exists():
        pytorch_code = arch_path.read_text(encoding="utf-8").strip()
    else:
        pytorch_code = "# PyTorch reference code not provided"

    # Format error history if provided
    history_section = ""
    if error_history and error_history.strip():
        history_section = f"""Previous Repair Attempts (avoid repeating these errors):
{error_history.strip()}

"""
    else:
        history_section = "None\n"

    # Substitute all fields
    return COMPILE_ERROR.substitute(
        ERROR_HISTORY=history_section,
        PYTORCH_CODE=pytorch_code,
        ERROR_LOG=error_log.strip(),
        OLD_CODE=old_code.strip(),
        TRITON_GUIDANCE=triton_guidance,
    )
