#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operator Categories V3 - Fine-grained classification with Triton-specific optimizations

Key improvements over V2:
1. Separated Element-wise activations (hard to optimize)
2. Distinguished Conv from Norm (different optimization strategies)
3. Triton-feasible optimizations only (no unsupported features)
4. NCU metrics with context (baseline comparison)
"""

from typing import Dict, List, Any, Optional, Tuple

# ============================================================================
# Fine-grained Operator Categories
# ============================================================================

OPERATOR_CATEGORIES = {
    # ======================================================================
    # 1. MatMul / GEMM
    # ======================================================================
    "MatMul": {
        "description": "Matrix multiply, linear layers, projections",
        "keywords": [
            "matmul", "gemm", "bmm", "einsum",
            "linear", "fc", "projection"
        ],
        "count": 0,
        "stages": [
            # --------------------------------------------------------------
            # Stage 1 — Baseline Block Tiling
            # --------------------------------------------------------------
            {
                "name": "stage1_baseline_tiling",
                "description": "Establish BLOCK_M/N/K baseline and block mapping",
                "focus": "Tile shape selection & coalesced loads",
                "key_metrics": {
                    "compute": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
                    "mem_bw": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                },
                "optimization_guidance": """
**Goals**
• Choose BLOCK_M/BLOCK_N/BLOCK_K (power-of-2, ≤256).
• Ensure load patterns are contiguous & coalesced.
• Establish correct grid mapping (pid → M/N tiles).

**Allowed**
• Change BLOCK_M/N/K.
• Change num_warps (default 4).
• Adjust program_id mapping.
• Improve tl.load/tl.store indexing.
• Use @triton.autotune if searching >4 configs (BLOCK_M/N/K, num_warps).

**Use When**
• compute < 40% or occupancy < 60%.

**Actions**
• Try smaller blocks if occupancy low.
• Increase blocks if compute is underutilized.
• If "OutOfResources: shared memory" error occurs, reduce BLOCK sizes.
""",
            },

            # --------------------------------------------------------------
            # Stage 2 — Memory Pipelining
            # --------------------------------------------------------------
            {
                "name": "stage2_memory_pipelining",
                "description": "Use num_stages & reorder loads for latency hiding",
                "focus": "Memory-compute overlap",
                "key_metrics": {
                    "l2": "lts__t_sector_hit_rate.pct",
                    "stall": "smsp__warp_issue_stalled_long_scoreboard_per_issue_active.pct",
                    "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                },
                "optimization_guidance": """
**Goals**
• Reduce memory stall cycles.
• Improve L2 locality by tuning BLOCK_K + load ordering.

**Allowed**
• Change num_stages = 2/3/4.
• Adjust BLOCK_K.
• Reorder load order inside reduction loop.
• Use @triton.autotune for num_stages + BLOCK_K combinations if uncertain.

**Use When**
• stall > 30% or L2 hit rate < 80%.

**Actions**
• Increase num_stages for memory-bound kernels.
• Reduce BLOCK_K for better L2 hit rate.
""",
            },

            # --------------------------------------------------------------
            # Stage 3 — Warp Specialization & Swizzling
            # --------------------------------------------------------------
            {
                "name": "stage3_warp_specialization",
                "description": "Tune num_warps and apply block swizzling for L2 reuse",
                "focus": "Occupancy, register usage, block scheduling",
                "key_metrics": {
                    "warp_eff": "smsp__sass_average_branch_targets_threads_uniform.pct",
                    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
                },
                "optimization_guidance": """
**Goals**
• Increase warp efficiency.
• Improve L2 locality via block scheduling (GROUP_M/N).

**Allowed**
• Change num_warps (1/2/4/8).
• Introduce block swizzling or GROUP_M.
• Minor tuning of BLOCK_M/N to reduce registers.

**Use When**
• L2 hit < 70% (after Stage 2).
• occupancy < 50% due to register pressure.

**Actions**
• Reduce num_warps if registers spill.
• Increase num_warps if compute throughput is low.
""",
            },
        ],
        "early_exit_enabled": False,
    },

    # ======================================================================
    # 2. Convolution
    # ======================================================================
    "Conv": {
        "description": "2D/ND convolution and variants",
        "keywords": ["conv", "convolution"],
        "count": 0,
        "stages": [
            # --------------------------------------------------------------
            # Stage 1 — Baseline (Direct or Im2Col)
            # --------------------------------------------------------------
            {
                "name": "stage1_baseline",
                "description": "Pick direct conv vs im2col; establish block mapping",
                "focus": "Baseline correctness + coalescing",
                "key_metrics": {
                    "mem_bw": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "coalesce": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                    "compute": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                },
                "optimization_guidance": """
**Goals**
• Select direct conv (small kernels) or im2col (large kernels).
• Achieve coalesced input/weight loads.
• Define block assignment over (N, C_out, H_out).

**Allowed**
• Change mapping of program_id axes.
• Change block sizes for H/W/C_out.
• Improve load patterns.
• Use @triton.autotune for BLOCK_H/W/C_out if input size varies.

**Early Exit Condition**
• kernel_size ≤ 3 AND score < 0.5 → stop (cuDNN stronger).

**Actions**
• If coalescing < 70%, reorganize indexing.
• If DRAM > 80%, prepare Stage 2 (tiling).
""",
            },

            # --------------------------------------------------------------
            # Stage 2 — Tile Reuse via On-Chip Cache
            # --------------------------------------------------------------
            {
                "name": "stage2_tile_reuse",
                "description": "Introduce spatial tiling to reuse input patches",
                "focus": "Improve L2/shared reuse",
                "key_metrics": {
                    "l2": "lts__t_sector_hit_rate.pct",
                    "global": "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
                },
                "skip_conditions": [
                    "kernel_size <= 3",
                ],
                "optimization_guidance": """
**Goals**
• Load a spatial tile and reuse it across kernel windows.
• Reduce global memory traffic.

**Allowed**
• Introduce BLOCK_H/BLOCK_W tiling.
• Reorder loops to reuse loaded tiles.
• Reduce tile size to fit cache.

**Use When**
• l2 < 80%, or global memory load too high.

**Actions**
• Reduce tile size if register pressure increases.
• Ensure contiguous loads for tile regions.
""",
            },

            # --------------------------------------------------------------
            # Stage 3 — Channel Grouping
            # --------------------------------------------------------------
            {
                "name": "stage3_channel_grouping",
                "description": "Vectorize over channels to improve bandwidth use",
                "focus": "Warp efficiency + vector loads",
                "key_metrics": {
                    "warp_eff": "smsp__sass_average_branch_targets_threads_uniform.pct",
                    "coalesce": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                },
                "optimization_guidance": """
**Goals**
• Increase per-block parallelism by grouping channels.
• Improve load/store coalescing for C dimension.

**Allowed**
• Introduce CHANNELS_PER_BLOCK.
• Vectorized load/store along channels.

**Use When**
• C_out > 64 or warp_eff < 80%.

**Actions**
• Try grouping of 2/4/8 channels.
""",
            },
        ],
        "early_exit_enabled": True,
        "early_exit_conditions": {
            "small_kernel_poor_perf": {
                "check": "score < 0.5",
                "message": "Small conv kernel (≤3×3) underperforming; stop optimization.",
            },
        },
    },

    # ======================================================================
    # 3. Normalization
    # ======================================================================
    "Normalization": {
        "description": "BatchNorm, LayerNorm, GroupNorm, RMSNorm",
        "keywords": ["norm", "batchnorm", "layernorm", "instancenorm", "groupnorm", "rmsnorm"],
        "count": 0,
        "stages": [
            # --------------------------------------------------------------
            # Stage 1 — Baseline 2-Pass
            # --------------------------------------------------------------
            {
                "name": "stage1_baseline",
                "description": "Standard 2-pass mean/var then normalize",
                "focus": "Coalesced reduction + correct strides",
                "key_metrics": {
                    "coalesce": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                    "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                },
                "optimization_guidance": """
**Goals**
• Efficient per-row or per-channel reduction.
• Maximize coalescing in reduction dimension.

**Allowed**
• Tune BLOCK_SIZE for reduction.
• Change mapping of program_id to normalized dims.
• Improve stride-based pointer arithmetic.
• Use @triton.autotune for BLOCK_SIZE.

**Use When**
• baseline throughput < expected memory-bound limit.

**Actions**
• Tune BLOCK_SIZE to be multiple of warp 32.
""",
            },

            # --------------------------------------------------------------
            # Stage 2 — Welford (More stable statistics)
            # --------------------------------------------------------------
            {
                "name": "stage2_welford",
                "description": "Use Welford one-pass stats (still requires 2nd pass for normalize)",
                "focus": "Numerically stable + slightly improved cache behavior",
                "key_metrics": {
                    "dram": "dram__bytes_read.sum",
                },
                "optimization_guidance": """
**Goals**
• More stable mean/var computation.
• Possibly better temporal locality (but still 2 passes overall).

**Allowed**
• Replace stats loop with Welford update.
• Preserve BLOCK_SIZE & indexing.

**Use When**
• Large reduction dimension (N > 1024).

**Actions**
• Apply only if Stage1 score < 1.3.
""",
            },

            # --------------------------------------------------------------
            # Stage 3 — Warp Reduction
            # --------------------------------------------------------------
            {
                "name": "stage3_warp_reduction",
                "description": "Make reduction dimension align with warp size",
                "focus": "Use warp-friendly BLOCK_SIZE",
                "key_metrics": {
                    "warp_eff": "smsp__sass_average_branch_targets_threads_uniform.pct",
                },
                "optimization_guidance": """
**Goals**
• Better warp utilization for the reduction axis.

**Allowed**
• Adjust BLOCK_SIZE to multiples of 32 or 64.
• Split reduction loops to expose warp-parallel chunks.

**Use When**
• warp_eff < 85%.

**Actions**
• Try BLOCK_SIZE = 64/128/256 depending on hidden dim.
""",
            },
        ],
        "early_exit_enabled": True,
        "early_exit_conditions": {
            "already_optimal": {
                "check": "score > 1.3",
                "message": "Normalization already well-optimized; diminishing returns.",
            },
        },
    },

    # ======================================================================
    # 4. Element-wise Activations
    # ======================================================================
    "Activation": {
        "description": "ReLU, Sigmoid, Tanh, GELU, SiLU, etc.",
        "keywords": [
            "relu", "leakyrelu", "sigmoid", "tanh", "gelu", "silu", "swish",
            "elu", "selu", "softplus", "softsign", "hardsigmoid", "hardtanh"
        ],
        "count": 0,
        "stages": [
            {
                "name": "stage1_simple",
                "description": "Simple parallel element-wise kernel",
                "focus": "Maximize coalescing; avoid over-tiling",
                "key_metrics": {
                    "coalesce": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                    "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                },
                "optimization_guidance": """
**Goals**
• Keep kernel minimal; avoid unnecessary tiling.

**Allowed**
• Tune BLOCK_SIZE.
• Improve indexing for contiguous load/store.
• Use @triton.autotune if activation function is complex (e.g., GELU).

**Early Exit**
• If score < 1.0 → Triton slower → stop.
• If score < 1.1 after Stage1 → stop.

**Actions**
• Try BLOCK_SIZE = 128/256 only.
""",
            },
        ],
        "early_exit_enabled": True,
        "early_exit_conditions": {
            "slower_than_pytorch": {
                "check": "score < 1.0",
                "message": "Activation slower than PyTorch; stop.",
            }
        },
    },

    # ======================================================================
    # 5. Reduction Ops
    # ======================================================================
    "Reduction": {
        "description": "sum, mean, max, softmax, logsoftmax",
        "keywords": ["sum", "mean", "max", "min", "argmax", "argmin", "reduce", "softmax", "logsoftmax"],
        "count": 0,
        "stages": [
            # --------------------------------------------------------------
            # Stage 1 — Baseline Block Reduction
            # --------------------------------------------------------------
            {
                "name": "stage1_baseline",
                "description": "Block-wise reduction over one dimension",
                "focus": "Efficient accumulation + coalesced loads",
                "key_metrics": {
                    "coalesce": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                    "warp_eff": "smsp__sass_average_branch_targets_threads_uniform.pct",
                },
                "optimization_guidance": """
**Goals**
• Implement block-level reduction with coalesced access.

**Allowed**
• Tune BLOCK_SIZE (must be power of 2, multiple of 32).
• Change mapping of program_id to reduction dim.
• Use @triton.autotune for BLOCK_SIZE.

**Use When**
• baseline reduction too slow.

**Actions**
• Make BLOCK_SIZE multiple of 32.
""",
            },

            # --------------------------------------------------------------
            # Stage 2 — Online Algorithm
            # --------------------------------------------------------------
            {
                "name": "stage2_online",
                "description": "Online reduction for large dims (softmax-specific)",
                "focus": "Reduce number of passes where applicable",
                "key_metrics": {
                    "dram": "dram__bytes_read.sum",
                },
                "optimization_guidance": """
**Goals**
• Reduce passes for softmax-like ops.
• Maintain numerical stability.

**Allowed**
• Introduce running max + adjusted exponential accumulation.
• Fuse max+exp loops where safe.

**Use When**
• Reduction dim N > 1000.

**Actions**
• Apply only if Stage1 score < 1.5.
""",
            },
        ],
        "early_exit_enabled": True,
        "early_exit_conditions": {
            "already_good": {
                "check": "score > 1.5",
                "message": "Reduction already optimized.",
            },
        },
    },

    # ======================================================================
    # 6. Fusion Ops
    # ======================================================================
    "Fusion-Ops": {
        "description": "Fused attention, fused matmul+bias+activation, etc.",
        "keywords": ["attention", "flash", "fused"],
        "count": 0,
        "stages": [
            {
                "name": "stage1_fusion_baseline",
                "description": "Remove intermediate writes, fuse loops",
                "focus": "Memory traffic reduction",
                "key_metrics": {
                    "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "compute": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                },
                "optimization_guidance": """
**Goals**
• Inline multiple ops into one kernel.
• Remove intermediate reads/writes.

**Allowed**
• Fuse matmul→bias→activation.
• Fuse Q/K/V projections.
• Fuse softmax→matmul in attention.
• Use @triton.autotune for BLOCK sizes across fused ops.

**Use When**
• DRAM traffic dominates baseline.

**Actions**
• Reduce intermediate stores; reuse on-chip results.
"""
            },
            {
                "name": "stage2_memory_hierarchy",
                "description": "Optimize tile reuse across fused stages",
                "focus": "On-chip reuse + L2 locality",
                "key_metrics": {
                    "l2": "lts__t_sector_hit_rate.pct",
                },
                "optimization_guidance": """
**Goals**
• Improve locality across fused ops.
• Reuse tiles across multiple matmul/softmax steps.

**Allowed**
• Reorder computation stages to maximize reuse.
• Adjust tile sizes to fit on-chip memory.

**Use When**
• L2 reuse low after fusion.

**Actions**
• Reduce BLOCK_K if working set too large.
""",
            },
        ],
        "early_exit_enabled": False,
    },
}


# ============================================================================
# Classification Logic
# ============================================================================

def classify_operator(task_name: str, level: str = "level1") -> str:
    """
    Classify operator into fine-grained categories.

    Args:
        task_name: Operator name (e.g., "33_BatchNorm", "20_LeakyReLU")
        level: Benchmark level ("level1" or "level2")

    Returns:
        Category name (e.g., "Normalization", "Activation")
    """
    task_lower = task_name.lower()

    # Priority order: Fusion detection first, then specific ops

    # 1. Detect Fusion ops FIRST (by checking multiple op combinations)
    # Common op keywords for fusion detection
    common_ops = ["gemm", "matmul", "linear", "conv", "add", "bias",
                  "relu", "gelu", "silu", "sigmoid", "tanh",
                  "bn", "batchnorm", "layernorm", "softmax"]

    # Count how many different op types appear in the name
    matched_ops = [op for op in common_ops if op in task_lower]

    # If 2+ different ops found → Fusion
    if len(matched_ops) >= 2:
        return "Fusion-Ops"

    # Also check explicit fusion keywords
    fusion_keywords = OPERATOR_CATEGORIES["Fusion-Ops"]["keywords"]
    if any(kw in task_lower for kw in fusion_keywords):
        return "Fusion-Ops"

    # 2. Element-wise Activations (single activation only)
    activation_keywords = OPERATOR_CATEGORIES["Activation"]["keywords"]
    if any(kw in task_lower for kw in activation_keywords):
        return "Activation"

    # 3. Normalization layers
    norm_keywords = OPERATOR_CATEGORIES["Normalization"]["keywords"]
    if any(kw in task_lower for kw in norm_keywords):
        return "Normalization"

    # 4. Convolution
    conv_keywords = OPERATOR_CATEGORIES["Conv"]["keywords"]
    if any(kw in task_lower for kw in conv_keywords):
        return "Conv"

    # 5. Reduction operations
    reduction_keywords = OPERATOR_CATEGORIES["Reduction"]["keywords"]
    if any(kw in task_lower for kw in reduction_keywords):
        return "Reduction"

    # 6. MatMul and compute-intensive
    matmul_keywords = OPERATOR_CATEGORIES["MatMul"]["keywords"]
    if any(kw in task_lower for kw in matmul_keywords):
        return "MatMul"

    # Default fallback: try to infer from context
    # If contains "fused" or multiple ops → Fusion
    if "fused" in task_lower or "_" in task_name and len(task_name.split("_")) > 3:
        return "Fusion-Ops"

    # Final fallback: MatMul (safest default)
    return "MatMul"


def get_key_ncu_metrics(category: str, stage_id: int) -> Dict[str, str]:
    """
    Get key NCU metrics for a specific category and stage.

    Args:
        category: Operator category name
        stage_id: Stage index (0-based)

    Returns:
        Dictionary mapping metric names to NCU metric identifiers
    """
    if category not in OPERATOR_CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    stages = OPERATOR_CATEGORIES[category]["stages"]
    if stage_id < 0 or stage_id >= len(stages):
        raise ValueError(f"Invalid stage_id {stage_id} for category {category}")

    return stages[stage_id]["key_metrics"]


def check_early_exit(
    category: str,
    stage_id: int,
    performance_score: float,
    op_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Check if optimization should exit early.

    Args:
        category: Operator category
        stage_id: Current stage index
        performance_score: Current speedup score
        op_metadata: Additional operator metadata (e.g., kernel_size for Conv)

    Returns:
        (should_exit, reason) tuple
    """
    if category not in OPERATOR_CATEGORIES:
        return (False, "")

    if not OPERATOR_CATEGORIES[category].get("early_exit_enabled", False):
        return (False, "")

    conditions = OPERATOR_CATEGORIES[category].get("early_exit_conditions", {})
    op_metadata = op_metadata or {}

    for cond_name, cond_config in conditions.items():
        check_expr = cond_config.get("check", "").strip()
        message = cond_config.get("message", "")

        # Build evaluation context
        eval_ctx = {
            "score": performance_score,
            "stage_id": stage_id,
            **op_metadata,
        }

        try:
            # Enhanced evaluation supporting category-specific logic

            # Category-specific early exit logic
            if category == "Conv" and cond_name == "small_kernel_poor_perf":
                # For Conv: only exit if BOTH small kernel AND poor performance

                # Skip early exit for transposed convolutions (they need more optimization attempts)
                op_type = eval_ctx.get("op_type", "").lower()
                if "transpose" in op_type or "deconv" in op_type:
                    # Transposed convolutions are harder to optimize, don't give up early
                    continue

                kernel_size = eval_ctx.get("kernel_size", 999)  # Default to large value
                threshold = 0.3  # Lowered from 0.5 to be more lenient
                if "score <" in check_expr:
                    threshold = float(check_expr.split("<")[1].strip())

                # Early exit only if kernel_size <= 3 AND score < threshold
                if kernel_size <= 3 and eval_ctx["score"] < threshold:
                    detailed_msg = f"{message} [kernel={kernel_size}x{kernel_size}, score={eval_ctx['score']:.2f}]"
                    return (True, detailed_msg)
                # If large kernel (>3), don't exit even if score is low (optimization may still help)
                continue

            # Generic score-based conditions
            if "score <" in check_expr and "AND" not in check_expr.upper():
                threshold = float(check_expr.split("<")[1].strip())
                if eval_ctx["score"] < threshold:
                    return (True, message)
            elif "score >" in check_expr and "AND" not in check_expr.upper():
                threshold = float(check_expr.split(">")[1].strip())
                if eval_ctx["score"] > threshold:
                    return (True, message)

        except Exception as e:
            # If evaluation fails, don't exit (continue optimization)
            continue

    return (False, "")


def should_skip_stage(
    category: str,
    stage_id: int,
    op_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Check if a specific stage should be skipped for this operator.

    Args:
        category: Operator category
        stage_id: Stage index to check
        op_metadata: Additional operator metadata

    Returns:
        (should_skip, reason) tuple
    """
    if category not in OPERATOR_CATEGORIES:
        return (False, "")

    stages = OPERATOR_CATEGORIES[category]["stages"]
    if stage_id < 0 or stage_id >= len(stages):
        return (False, "")

    stage = stages[stage_id]
    skip_conditions = stage.get("skip_conditions", [])

    if not skip_conditions:
        return (False, "")

    op_metadata = op_metadata or {}

    # Check skip conditions with enhanced evaluation
    for condition in skip_conditions:
        cond_lower = condition.lower().strip()

        try:
            # Handle kernel_size conditions (for Conv)
            if "kernel_size" in cond_lower and "kernel_size" in op_metadata:
                kernel_size = op_metadata["kernel_size"]

                if "<=" in cond_lower:
                    # Parse threshold (handle "kernel_size <= 3" or "kernel_size <= 3 (comment)")
                    threshold_str = cond_lower.split("<=")[1].strip()
                    # Extract just the number (ignore any trailing text)
                    threshold = int(''.join(filter(str.isdigit, threshold_str.split()[0])))

                    if kernel_size <= threshold:
                        reason = f"Stage skipped: {condition} [actual kernel_size={kernel_size}]"
                        return (True, reason)

                elif ">=" in cond_lower:
                    threshold_str = cond_lower.split(">=")[1].strip()
                    threshold = int(''.join(filter(str.isdigit, threshold_str.split()[0])))

                    if kernel_size >= threshold:
                        reason = f"Stage skipped: {condition} [actual kernel_size={kernel_size}]"
                        return (True, reason)

            # Handle score-based conditions (if any added in the future)
            elif "score" in cond_lower and "score" in op_metadata:
                score = op_metadata["score"]

                if ">" in cond_lower:
                    threshold = float(cond_lower.split(">")[1].strip())
                    if score > threshold:
                        return (True, f"Stage skipped: {condition} [actual score={score:.2f}]")
                elif "<" in cond_lower:
                    threshold = float(cond_lower.split("<")[1].strip())
                    if score < threshold:
                        return (True, f"Stage skipped: {condition} [actual score={score:.2f}]")

        except (ValueError, IndexError, KeyError) as e:
            # If parsing fails, don't skip (continue to next condition)
            continue

    return (False, "")


def build_stage_prompt_section(category: str, stage_id: int) -> str:
    """
    Build the stage-specific prompt section with guidance.

    Args:
        category: Operator category
        stage_id: Stage index

    Returns:
        Formatted prompt section string
    """
    if category not in OPERATOR_CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    stages = OPERATOR_CATEGORIES[category]["stages"]
    if stage_id < 0 or stage_id >= len(stages):
        raise ValueError(f"Invalid stage_id {stage_id} for category {category}")

    stage = stages[stage_id]

    # Build stage context
    prompt = f"""## Current Optimization Stage
**Stage**: {stage["description"]}

{stage["optimization_guidance"]}
"""

    return prompt


# ============================================================================
# Initialization
# ============================================================================

def initialize_category_counts():
    """Initialize operator counts (to be called after classification)"""
    for category in OPERATOR_CATEGORIES.values():
        category["count"] = 0


# Auto-initialize
initialize_category_counts()
