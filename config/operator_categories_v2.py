"""
算子分类配置与优化策略 (精简版 - LLM友好)

设计原则：
1. 每个stage只关注2-3个核心指标
2. 指标要能直接映射到优化动作
3. 提供清晰的条件判断和优化建议
"""

import re
from typing import Dict, Tuple, Optional

# ============================================================================
# 四大类别配置
# ============================================================================

OPERATOR_CATEGORIES = {

    # ------------------------------------------------------------------------
    # 1. Compute-Intensive: Matmul/GEMM/BMM (16个)
    # ------------------------------------------------------------------------
    "Compute-Intensive": {
        "description": "计算密集型算子 (Matmul/GEMM/BMM)",
        "count": 16,
        "primary_ops": ["Matmul", "Gemm", "BMM", "matrix_multiplication"],

        "stages": [
            {
                "name": "stage1_tiling_and_shared_memory",
                "description": "2D Tiling + Shared Memory",
                "focus": "Establish baseline with 2D tiling and shared memory caching",

                "key_metrics": {
                    "shared_memory_usage": "smsp__inst_executed_op_shared.sum",
                    "compute_throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "global_load_efficiency": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                },

                "optimization_guidance": """
**Goal**: Implement efficient 2D tiling for matmul.

**Key Implementation**:
• Use 2D grid: `(M/BLOCK_M, N/BLOCK_N)`
• Use `tl.load()` + `tl.dot(a, b)` - Triton automatically caches in shared memory
• **Do NOT** manually allocate shared memory with `tl.zeros()` unless needed for advanced patterns
• Start with: BLOCK_M=BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2

**IMPORTANT - Avoid Common Mistakes**:
• ❌ Do NOT create unused `tl.zeros((BLOCK_M, BLOCK_K))` for "shared memory"
• ❌ Do NOT increase num_warps beyond 4-8 without justification
• ✅ Keep it simple: load → dot → accumulate → store

**NCU Validation**:
• shared_memory_usage > 0 → Triton is using shared memory automatically
• global_load_efficiency > 70% → good memory coalescing
""",
            },

            {
                "name": "stage2_block_size_tuning",
                "description": "Autotune Block Sizes + Warps",
                "focus": "Find optimal config via autotune to improve SM utilization",

                "key_metrics": {
                    "compute_throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
                    "pipeline_stalls": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct",
                },

                "optimization_guidance": """
**Goal**: Find optimal block sizes via autotune.

**IMPORTANT - Check Current Performance First**:
• If current kernel already achieves score > 2.0 → use narrow tuning range
• If score < 1.0 → wider exploration may help

**Conservative Autotune** (recommended for good baselines):
• BLOCK_M/N: 64, 128 (avoid 256 unless matrix is huge)
• BLOCK_K: 32, 64
• num_warps: 4, 8 (avoid higher values)
• num_stages: 2, 3 (avoid 4+ unless memory stalls > 40%)

**NCU-Guided Tuning**:
• occupancy < 50% → reduce block size or num_warps
• pipeline_stalls > 40% → try num_stages=3
• compute_throughput < 40% → try BLOCK_M/N=128

**Avoid**:
• ❌ Excessive configs (more configs = slower compilation, not better results)
• ❌ Extreme values (BLOCK_M=256, num_warps=16)
""",
            },

            {
                "name": "stage3_advanced_optimizations",
                "description": "Vectorization + Block Swizzling",
                "focus": "Optimize memory access pattern and L2 cache utilization",

                "key_metrics": {
                    "l2_hit_rate": "lts__t_sector_hit_rate.pct",
                    "dram_throughput": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                },

                "optimization_guidance": """
**Goal**: Micro-optimizations for L2 cache (typically 5-15% gains).

**IMPORTANT - Check If Needed**:
• If current score > 2.0 → **SKIP this stage** (likely already optimal)
• If DRAM throughput < 60% AND L2 hit > 80% → **SKIP** (not memory-bound)
• Only proceed if NCU shows clear memory bottleneck

**Block Swizzling** (advanced):
• Reorders block IDs for better L2 cache sharing across SMs
• Use group_size=8 typically
• **Warning**: Complex to implement correctly, easy to break existing code

**Safer Alternative**:
• Adjust BLOCK_K to improve spatial locality (e.g., 32→64)
• Ensure block sizes are power-of-2

**Recommendation**:
• For already-good kernels (score > 2.0), prefer **no changes** over risky optimizations
""",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 2. Memory-Intensive: Conv/Activation/Norm/Reduction (84个)
    # ------------------------------------------------------------------------
    "Memory-Intensive": {
        "description": "访存密集型算子 (Conv/Activation/Norm/Reduction)",
        "count": 84,

        "stages": [
            {
                "name": "stage1_baseline_vectorization",
                "description": "Baseline + Vectorized Access",
                "focus": "Establish baseline with vectorized memory access",

                "key_metrics": {
                    "dram_throughput": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "memory_coalescing": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                    "compute_throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                },

                "optimization_guidance": """
**Goal**: Vectorized access + assess optimization feasibility.

**⚠️ CRITICAL - Element-wise ops (ReLU, LeakyReLU, Sigmoid, Tanh, GELU, Swish)**:
• If current score < 1.0 → **STOP OPTIMIZATION IMMEDIATELY**
• PyTorch's element-wise kernels are already highly optimized (cuDNN/custom CUDA)
• Triton cannot beat them for simple element-wise operations
• ✅ Use simple implementation: BLOCK_SIZE=256, contiguous access only
• ❌ DO NOT attempt vectorization tricks (e.g., stride in arange)
• ❌ DO NOT proceed to Stage 2/3 if score < 1.0

**Conv**:
• WARNING: If kernel_size <= 3 AND score < 0.3 → optimization unlikely to help (cuDNN is highly optimized)

**Norm** (BatchNorm/LayerNorm):
• Use 2-pass or Welford algorithm
• Block-level reduction for mean/variance
• **MEMORY EFFICIENCY**:
  - ❌ DO NOT create output tensor then reshape it with `.contiguous()` (creates copies)
  - ✅ Create output in final shape directly
  - ❌ Avoid: `y = empty_like(x); y_reshaped = y.permute(...).contiguous()` (2x memory!)
  - ✅ Use: `y = empty(target_shape); ... ; y = y.view(original_shape)`

**NCU Validation**:
• memory_coalescing > 60% → good vectorization
• compute_throughput < 20% → memory-bound (expected)
""",

                "early_exit_conditions": {
                    "elementwise_poor_performance": {
                        "check": "op_type in ['relu', 'leakyrelu', 'sigmoid', 'tanh', 'gelu', 'swish'] AND score < 1.0",
                        "message": "Element-wise op slower than PyTorch - further optimization unlikely to help",
                    },
                    "conv_small_kernel": {
                        "check": "op_type contains 'conv' AND kernel_size <= 3 AND score < 0.3",
                        "message": "Conv with small kernel is hard to optimize, cuDNN is better",
                    }
                },
            },

            {
                "name": "stage2_shared_memory_optimization",
                "description": "Shared Memory Caching",
                "focus": "Use shared memory to reduce global memory accesses",

                "key_metrics": {
                    "global_load_bytes": "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
                    "shared_load_transactions": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
                },

                "optimization_guidance": """
**Goal**: Use shared memory to reduce global memory traffic (only when beneficial).

**IMPORTANT - Check Applicability**:
• **Conv**: SKIP if kernel_size <= 3 (limited reuse, not worth complexity)
• **Element-wise** (ReLU, Sigmoid, etc.): **ALWAYS SKIP** (no data reuse)
• **Norm/Reduction**: May benefit from shared memory for partial results

**For Norm/Reduction**:
• Use shared memory for block-level reductions
• **Do NOT** manually allocate unless using advanced patterns
• Use Triton's atomic operations or reduction primitives

**Common Pitfall**:
• ❌ Adding `tl.zeros()` for "shared memory" without actually using it
• ❌ Using shared memory when global memory is already sufficient

**When to Skip This Stage**:
• Element-wise ops
• Conv with kernel_size <= 3
• Already good performance (score > 1.5)
""",

                "skip_conditions": [
                    "Element-wise activation ops (ReLU, Sigmoid, Tanh, GELU, etc.)",
                    "Operations with no data reuse pattern",
                ],
            },

            {
                "name": "stage3_algorithm_improvements",
                "description": "Advanced Algorithms (Welford/Warp Shuffle)",
                "focus": "Single-pass algorithms and warp-level optimizations",

                "key_metrics": {
                    "warp_uniformity": "smsp__sass_average_branch_targets_threads_uniform.pct",
                },

                "optimization_guidance": """
**Goal**: Algorithmic improvements for Norm/Reduction ops.

**LayerNorm**:
• Use Welford's single-pass algorithm instead of 2-pass
• Reduces memory reads from 3N to N

**Reductions**:
• Use warp shuffle for efficient block-level reduction
• Minimize shared memory usage

**When to Apply**:
• Norm/Reduction ops → try algorithmic improvements
• SKIP for: element-wise ops, Conv (kernel<=3), score > 1.0
""",

                "skip_conditions": [
                    "Element-wise ops",
                    "Conv with small kernels",
                    "score > 1.0 (already good)",
                ],
            },
        ],

        "early_exit_enabled": True,
    },

    # ------------------------------------------------------------------------
    # 3. Fusion-Compute: Matmul/GEMM + ops (37个)
    # ------------------------------------------------------------------------
    "Fusion-Compute": {
        "description": "计算密集型融合 (Matmul/GEMM + 后续ops)",
        "count": 37,

        "stages": [
            {
                "name": "stage1_optimize_primary_matmul",
                "description": "Optimize Primary Matmul",
                "focus": "Ensure Matmul is fast before fusion (inherit Compute-Intensive strategy)",

                "key_metrics": {
                    "compute_throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "dram_write_baseline": "dram__bytes_write.sum",
                },

                "optimization_guidance": """
**Goal**: Optimize primary Matmul ONLY (do not fuse yet).

**Approach**:
1. Apply Compute-Intensive stage1 optimizations
2. Use simple tiling: BLOCK_M/N=64-128, num_warps=4, num_stages=2
3. **Keep all subsequent ops in PyTorch** (sigmoid/sum/etc.)

**Example Structure**:
```python
def forward(x, weight):
    # Stage1: Only optimize matmul
    matmul_out = triton_matmul(x, weight)

    # Keep these in PyTorch (will fuse later)
    out = torch.sigmoid(matmul_out)
    out = torch.sum(out, dim=1)
    return out
```

**Early Exit Check**:
• If Matmul score < 0.5 → fusion won't help enough

**NCU**: Record dram_bytes_write as baseline
""",

                "early_exit_conditions": {
                    "poor_matmul_baseline": {
                        "check": "score < 0.5",
                        "message": "Matmul baseline too slow, fusion won't compensate",
                    }
                },
            },

            {
                "name": "stage2_fuse_elementwise",
                "description": "Fuse Element-wise Ops",
                "focus": "Eliminate intermediate writes by fusing activations",

                "key_metrics": {
                    "dram_write_reduction": "dram__bytes_write.sum",
                    "dram_total": "dram__bytes.sum",
                },

                "optimization_guidance": """
**Goal**: Fuse element-wise activation into Matmul (eliminate intermediate write).

**Implementation**:
• After `tl.dot(a, b)`, apply activation on accumulator **before storing**
• Use: `tl.sigmoid()`, `tl.maximum(acc, 0)`, `tl.tanh()`, etc.

**Example**:
```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(...):
    acc += tl.dot(a, b)

# Fuse activation here (no intermediate store)
acc = tl.sigmoid(acc)  # or tl.maximum(acc, 0) for ReLU

tl.store(c_ptr, acc)  # Single write
```

**NCU Expected**:
• dram_bytes_write should decrease by M×N×4 bytes (one less write)

**Speedup**: 10-30% if I/O bound, less if compute bound
""",
            },

            {
                "name": "stage3_fuse_reduction",
                "description": "Fuse Reduction Ops",
                "focus": "Output reduced result directly (sum/max/etc.)",

                "key_metrics": {
                    "output_size_reduction": "dram__bytes_write.sum",
                },

                "optimization_guidance": """
**Goal**: Fuse reduction (sum/max) to directly output reduced result.

**IMPORTANT - Complex Optimization**:
• This changes grid structure (2D → 1D)
• Requires careful index management
• **Only apply if stage2 fusion already successful**

**Key Changes**:
• Grid: `(M/BLOCK_M, N/BLOCK_N)` → `(M/BLOCK_M,)` (1D only)
• Loop over N tiles inside kernel
• Accumulate: `row_sum += tl.sum(tile, axis=1)`
• Output: [M, 1] instead of [M, N]

**NCU Expected**:
• dram_bytes_write: M×N×4 → M×4 (massive reduction)

**Warning**:
• Complex to implement correctly
• If unsure, prefer keeping stage2's simpler fusion

**Speedup**: 30-60% for reduction-heavy ops, but risky
""",
            },
        ],
    },

    # ------------------------------------------------------------------------
    # 4. Fusion-Memory: Conv + ops (63个)
    # ------------------------------------------------------------------------
    "Fusion-Memory": {
        "description": "访存密集型融合 (Conv + 后续ops)",
        "count": 63,

        "stages": [
            {
                "name": "stage1_optimize_primary_conv",
                "description": "Optimize Primary Conv",
                "focus": "Optimize Conv first, assess fusion potential",

                "key_metrics": {
                    "dram_throughput": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "dram_write_baseline": "dram__bytes_write.sum",
                },

                "optimization_guidance": """
**Goal**: Implement baseline Conv, assess if fusion makes sense.

**IMPORTANT - Conv Reality Check**:
• **Small kernels (size <= 3)**: Very hard to beat cuDNN
• If baseline score < 0.3 → fusion won't help enough, consider early exit
• Conv is memory-bound, focus on bandwidth not compute

**Approach**:
1. Implement im2col or direct convolution
2. Use vectorized memory access
3. **Keep BN/ReLU/Pooling in PyTorch** (fuse later)

**Realistic Expectations**:
• Score 0.5-0.8 is already good for Triton Conv vs cuDNN
• Don't expect 2x speedup unless kernel is large (size > 5)

**NCU**: Record dram_bytes_write as baseline
""",

                "early_exit_conditions": {
                    "conv_too_slow": {
                        "check": "kernel_size <= 3 AND score < 0.2",
                        "message": "Conv baseline too slow for small kernel, skip fusion",
                    }
                },
            },

            {
                "name": "stage2_fuse_normalization",
                "description": "Fuse BatchNorm/GroupNorm",
                "focus": "Apply normalization directly on Conv output",

                "key_metrics": {
                    "dram_write_reduction": "dram__bytes_write.sum",
                    "dram_read_reduction": "dram__bytes_read.sum",
                },

                "optimization_guidance": """
**Goal**: Fuse normalization (BatchNorm/GroupNorm) into Conv kernel.

**Key Idea**:
• Load BN parameters (mean, var, gamma, beta) per channel
• Apply transform: `(acc - mean) / sqrt(var + eps) * gamma + beta`
• Store normalized result (no intermediate Conv output)

**NCU Validation**:
• dram_bytes_write ↓ (no intermediate write)
• dram_bytes_read ↓ (BN doesn't read intermediate)
• Net reduction: ~20-40%

**Expected Speedup**: 20-40%
""",
            },

            {
                "name": "stage3_fuse_activation_pooling",
                "description": "Fuse Activation + Pooling",
                "focus": "Fully fuse all operations",

                "key_metrics": {
                    "final_dram_bytes": "dram__bytes.sum",
                },

                "optimization_guidance": """
**Goal**: Fuse activation (ReLU/Tanh) and pooling (MaxPool/AvgPool).

**Approach**:
• After BatchNorm, apply activation: `tl.maximum(acc, 0.0)` for ReLU
• For pooling: may need grid adjustment or separate pass depending on complexity
• Store final result only (all intermediates eliminated)

**NCU Validation**:
• dram_bytes ≈ input + weight + final_output only
• All intermediate tensors eliminated

**Expected Speedup**: 10-30% additional (total 0.8-1.8x vs PyTorch)
""",
            },
        ],

        "early_exit_enabled": True,
    },
}


# ============================================================================
# 辅助函数
# ============================================================================

def classify_operator(op_name: str, level: str) -> str:
    """分类算子，返回类别名称"""
    is_level2 = (level == "level2")

    if is_level2:
        match = re.match(r'\d+_(Conv\w+|Matmul|Gemm|BMM)', op_name)
        if match:
            primary = match.group(1)
            if any(op in primary for op in ["Matmul", "Gemm", "BMM"]):
                return "Fusion-Compute"
            elif "Conv" in primary:
                return "Fusion-Memory"
        return "Fusion-Memory"

    # Level1
    if any(op in op_name for op in ["Matmul", "Gemm", "BMM", "matrix_multiplication"]):
        return "Compute-Intensive"
    else:
        return "Memory-Intensive"


def get_stage_config(category: str, stage_id: int) -> Dict:
    """获取stage配置"""
    stages = OPERATOR_CATEGORIES[category]["stages"]
    if 0 <= stage_id < len(stages):
        return stages[stage_id]
    raise ValueError(f"Invalid stage_id {stage_id} for category {category}")


def build_stage_prompt_section(category: str, stage_id: int) -> str:
    """
    构建stage的prompt section (类似optimization.py的stage_focus_map)

    Returns:
        格式化的stage描述和优化指导
    """
    stage_config = get_stage_config(category, stage_id)

    return f"""
## Current Optimization Stage
**Stage**: {stage_config['description']}

{stage_config['optimization_guidance']}
""".strip()


def get_key_ncu_metrics(category: str, stage_id: int) -> Dict[str, str]:
    """
    获取当前stage需要关注的核心NCU指标

    Returns:
        {"metric_display_name": "ncu_metric_name"}
    """
    stage_config = get_stage_config(category, stage_id)
    return stage_config.get("key_metrics", {})


def check_early_exit(category: str, stage_id: int,
                     performance_score: float,
                     op_metadata: Dict) -> Tuple[bool, str]:
    """
    检查是否应该early exit

    Args:
        category: 算子类别
        stage_id: 当前stage
        performance_score: 性能得分 (vs PyTorch)
        op_metadata: {"op_type": "conv", "kernel_size": 3, ...}

    Returns:
        (should_exit, reason)
    """
    stage_config = get_stage_config(category, stage_id)
    conditions = stage_config.get("early_exit_conditions", {})

    for cond_name, cond_config in conditions.items():
        check_expr = cond_config["check"]

        # 简化版检查逻辑 (实际应该用更robust的parser)
        should_exit = False
        op_type_lower = op_metadata.get("op_type", "").lower()

        # Check element-wise ops
        if "op_type in [" in check_expr:
            # Extract list of op types from check expression
            elementwise_ops = ['relu', 'leakyrelu', 'sigmoid', 'tanh', 'gelu', 'swish']
            if any(op in op_type_lower for op in elementwise_ops):
                if "score <" in check_expr:
                    threshold = float(check_expr.split("score <")[1].strip())
                    if performance_score < threshold:
                        should_exit = True

        # Check conv ops
        elif "conv" in check_expr.lower():
            if "conv" in op_type_lower:
                kernel_size = op_metadata.get("kernel_size", 5)
                if "kernel_size <= 3" in check_expr and kernel_size <= 3:
                    if "score <" in check_expr:
                        threshold = float(check_expr.split("score <")[1].strip())
                        if performance_score < threshold:
                            should_exit = True

        # Generic score check
        elif "score <" in check_expr:
            threshold = float(check_expr.split("score <")[1].strip())
            if performance_score < threshold:
                should_exit = True

        if should_exit:
            return (True, cond_config["message"])

    return (False, "")


def should_skip_stage(category: str, stage_id: int, op_metadata: Dict) -> Tuple[bool, str]:
    """
    检查是否应该跳过当前stage

    Returns:
        (should_skip, reason)
    """
    stage_config = get_stage_config(category, stage_id)
    skip_conditions = stage_config.get("skip_conditions", [])

    for condition in skip_conditions:
        condition_lower = condition.lower()

        # 检查element-wise ops
        if "element-wise" in condition_lower:
            op_type = op_metadata.get("op_type", "").lower()
            if any(act in op_type for act in ["relu", "sigmoid", "tanh", "gelu", "swish"]):
                return (True, f"Element-wise op: {condition}")

        # 检查性能已经很好
        if "score >" in condition_lower:
            score = op_metadata.get("score", 0.0)
            threshold = 1.0  # 从condition中解析
            if score > threshold:
                return (True, condition)

    return (False, "")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'OPERATOR_CATEGORIES',
    'classify_operator',
    'get_stage_config',
    'build_stage_prompt_section',
    'get_key_ncu_metrics',
    'check_early_exit',
    'should_skip_stage',
]
