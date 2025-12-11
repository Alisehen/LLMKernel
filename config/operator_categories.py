"""
算子分类配置与优化策略

四大类别：
1. Compute-Intensive: Matmul/GEMM/BMM (16个)
2. Memory-Intensive: Conv/Activation/Norm/Reduction (84个)
3. Fusion-Compute: Matmul/GEMM + 融合ops (37个)
4. Fusion-Memory: Conv + 融合ops (63个)
"""

import re
from typing import Dict, List, Tuple, Optional

# ============================================================================
# 算子分类配置
# ============================================================================

OPERATOR_CATEGORIES = {

    # ------------------------------------------------------------------------
    # 1. Compute-Intensive (16个): Matmul/GEMM/BMM
    # ------------------------------------------------------------------------
    "Compute-Intensive": {
        "description": "计算密集型算子 (Matmul/GEMM/BMM)",
        "count": 16,
        "primary_ops": ["Matmul", "Gemm", "BMM", "matrix_multiplication"],

        "stages": [
            {
                "id": 1,
                "name": "stage1_tiling_and_shared_memory",
                "description": "实现2D block tiling + shared memory缓存",
                "focus": "基础架构：grid设计、shared memory、tl.dot()",

                "ncu_metrics": {
                    # 计算相关
                    "compute_throughput": {
                        "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "> 30%",
                        "meaning": "SM整体利用率"
                    },
                    "fma_instructions": {
                        "metric": "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
                        "target": "≈ M*N*K*2 (每个FMA = 2 FLOPs)",
                        "meaning": "融合乘加指令数"
                    },
                    # Shared memory使用
                    "shared_memory_used": {
                        "metric": "smsp__inst_executed_op_shared.sum",
                        "target": "> 0",
                        "meaning": "确认使用了shared memory"
                    },
                    "shared_efficiency": {
                        "metric": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
                        "target": "良好",
                        "meaning": "Shared memory load transactions"
                    },
                    # 访存基础指标
                    "global_load_throughput": {
                        "metric": "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second",
                        "target": "记录baseline",
                        "meaning": "Global memory读取吞吐"
                    },
                    "dram_bytes": {
                        "metric": "dram__bytes.sum",
                        "target": "≈ (M*K + K*N + M*N) * 4字节",
                        "meaning": "DRAM总访存量"
                    },
                },

                "optimization_tips": [
                    "实现标准的2D tiling: grid = (M/BLOCK_M, N/BLOCK_N)",
                    "使用shared memory缓存A和B的tile",
                    "初始BLOCK_M=BLOCK_N=64, BLOCK_K=32",
                    "使用tl.dot(a, b)进行矩阵乘法",
                    "正确处理边界条件 (mask)",
                ],

                "expected_improvement": "baseline，可能慢于PyTorch 1-3x",
            },

            {
                "id": 2,
                "name": "stage2_block_size_tuning",
                "description": "通过autotune优化BLOCK_M/N/K和num_warps",
                "focus": "性能调优：找到最优block配置",

                "ncu_metrics": {
                    # 计算效率提升
                    "compute_throughput": {
                        "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "↑ 10-20%",
                        "meaning": "SM利用率应提升"
                    },
                    "achieved_occupancy": {
                        "metric": "sm__warps_active.avg.pct_of_peak_sustained_active",
                        "target": "> 50%",
                        "meaning": "Warp占用率"
                    },
                    # Pipeline效率
                    "pipeline_active": {
                        "metric": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
                        "target": "↑",
                        "meaning": "FMA流水线活跃度"
                    },
                    "num_stages_effect": {
                        "metric": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct",
                        "target": "↓",
                        "meaning": "记分板停顿减少 (pipeline隐藏延迟)"
                    },
                    # 访存效率
                    "l1_hit_rate": {
                        "metric": "l1tex__t_sector_hit_rate.pct",
                        "target": "> 80%",
                        "meaning": "L1缓存命中率"
                    },
                    "memory_efficiency": {
                        "metric": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                        "target": "> 80%",
                        "meaning": "全局内存访问效率 (coalescing)"
                    },
                },

                "optimization_tips": [
                    "使用@triton.autotune with 4-6个配置",
                    "尝试BLOCK_M/N: 64, 128, 256",
                    "尝试BLOCK_K: 32, 64, 128",
                    "调整num_warps: 4, 8, 16",
                    "调整num_stages: 2, 3, 4 (pipeline depth)",
                    "根据NCU的occupancy和pipeline stall调整",
                ],

                "expected_improvement": "20-50% 性能提升",
            },

            {
                "id": 3,
                "name": "stage3_vectorized_load_and_swizzle",
                "description": "向量化访存 + block swizzling减少L2冲突",
                "focus": "高级优化：访存模式、L2缓存、数据重用",

                "ncu_metrics": {
                    # L2缓存优化
                    "l2_hit_rate": {
                        "metric": "lts__t_sector_hit_rate.pct",
                        "target": "↑ 5-10%",
                        "meaning": "L2缓存命中率提升 (swizzle效果)"
                    },
                    "l2_read_throughput": {
                        "metric": "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum.per_second",
                        "target": "↓",
                        "meaning": "L2读取减少 (更多L1命中)"
                    },
                    # DRAM访问
                    "dram_read_bytes": {
                        "metric": "dram__bytes_read.sum",
                        "target": "↓ 10-20%",
                        "meaning": "DRAM读取量减少"
                    },
                    "dram_utilization": {
                        "metric": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "< 70%",
                        "meaning": "非DRAM瓶颈"
                    },
                    # 向量化效果
                    "vectorized_loads": {
                        "metric": "smsp__inst_executed_op_global_ld.sum",
                        "target": "↓ (wider loads)",
                        "meaning": "Load指令数减少 (向量化)"
                    },
                    # 整体性能
                    "ipc": {
                        "metric": "smsp__average_inst_executed_per_warp.ratio",
                        "target": "↑",
                        "meaning": "每warp执行指令数增加"
                    },
                },

                "optimization_tips": [
                    "实现block swizzling (group_size=8)",
                    "增大BLOCK_K以提高数据重用",
                    "使用tl.trans()优化矩阵转置",
                    "考虑使用eviction policy (如supported)",
                    "验证tl.load的向量化宽度",
                ],

                "expected_improvement": "10-30% 性能提升",
            },
        ],

        "total_stages": 3,
        "fallback_threshold": 0.5,  # 如果stage3后仍 < 0.5x PyTorch，标记为不适合优化
    },

    # ------------------------------------------------------------------------
    # 2. Memory-Intensive (84个): Conv/Activation/Norm/Reduction
    # ------------------------------------------------------------------------
    "Memory-Intensive": {
        "description": "访存密集型算子 (Conv/Activation/Norm/Reduction)",
        "count": 84,
        "primary_ops": ["conv", "Conv", "ReLU", "Sigmoid", "Tanh", "Softmax", "GELU",
                       "BatchNorm", "LayerNorm", "GroupNorm", "RMSNorm", "InstanceNorm",
                       "Pool", "Loss", "cumsum", "cumprod", "Attention", "Norm"],

        "stages": [
            {
                "id": 1,
                "name": "stage1_baseline_and_vectorization",
                "description": "基础实现 + 向量化访存 + 可行性评估",
                "focus": "建立baseline，评估是否值得继续优化",

                "ncu_metrics": {
                    # 访存带宽
                    "dram_throughput": {
                        "metric": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "> 40%",
                        "meaning": "DRAM带宽利用率"
                    },
                    "dram_bytes_total": {
                        "metric": "dram__bytes.sum",
                        "target": "接近理论最小值",
                        "meaning": "总DRAM访存量"
                    },
                    # 访存合并
                    "memory_coalescing": {
                        "metric": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                        "target": "> 60%",
                        "meaning": "Load操作的访存效率"
                    },
                    "global_load_efficiency": {
                        "metric": "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct",
                        "target": "> 60%",
                        "meaning": "Store操作的访存效率"
                    },
                    # 计算强度 (判断瓶颈)
                    "compute_throughput": {
                        "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "记录baseline",
                        "meaning": "如果很低，说明是memory-bound"
                    },
                    "arithmetic_intensity": {
                        "metric": "calculated",  # FLOPs / DRAM_bytes
                        "target": "< 50 FLOP/Byte",
                        "meaning": "计算访存比 (低则为memory-bound)"
                    },
                },

                "optimization_tips": [
                    "对于Element-wise: 简单的grid划分，向量化load/store",
                    "对于Conv: 判断kernel_size，小kernel (<= 3) 考虑early exit",
                    "对于Norm: 实现标准的两遍或三遍算法",
                    "对于Reduction: block-level reduction",
                    "确保访存是coalesced的 (连续访问)",
                    "使用BLOCK_SIZE=256或512",
                ],

                "expected_improvement": "baseline, 可能慢于PyTorch 2-10x (Conv更慢)",

                "early_exit_check": {
                    "condition": "算子类型包含'conv'且kernel_size<=3且score<0.3",
                    "action": "建议跳过，使用cuDNN",
                    "reason": "小kernel的Conv优化空间有限"
                }
            },

            {
                "id": 2,
                "name": "stage2_memory_hierarchy_optimization",
                "description": "利用shared memory缓存 (适用于Conv/Norm)",
                "focus": "优化内存层次，减少global memory访问",

                "ncu_metrics": {
                    # Global memory访问减少
                    "global_load_bytes": {
                        "metric": "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
                        "target": "↓ 15-30%",
                        "meaning": "Global load字节数减少"
                    },
                    "global_load_transactions": {
                        "metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                        "target": "↓",
                        "meaning": "Load transactions减少"
                    },
                    # Shared memory使用
                    "shared_load_transactions": {
                        "metric": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
                        "target": "↑ (新增)",
                        "meaning": "Shared memory读取增加"
                    },
                    "shared_store_transactions": {
                        "metric": "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
                        "target": "↑ (新增)",
                        "meaning": "Shared memory写入增加"
                    },
                    "shared_efficiency": {
                        "metric": "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed",
                        "target": "> 50%",
                        "meaning": "Shared memory利用率"
                    },
                    # L1/L2缓存
                    "l1_hit_rate": {
                        "metric": "l1tex__t_sector_hit_rate.pct",
                        "target": "↑",
                        "meaning": "L1命中率提升"
                    },
                    # 带宽优化
                    "dram_read_bytes": {
                        "metric": "dram__bytes_read.sum",
                        "target": "↓ 10-20%",
                        "meaning": "DRAM读取减少"
                    },
                },

                "optimization_tips": [
                    "对于Conv: shared memory缓存input feature map tile",
                    "对于LayerNorm: 缓存中间统计量 (mean/var)",
                    "对于Reduction: shared memory做block-level reduction",
                    "注意shared memory bank conflict",
                    "Element-wise ops跳过此stage (无需shared memory)",
                    "合理设置shared memory大小，避免occupancy下降",
                ],

                "expected_improvement": "Conv: 20-50%; Norm: 30-60%; Element-wise: 跳过",

                "skip_conditions": [
                    "Element-wise activation ops (ReLU, Sigmoid等)",
                    "已经很快的算子 (score > 0.9)",
                ]
            },

            {
                "id": 3,
                "name": "stage3_advanced_patterns",
                "description": "Warp-level优化 (Norm/Reduction) 或 算法改进",
                "focus": "高级优化：warp shuffle, Welford, 单遍算法",

                "ncu_metrics": {
                    # Warp执行效率
                    "warp_execution_efficiency": {
                        "metric": "smsp__sass_average_branch_targets_threads_uniform.pct",
                        "target": "> 95%",
                        "meaning": "Warp内线程执行一致性 (分支一致)"
                    },
                    "warp_stall_barrier": {
                        "metric": "smsp__average_warps_issue_stalled_barrier_per_issue_active.pct",
                        "target": "↓",
                        "meaning": "Barrier停顿减少"
                    },
                    # Reduction效率
                    "reduction_efficiency": {
                        "metric": "smsp__inst_executed_op_shared.sum",
                        "target": "观察变化",
                        "meaning": "Shared memory操作 (warp shuffle可能减少)"
                    },
                    # 指令效率
                    "issued_ipc": {
                        "metric": "smsp__inst_issued.avg.per_cycle_active",
                        "target": "↑",
                        "meaning": "每周期发射指令数"
                    },
                    # 访存次数 (单遍算法)
                    "global_load_passes": {
                        "metric": "calculated",  # 通过kernel分析
                        "target": "3遍 → 1遍 (Welford)",
                        "meaning": "数据遍数减少"
                    },
                    # 整体性能
                    "compute_throughput": {
                        "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "↑",
                        "meaning": "整体吞吐提升"
                    },
                },

                "optimization_tips": [
                    "对于LayerNorm: 使用Welford单遍算法",
                    "对于Reduction: 使用warp shuffle (tl.reduce)",
                    "对于Conv: 尝试L2 cache eviction hints (如不报错)",
                    "优化warp-level的执行模式",
                    "减少同步点 (__syncthreads / barrier)",
                    "Element-wise和简单Conv跳过此stage",
                ],

                "expected_improvement": "Norm/Reduction: 20-40%; Conv: 10-20%或跳过",

                "skip_conditions": [
                    "Element-wise ops",
                    "Conv with kernel_size <= 3",
                    "已经达到性能目标 (score > 1.0)",
                ]
            },
        ],

        "total_stages": 3,
        "early_exit_enabled": True,
        "early_exit_threshold": 0.3,  # Conv类如果stage1 < 0.3，建议退出
    },

    # ------------------------------------------------------------------------
    # 3. Fusion-Compute (37个): Matmul/GEMM + 后续ops
    # ------------------------------------------------------------------------
    "Fusion-Compute": {
        "description": "计算密集型融合算子 (Matmul/GEMM + 后续ops)",
        "count": 37,
        "primary_ops": ["Matmul", "Gemm", "BMM"],
        "fusion_patterns": ["Sigmoid", "ReLU", "Tanh", "Sum", "Max", "Softmax", "BatchNorm"],

        "stages": [
            {
                "id": 1,
                "name": "stage1_optimize_primary_matmul",
                "description": "先优化主算子Matmul (继承Compute-Intensive)",
                "focus": "确保Matmul本身性能良好",

                "ncu_metrics": {
                    # 继承Compute-Intensive的stage1+2指标
                    "compute_throughput": {
                        "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "> 40%",
                        "meaning": "SM利用率"
                    },
                    "fma_instructions": {
                        "metric": "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
                        "target": "≈ M*N*K*2",
                        "meaning": "FMA指令数正确"
                    },
                    "matmul_efficiency": {
                        "metric": "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
                        "target": "低 (应使用FMA)",
                        "meaning": "应使用融合乘加"
                    },
                    # 访存baseline (后续阶段对比)
                    "baseline_dram_write": {
                        "metric": "dram__bytes_write.sum",
                        "target": "≈ M*N*4 (中间结果)",
                        "meaning": "记录baseline，后续应减少"
                    },
                    "baseline_dram_total": {
                        "metric": "dram__bytes.sum",
                        "target": "记录",
                        "meaning": "总访存量baseline"
                    },
                },

                "optimization_tips": [
                    "实现完整的Matmul kernel (2D tiling + autotune)",
                    "暂不融合后续ops，先确保Matmul接近PyTorch性能",
                    "使用Compute-Intensive的stage1和stage2的最佳实践",
                    "验证输出正确性",
                    "如果Matmul本身很慢 (< 0.5x)，考虑不继续融合",
                ],

                "expected_improvement": "Matmul部分接近PyTorch (0.8-1.2x)",
            },

            {
                "id": 2,
                "name": "stage2_fuse_elementwise_ops",
                "description": "融合element-wise ops (sigmoid/relu/tanh等)",
                "focus": "消除Matmul输出的global memory写入",

                "ncu_metrics": {
                    # 访存减少 (关键指标)
                    "dram_write_reduction": {
                        "metric": "dram__bytes_write.sum",
                        "target": "↓ M*N*4字节",
                        "meaning": "消除了Matmul中间结果写入"
                    },
                    "dram_read_reduction": {
                        "metric": "dram__bytes_read.sum",
                        "target": "↓ M*N*4字节",
                        "meaning": "消除了element-wise op的输入读取"
                    },
                    "dram_total_reduction": {
                        "metric": "dram__bytes.sum",
                        "target": "↓ 20-30%",
                        "meaning": "总访存量显著减少"
                    },
                    # 计算开销 (应该很小)
                    "activation_overhead": {
                        "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "几乎不变或略降",
                        "meaning": "Activation计算开销可忽略"
                    },
                    # Kernel数量
                    "kernel_launches": {
                        "metric": "external",  # 运行时统计
                        "target": "减少1个kernel",
                        "meaning": "Kernel launch开销减少"
                    },
                    # 整体性能
                    "end_to_end_speedup": {
                        "metric": "calculated",
                        "target": "1.1 - 1.3x",
                        "meaning": "端到端加速比"
                    },
                },

                "optimization_tips": [
                    "在tl.dot()累加器上直接应用activation",
                    "支持的ops: tl.sigmoid, tl.maximum(x,0), tl.tanh, tl.exp等",
                    "不要写回中间结果到global memory",
                    "保持autotune配置不变",
                    "示例: acc = tl.sigmoid(acc) 在tl.store前",
                    "验证数值精度 (与分离实现对比)",
                ],

                "expected_improvement": "10-30% (element-wise越多，收益越大)",
            },

            {
                "id": 3,
                "name": "stage3_fuse_reduction_ops",
                "description": "融合reduction ops (sum/max/softmax等)",
                "focus": "直接输出reduction结果，避免存储完整矩阵",

                "ncu_metrics": {
                    # 访存大幅减少 (最大收益)
                    "output_size_reduction": {
                        "metric": "dram__bytes_write.sum",
                        "target": "M*N*4 → M*1*4 (sum) 或更小",
                        "meaning": "输出大小显著减少"
                    },
                    "dram_total_reduction": {
                        "metric": "dram__bytes.sum",
                        "target": "↓ 40-60%",
                        "meaning": "总访存量大幅减少"
                    },
                    # Grid变化
                    "grid_dimension": {
                        "metric": "external",  # kernel配置
                        "target": "2D → 1D (只在M维度并行)",
                        "meaning": "Grid变为1D，每个block处理一整行"
                    },
                    # Reduction效率
                    "reduction_instructions": {
                        "metric": "smsp__inst_executed_op_shared.sum",
                        "target": "增加 (block内reduction)",
                        "meaning": "使用shared memory做reduction"
                    },
                    "warp_efficiency": {
                        "metric": "smsp__sass_average_branch_targets_threads_uniform.pct",
                        "target": "> 90%",
                        "meaning": "Reduction时warp执行一致"
                    },
                    # 整体加速
                    "end_to_end_speedup": {
                        "metric": "calculated",
                        "target": "1.3 - 2.0x vs 分离实现",
                        "meaning": "显著的融合收益"
                    },
                },

                "optimization_tips": [
                    "对于row-wise sum: 使用tl.sum(acc, axis=1)",
                    "对于row-wise max: 使用tl.max(acc, axis=1)",
                    "对于softmax: 先max, 再exp和sum, 最后除",
                    "调整grid为1D: grid=(M/BLOCK_M,)",
                    "在N维度循环，累积到row_acc",
                    "示例: row_sum += tl.sum(sigmoid(acc), axis=1)",
                    "注意：输出shape变化，需要调整",
                ],

                "expected_improvement": "30-60% (reduction越重，收益越大)",
            },
        ],

        "total_stages": 3,
        "fallback_threshold": 0.5,  # 如果stage1的Matmul < 0.5x，建议不继续
    },

    # ------------------------------------------------------------------------
    # 4. Fusion-Memory (63个): Conv + 后续ops
    # ------------------------------------------------------------------------
    "Fusion-Memory": {
        "description": "访存密集型融合算子 (Conv + 后续ops)",
        "count": 63,
        "primary_ops": ["Conv2d", "Conv3d", "ConvTranspose2d", "ConvTranspose3d"],
        "fusion_patterns": ["BatchNorm", "GroupNorm", "ReLU", "Tanh", "MaxPool", "Add", "Multiply"],

        "stages": [
            {
                "id": 1,
                "name": "stage1_optimize_primary_conv",
                "description": "先优化主算子Conv (继承Memory-Intensive)",
                "focus": "评估Conv baseline，判断融合是否值得",

                "ncu_metrics": {
                    # 继承Memory-Intensive的stage1指标
                    "dram_throughput": {
                        "metric": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "> 40%",
                        "meaning": "带宽利用率"
                    },
                    "memory_coalescing": {
                        "metric": "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
                        "target": "> 60%",
                        "meaning": "访存效率"
                    },
                    # Conv特定指标
                    "conv_efficiency": {
                        "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "记录baseline",
                        "meaning": "Conv计算效率"
                    },
                    # 访存baseline (后续对比)
                    "baseline_dram_write": {
                        "metric": "dram__bytes_write.sum",
                        "target": "记录Conv输出大小",
                        "meaning": "后续融合应减少"
                    },
                    "baseline_dram_read": {
                        "metric": "dram__bytes_read.sum",
                        "target": "记录",
                        "meaning": "Input + weight访存"
                    },
                },

                "optimization_tips": [
                    "实现基础Conv (向量化 + 合理grid)",
                    "暂不融合后续ops",
                    "对于小kernel (<=3): 如果很慢考虑early exit",
                    "使用Memory-Intensive的stage1最佳实践",
                    "记录baseline性能以评估融合价值",
                ],

                "expected_improvement": "baseline, 可能慢3-10x vs cuDNN",

                "early_exit_check": {
                    "condition": "kernel_size<=3 且 score<0.2",
                    "action": "建议跳过整个算子",
                    "reason": "Conv太慢，融合也无法弥补"
                }
            },

            {
                "id": 2,
                "name": "stage2_fuse_normalization",
                "description": "融合BatchNorm/GroupNorm到Conv输出",
                "focus": "消除Conv输出写入和Norm输入读取",

                "ncu_metrics": {
                    # 访存减少
                    "dram_write_reduction": {
                        "metric": "dram__bytes_write.sum",
                        "target": "↓ conv_output_size",
                        "meaning": "不存储未归一化的Conv输出"
                    },
                    "dram_read_reduction": {
                        "metric": "dram__bytes_read.sum",
                        "target": "↓ conv_output_size",
                        "meaning": "Norm不需要读取Conv输出"
                    },
                    "dram_total_reduction": {
                        "metric": "dram__bytes.sum",
                        "target": "↓ 20-40%",
                        "meaning": "总访存量减少"
                    },
                    # BN参数访问
                    "bn_param_loads": {
                        "metric": "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
                        "target": "增加少量 (mean/var/gamma/beta)",
                        "meaning": "BN参数访问开销小"
                    },
                    # Kernel数量
                    "kernel_count": {
                        "metric": "external",
                        "target": "减少1个",
                        "meaning": "Conv+BN合为1个kernel"
                    },
                },

                "optimization_tips": [
                    "在Conv kernel内，tl.store前应用BN变换",
                    "预加载BN参数: mean, var, gamma, beta",
                    "公式: out = (conv_out - mean) / sqrt(var+eps) * gamma + beta",
                    "BN参数在channel维度，每个channel加载一次",
                    "对于GroupNorm: 需要先计算group统计量",
                    "确保数值稳定性 (eps=1e-5)",
                ],

                "expected_improvement": "20-40% (消除Conv输出读写)",
            },

            {
                "id": 3,
                "name": "stage3_fuse_activation_and_pooling",
                "description": "融合activation和pooling (如适用)",
                "focus": "完全融合所有ops，无中间结果",

                "ncu_metrics": {
                    # 访存极简化
                    "final_dram_bytes": {
                        "metric": "dram__bytes.sum",
                        "target": "≈ input + weight + final_output",
                        "meaning": "无中间tensor访存"
                    },
                    "intermediate_eliminated": {
                        "metric": "calculated",
                        "target": "100%",
                        "meaning": "所有中间结果消除"
                    },
                    # Activation开销
                    "activation_cost": {
                        "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                        "target": "几乎不变",
                        "meaning": "Activation计算可忽略"
                    },
                    # Pooling (如有)
                    "pooling_overhead": {
                        "metric": "smsp__inst_executed_op_shared.sum",
                        "target": "观察",
                        "meaning": "Pooling可能需要shared memory"
                    },
                    # 整体加速
                    "total_speedup": {
                        "metric": "calculated",
                        "target": "0.8 - 1.8x vs PyTorch分离",
                        "meaning": "融合总收益 (取决于Conv baseline)"
                    },
                },

                "optimization_tips": [
                    "在BN后直接应用activation: tl.maximum(bn_out, 0)",
                    "对于MaxPool: 需要调整grid，或在循环内处理邻域",
                    "对于AvgPool: 累加邻域并除以pool_size",
                    "对于Add/Multiply: 读取residual并融合",
                    "完整流程: Conv → BN → Act → Pool，全在一个kernel",
                    "验证最终输出与PyTorch一致",
                ],

                "expected_improvement": "10-30% (在stage2基础上)",
            },
        ],

        "total_stages": 3,
        "early_exit_enabled": True,
        "early_exit_threshold": 0.2,  # Conv baseline < 0.2x时退出
    },
}


# ============================================================================
# 辅助函数
# ============================================================================

def classify_operator(op_name: str, level: str) -> Tuple[str, Optional[str]]:
    """
    对算子进行分类

    Args:
        op_name: 算子文件名 (不含.py)
        level: "level1" 或 "level2"

    Returns:
        (category, subcategory)
        例如: ("Fusion-Compute", None) 或 ("Memory-Intensive", None)
    """
    is_level2 = (level == "level2")

    # Level2全是融合算子
    if is_level2:
        # 提取主算子
        match = re.match(r'\d+_(Conv\w+|Matmul|Gemm|BMM)', op_name)
        if match:
            primary = match.group(1)
            if any(op in primary for op in ["Matmul", "Gemm", "BMM"]):
                return ("Fusion-Compute", None)
            elif "Conv" in primary:
                return ("Fusion-Memory", None)
        # 默认归为Fusion-Memory (大部分是Conv)
        return ("Fusion-Memory", None)

    # Level1算子分类
    if any(op in op_name for op in OPERATOR_CATEGORIES["Compute-Intensive"]["primary_ops"]):
        return ("Compute-Intensive", None)
    else:
        return ("Memory-Intensive", None)


def get_stage_config(category: str, stage_id: int) -> Dict:
    """
    获取指定类别和阶段的配置

    Args:
        category: 算子类别
        stage_id: 阶段ID (1-3)

    Returns:
        stage配置字典
    """
    if category not in OPERATOR_CATEGORIES:
        raise ValueError(f"Unknown category: {category}")

    stages = OPERATOR_CATEGORIES[category]["stages"]
    for stage in stages:
        if stage["id"] == stage_id:
            return stage

    raise ValueError(f"Stage {stage_id} not found in category {category}")


def extract_ncu_metrics_for_stage(ncu_report: Dict, category: str, stage_id: int) -> Dict:
    """
    从NCU报告中提取当前阶段关注的指标

    Args:
        ncu_report: NCU完整报告 (dict格式)
        category: 算子类别
        stage_id: 阶段ID

    Returns:
        {
            "metric_name": {
                "value": float,
                "target": str,
                "meaning": str,
                "status": "good" | "warning" | "bad"
            }
        }
    """
    stage_config = get_stage_config(category, stage_id)
    ncu_metrics_config = stage_config["ncu_metrics"]

    extracted = {}

    for metric_name, config in ncu_metrics_config.items():
        metric_key = config["metric"]

        # 跳过计算型指标和外部指标
        if metric_key in ["calculated", "external"]:
            extracted[metric_name] = {
                "value": None,
                "target": config["target"],
                "meaning": config["meaning"],
                "status": "info"
            }
            continue

        # 从NCU报告中查找指标值
        value = ncu_report.get(metric_key, None)

        # 评估状态 (简化版，实际需要更复杂的逻辑)
        status = "info"
        if value is not None:
            # TODO: 根据target判断good/warning/bad
            pass

        extracted[metric_name] = {
            "value": value,
            "target": config["target"],
            "meaning": config["meaning"],
            "status": status
        }

    return extracted


def should_early_exit(category: str, stage_id: int, performance_score: float,
                     operator_metadata: Dict) -> Tuple[bool, str]:
    """
    判断是否应该early exit

    Args:
        category: 算子类别
        stage_id: 当前阶段
        performance_score: 性能得分 (相对于PyTorch)
        operator_metadata: 算子元数据 (如kernel_size等)

    Returns:
        (should_exit, reason)
    """
    cat_config = OPERATOR_CATEGORIES[category]

    # 检查是否启用early exit
    if not cat_config.get("early_exit_enabled", False):
        return (False, "")

    threshold = cat_config.get("early_exit_threshold", 0.3)

    # Memory-Intensive和Fusion-Memory的特殊检查
    if category in ["Memory-Intensive", "Fusion-Memory"]:
        # Conv类算子，小kernel且性能差
        if "conv" in operator_metadata.get("op_type", "").lower():
            kernel_size = operator_metadata.get("kernel_size", 5)
            if kernel_size <= 3 and performance_score < threshold:
                return (True, f"Conv with small kernel ({kernel_size}) and poor performance ({performance_score:.2f}x)")

    # Fusion-Compute的特殊检查
    if category == "Fusion-Compute" and stage_id == 1:
        # Matmul baseline太慢
        if performance_score < cat_config.get("fallback_threshold", 0.5):
            return (True, f"Matmul baseline too slow ({performance_score:.2f}x), fusion won't help")

    return (False, "")


def get_optimization_tips_for_stage(category: str, stage_id: int,
                                    ncu_analysis: Optional[Dict] = None) -> List[str]:
    """
    获取当前阶段的优化建议

    Args:
        category: 算子类别
        stage_id: 阶段ID
        ncu_analysis: NCU分析结果 (可选，用于生成动态建议)

    Returns:
        优化建议列表
    """
    stage_config = get_stage_config(category, stage_id)
    tips = stage_config["optimization_tips"].copy()

    # TODO: 根据NCU分析添加动态建议
    if ncu_analysis:
        # 例如：如果occupancy低，建议调整block size
        pass

    return tips


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'OPERATOR_CATEGORIES',
    'classify_operator',
    'get_stage_config',
    'extract_ncu_metrics_for_stage',
    'should_early_exit',
    'get_optimization_tips_for_stage',
]
