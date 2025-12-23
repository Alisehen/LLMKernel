# Algorithm Analysis Prompt 更新总结

## 问题

原本使用 `build_judger_optimization_prompts` 进行算法分析，但该prompt专注于**实现层面**优化：
- ❌ BLOCK_M/N/K tuning
- ❌ num_warps/num_stages
- ❌ Memory access patterns

这些都是**低层次的Triton参数调优**，不适合**算法结构分析**。

## 解决方案

创建专门的 `build_algorithm_analysis_prompt` (prompts/algorithm_analysis.py)，关注**高层次算法优化**：
- ✅ Operator Fusion
- ✅ Algorithm Replacement (Flash Attention, Winograd等)
- ✅ Computational Graph Reordering
- ✅ Kernel Launch Reduction
- ✅ Memory Layout Optimization

## 核心改进

### 1. **Evidence-based 分析流程**

要求LLM结合代码+metrics进行分析：

```
1. Code Structure Analysis:
   - 统计kernel数量
   - 识别操作类型
   - 检查实现方式

2. Performance Metrics Interpretation:
   - DRAM Throughput → 内存瓶颈
   - SM Utilization → 计算利用率
   - Kernel launch overhead → 过多小kernel

3. Combined Diagnosis:
   - 代码+指标 → 确定优化方向
```

### 2. **优化分类指导**

每种优化类型都说明了**触发条件**：

| 优化类型 | 触发条件 | 示例 |
|---------|---------|------|
| Operator Fusion | 多kernel + 高内存流量 | Conv→BN→ReLU融合 |
| Algorithm Replacement | 低SM利用率 + 朴素实现 | Attention → Flash Attention |
| Graph Reordering | 低cache命中率 | 调整操作顺序 |
| Launch Reduction | 多小kernel + 高overhead | 7 kernels → 3 kernels |
| Memory Optimization | 高DRAM流量 + 大中间张量 | In-place操作 |

### 3. **结构化输出**

新增字段确保分析有依据：

```json
{
  "code_observation": "从代码中观察到的问题",
  "metrics_observation": "从NCU指标中发现的瓶颈",
  "bottleneck": "综合诊断",
  "optimisation method": "优化方法",
  "modification plan": "实现步骤",
  "expected_speedup": "预期提升",
  "risk_level": "风险评估"
}
```

## 代码修改

### 1. 新文件
- `prompts/algorithm_analysis.py`: 算法分析prompt构建器

### 2. 修改文件
- `main.py`:
  - 导入 `build_algorithm_analysis_prompt`
  - 替换分析prompt生成函数
  - 更新输出显示（显示code/metrics observation）

## 使用示例

### Level 2/3任务（自动触发算法分析）

```bash
python main.py KernelBench/level3/8_ResNetBasicBlock.py \
    --model network \
    --num_seeds 2
```

**输出示例**:
```
[Algorithm Analysis] Analyzing best seed for algorithmic optimization...
[Algorithm Analysis] Best seed score: 1.0543
[Algorithm Analysis] Requesting LLM analysis...
[Algorithm Analysis] Code observation: Current implementation uses 7 separate kernel launches: conv1, bn1...
[Algorithm Analysis] Metrics observation: DRAM throughput is 78% (memory-bound), SM utilization only 45%...
[Algorithm Analysis] Bottleneck: Excessive kernel launches cause synchronization overhead
[Algorithm Analysis] Optimization: Fuse residual block into 3 kernels
[Algorithm Analysis] Expected speedup: 30-40%
[Algorithm Analysis] Optimized seed score: 1.3821 ✓
```

## 对比分析

### 旧Prompt (judger_optimization)

```
Focus on Triton-specific optimizations:
- BLOCK_M/N/K tuning: Adjust tile sizes...
- num_warps: Control occupancy...
- num_stages: Enable software pipelining...
```

❌ **问题**: 太底层，无法识别算法级优化机会

### 新Prompt (algorithm_analysis)

```
Analysis Approach:
1. Code Structure Analysis: Count kernels, identify ops...
2. Performance Metrics: DRAM/SM throughput...
3. Combined Diagnosis: Code + metrics → optimization

Optimization Categories:
- Operator Fusion (when: many kernels + high memory)
- Algorithm Replacement (when: low SM + naive impl)
...
```

✅ **改进**:
- Evidence-based（基于代码+metrics）
- 高层次（算法而非参数）
- 有指导性（何时用何种优化）

## 预期效果

### Level 2/3
- **成功率**: 50% → 70%+
- **加速比**: 提升30-50%
- **优化案例**:
  - ResNet Block: 7 kernels → 3 fused kernels
  - Transformer: QKV分离 → QKV融合
  - Attention: Standard → Flash Attention

### Level 1
- 跳过算法分析（触发条件不满足）
- 保持原有性能

## 论文贡献

这个改进支持以下论文论点：

1. **三层优化架构**:
   - Layer 1: 算法分析（高层）← **新增**
   - Layer 2: 三阶段优化（中层）
   - Layer 3: 参数调优（底层）

2. **Evidence-based AI**:
   - LLM不是随意猜测
   - 基于代码分析+性能指标的理性决策
   - 可解释性强（显示code/metrics observation）

3. **分层优化策略**:
   - Level 1/2: 轻量级（跳过或简单分析）
   - Level 3: 重量级（完整算法分析）
   - 适应性强

## 测试建议

### 单元测试
```bash
# 测试prompt生成
python prompts/algorithm_analysis.py \
    KernelBench/level3/8_ResNetBasicBlock.py \
    --cuda_code results/level3/8_ResNetBasicBlock/code/kernel_seed.py \
    -o test_prompt.txt
```

### 集成测试
```bash
# 测试完整流程
python main.py KernelBench/level3/8_ResNetBasicBlock.py \
    --model network \
    --num_seeds 2 \
    --round 5
```

### 对比实验
```bash
# 修改main.py临时禁用算法分析（注释掉line 1025-1156）
# 对比有/无算法分析的效果
```

## 下一步

- [ ] 实验验证level3提升效果
- [ ] 收集5-10个典型案例
- [ ] 优化prompt（根据实验结果）
- [ ] 论文撰写（方法章节+实验章节）
- [ ] Ablation study设计

## 关键文件

- **Prompt定义**: `prompts/algorithm_analysis.py`
- **调用位置**: `main.py` line 1053-1111
- **日志输出**: `io_dir/algorithm_analysis_prompt.txt` 和 `algorithm_analysis_result.txt`
- **文档说明**: `ALGORITHM_ANALYSIS_CHANGES.md`
