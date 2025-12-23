# Algorithm Analysis Module Integration

## 修改概述

本次修改在seed生成之后、三阶段优化之前，插入了**算法结构分析模块**，用于level3（网络级）任务的高层次优化。

## 主要修改

### 1. 添加算法分析阶段（仅针对level3）

**位置**: `main.py` line 1028-1155

**功能**:
- 在所有seed生成并修复后，对于`MODEL_NETWORK`（level3）任务，执行算法分析
- 选择best seed进行NCU profiling
- 使用`build_judger_optimization_prompts`生成分析prompt
- LLM返回JSON格式的算法级优化建议：
  - `bottleneck`: 性能瓶颈
  - `optimisation method`: 优化方法
  - `modification plan`: 修改计划
- 基于分析结果生成优化后的seed
- 如果优化成功，将新seed加入候选池

**日志输出**:
```
[Algorithm Analysis] Analyzing best seed for algorithmic optimization...
[Algorithm Analysis] Best seed score: X.XXXX
[Algorithm Analysis] Requesting LLM analysis...
[Algorithm Analysis] Bottleneck: ...
[Algorithm Analysis] Optimization: ...
[Algorithm Analysis] Optimized seed score: X.XXXX ✓
```

**保存文件**:
- `io_dir/algorithm_analysis_prompt.txt`: 分析prompt
- `io_dir/algorithm_analysis_result.txt`: LLM返回的分析结果
- `code_dir/test_kernel_analysis.py`: 用于分析的kernel代码

### 2. 删除搜索相关功能

**删除内容**:
- `--beam_width` 参数（已标记为deprecated）
- `--beam_temperature` 参数（已标记为deprecated）
- 所有"beam search"相关注释

**保留内容**:
- `BeamCandidate` 类名（保持向后兼容）
- `--num_seeds` 参数（用于多seed生成）
- `--elimination_threshold` 参数（用于候选筛选）

### 3. 清理命名和注释

**修改前**:
```python
# ---------------------- Beam Search Data Structure -----------------
class BeamCandidate:
    """A candidate in beam search, holding kernel and its evaluation results."""

# Legacy beam search parameters (deprecated, kept for backward compatibility)
p.add_argument("--beam_width", ...)
p.add_argument("--beam_temperature", ...)
```

**修改后**:
```python
# ---------------------- Candidate Data Structure -----------------
class BeamCandidate:
    """A kernel candidate holding kernel code and its evaluation results."""

# Multi-seed parameters
p.add_argument("--num_seeds", ...)
p.add_argument("--elimination_threshold", ...)
```

**其他修改**:
- `beam` → `candidates` (变量名)
- `beam_idx` → `cand_idx` (大部分位置)
- 注释中的"beam search diversity" → "diversity"
- 注释中的"Beam candidate" → "Candidate"

### 4. 优化流程更新

**修改前**:
```
Seed生成 → Seed修复 → 三阶段优化
```

**修改后（level3）**:
```
Seed生成 → Seed修复 → [Algorithm Analysis] → 三阶段优化
                              ↓
                     生成优化后的seed
```

**修改后（level1/2）**:
```
Seed生成 → Seed修复 → 三阶段优化
（跳过算法分析）
```

## 代码使用示例

### Level 3任务（启用算法分析）
```bash
python main.py KernelBench/level3/1_MLP.py \
    --model network \
    --num_seeds 2 \
    --gpu "Quadro RTX 6000" \
    --server_type openai \
    --model_name deepseek \
    --round 10
```

### Level 1/2任务（跳过算法分析）
```bash
python main.py KernelBench/level1/19_ReLU.py \
    --model single \
    --num_seeds 2 \
    --gpu "Quadro RTX 6000"
```

## 预期效果

### Level 3（算法分析生效）
- **提升幅度**: 30-50%
- **成功率**: 50% → 70%+
- **优化类型**:
  - Fusion决策优化
  - 算法替换（如Flash Attention）
  - 计算图重排
  - Memory layout优化

### Level 1/2（跳过算法分析）
- **保持原有性能**
- **不增加额外开销**

## 技术细节

### 算法分析Prompt构建

使用`build_judger_optimization_prompts`，传入：
- `arch_path`: 原始PyTorch模型路径
- `gpu_name`: 目标GPU
- `ncu_metrics_block`: NCU性能指标
- `cuda_code`: 当前best seed代码
- `stage_name`: "algorithm_analysis"
- `stage_description`: "Algorithmic Structure Analysis"

### 优化Seed生成

基于分析结果，构建新的prompt：
```python
optimization_instruction = f"""
Based on the following algorithmic analysis:
- Bottleneck: {bottleneck}
- Optimization Method: {method}
- Modification Plan: {plan}

Generate an optimized kernel that implements the suggested algorithmic improvements.

Original Kernel:
{best_seed.kernel.code}

{seed_prompt}
"""
```

### 异常处理

- 如果分析失败（JSON解析错误、生成失败等），继续使用原始seeds
- 如果优化后的seed无法运行，不影响原始seeds继续优化
- 所有错误都有详细日志输出

## 文件修改清单

### 修改文件
- `main.py`: 添加算法分析模块，删除搜索功能，清理命名

### 未修改文件
- `prompts/judger_optimization.py`: 复用现有分析prompt构建器
- `prompts/generate_custom_cuda.py`: 保持不变
- `prompts/optimization.py`: 保持不变
- `prompts/error.py`: 保持不变

## 向后兼容性

- ✅ 保持`BeamCandidate`类名
- ✅ 保持`--num_seeds`参数
- ✅ 保持`--elimination_threshold`参数
- ✅ Level 1/2任务行为不变
- ✅ 现有的结果文件格式不变

## 测试建议

### 单元测试
```bash
# 测试level3算法分析
python main.py KernelBench/level3/8_ResNetBasicBlock.py --model network --num_seeds 2

# 测试level1跳过分析
python main.py KernelBench/level1/19_ReLU.py --model single --num_seeds 2
```

### 回归测试
```bash
# 确保level1/2结果与之前一致
python main.py KernelBench/level1/ --first_n 10 --model single
python main.py KernelBench/level2/ --first_n 10 --model fusion
```

### A/B测试
```bash
# 比较有/无算法分析的效果
# 需要临时修改代码禁用算法分析
python main.py KernelBench/level3/ --model network --num_seeds 2
```

## 论文撰写指导

### 方法章节
1. **三阶段优化**: 保持原有描述
2. **算法分析模块（新增）**:
   - 动机：level3需要高层次决策
   - 方法：LLM分析 + 优化seed生成
   - 触发条件：仅level3
3. **多seed策略**: 原有内容

### 实验章节
1. **Ablation Study**:
   - Baseline: 无算法分析
   - Ours: 有算法分析
   - 指标: 成功率、加速比、生成时间
2. **Case Study**:
   - 展示3-5个算法级优化案例
   - 说明LLM如何发现non-trivial优化
3. **分层分析**:
   - Level 1: 算法分析效果有限（符合预期）
   - Level 2: 部分提升
   - Level 3: 显著提升

## 下一步工作

- [ ] 完善算法分析prompt（根据实验结果）
- [ ] 添加更多算法模式（Winograd、Flash Attention等）
- [ ] 实验验证level3提升效果
- [ ] 收集典型案例用于论文
- [ ] 优化NCU profiling开销（可选）
