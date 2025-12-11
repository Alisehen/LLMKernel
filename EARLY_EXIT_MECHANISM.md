# Early Exit Mechanism (提前退出机制)

## 概述

CudaForge 现在支持基于 NCU 指标的智能提前退出机制。当某个优化阶段的关键指标已经达到优化目标时，系统会自动跳过该阶段，节省 LLM 调用成本和优化时间。

## 功能特性

### 1. 自动跳过优化阶段

系统会在每个优化阶段开始前：
1. 对当前 kernel 进行 NCU profiling
2. 检查关键性能指标是否已达到阈值
3. 如果指标已优化，跳过该阶段并继续下一阶段

### 2. 各阶段退出标准

#### Stage 1: grid_and_parallel（网格和并行优化）
- **检查指标**: `sm__maximum_warps_per_active_cycle_pct`（SM 占用率）
- **阈值**: ≥ 90%
- **逻辑**: AND（所有条件满足）
- **说明**: SM 占用率超过 90% 表示并行工作分布已经很优化

#### Stage 2: block_tiling（Block 大小优化）
- **检查指标**: `sm__maximum_warps_per_active_cycle_pct`（SM 占用率）
- **阈值**: ≥ 85%
- **逻辑**: AND
- **说明**: 良好的占用率表示 BLOCK_M/N/K 配置已经较合理

#### Stage 3: memory_access（内存访问优化）
- **检查指标**:
  - `lts__t_sector_hit_rate.pct`（L2 缓存命中率）
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed`（DRAM 吞吐量）
- **阈值**: ≥ 90% 或 ≥ 85%
- **逻辑**: OR（任一条件满足）
- **说明**: L2 命中率高或 DRAM 吞吐量高都表示内存访问模式已优化

#### Stage 4: advanced_memory（高级内存优化）
- **检查指标**: `lts__t_sector_hit_rate.pct`（L2 缓存命中率）
- **阈值**: ≥ 95%
- **逻辑**: AND
- **说明**: 非常高的缓存命中率表示缓存策略已接近最优

## 使用示例

### 正常运行（无需额外参数）

```bash
python main.py ref_0.py --device 3
```

系统会自动应用提前退出机制。当遇到跳过的阶段时，会显示：

```
⏩ [Stage 1] SKIPPED: SM occupancy already optimal (>90%) [sm__maximum_warps_per_active_cycle_pct=92.34%]
   Current metrics already meet optimization goals for this stage.
   Proceeding to next stage...
```

### 调整阈值（可选）

如果需要调整阈值，可以在 `main.py` 中修改 `STAGE_EXIT_CRITERIA` 字典：

```python
STAGE_EXIT_CRITERIA = {
    "grid_and_parallel": {
        "metrics": ["sm__maximum_warps_per_active_cycle_pct"],
        "thresholds": [95.0],  # 提高到 95%
        "operator": "and",
        "description": "SM occupancy already optimal (>95%)"
    },
    # ...
}
```

## 验证机制

运行测试脚本验证提前退出机制：

```bash
python test_early_exit.py
```

测试涵盖：
- ✓ 高指标值（应跳过）
- ✓ 低指标值（不应跳过）
- ✓ OR 逻辑（任一条件满足）
- ✓ AND 逻辑（所有条件满足）
- ✓ 边界情况（空数据、缺失指标）

## 性能影响

### 节省的资源
1. **LLM 调用次数**: 每跳过一个阶段节省至少 1-3 次 LLM 调用
2. **优化时间**: 每跳过一个阶段节省约 30-60 秒
3. **Token 成本**: 根据跳过的阶段数，可节省数千到数万 tokens

### 示例场景
假设一个 kernel 在 seed 阶段后已经达到：
- SM 占用率: 92%
- L2 命中率: 93%

系统行为：
- ✓ 跳过 Stage 1 (grid_and_parallel) - SM 占用率 > 90%
- ✓ 跳过 Stage 2 (block_tiling) - SM 占用率 > 85%
- ✓ 跳过 Stage 3 (memory_access) - L2 命中率 > 90%
- ✗ 执行 Stage 4 (advanced_memory) - L2 命中率 < 95%

**节省**: 3 个优化阶段，约 3-9 次 LLM 调用，90-180 秒时间

## 实现细节

### 核心函数

#### `_should_skip_stage(stage_name, metrics_df)`
检查是否应跳过某个优化阶段。

**参数**:
- `stage_name`: 优化阶段名称
- `metrics_df`: NCU profiling 结果的 DataFrame

**返回**:
- `(should_skip, reason)`: 是否跳过和跳过原因

**逻辑**:
1. 查找该阶段的退出标准
2. 从 metrics_df 中提取相关指标值
3. 根据 operator（AND/OR）评估条件
4. 返回判断结果和详细说明

### 集成位置

在 `_run_single_task()` 函数中，每个优化阶段的 NCU profiling 之后：

```python
# Step 1: Profile the current best_kernel to get NCU metrics
csv_path = profile_bench(...)
metrics_df = load_ncu_metrics(...)

# Check if stage should be skipped based on metrics
should_skip, skip_reason = _should_skip_stage(stage_name, metrics_df)
if should_skip:
    print(f"⏩ [Stage {stage_idx + 1}] SKIPPED: {skip_reason}")
    continue

# Step 2: Build optimization prompt (only if not skipped)
opt_prompt = build_optimization_prompt(...)
```

## 扩展建议

### 1. 添加更多指标
可以在 `STAGE_EXIT_CRITERIA` 中添加更多 NCU 指标：
- `smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct`（内存依赖停顿）
- `l1tex__t_sector_hit_rate.pct`（L1 缓存命中率）
- 等等

### 2. 动态阈值
根据不同的 GPU 或问题规模动态调整阈值：

```python
def get_dynamic_threshold(gpu_name, stage_name):
    # 根据 GPU 型号返回不同阈值
    if "A100" in gpu_name:
        return 95.0
    return 90.0
```

### 3. 跳过统计
在 summary 中添加跳过统计信息：
```python
{
    "total_stages": 4,
    "stages_executed": 2,
    "stages_skipped": 2,
    "time_saved_seconds": 120
}
```

## 注意事项

1. **阈值设置**: 当前阈值是根据经验设置的，可能需要根据实际场景调整
2. **指标可用性**: 如果 NCU 无法获取某个指标，该条件会被视为"不满足"
3. **顺序依赖**: 优化阶段有依赖关系，跳过某个阶段可能影响后续阶段的效果

## 总结

提前退出机制通过智能检测性能指标，在保证优化质量的同时显著降低了优化成本。这对于已经部分优化的 kernels 或多次迭代的场景特别有用。
