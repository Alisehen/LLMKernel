# NCU指标完整性分析

## 评估标准

只考虑满足以下**全部**条件的指标：
1. ✅ **Triton直接可操作**：LLM可以通过调整Triton参数来优化
2. ✅ **优化目标指标**：直接反映性能，需要最大化/最小化
3. ✅ **同等或更重要**：不能比现有4个指标差

---

## 当前4指标覆盖分析

### 核心Triton参数映射

| Triton参数 | 对应NCU指标 | 覆盖状态 |
|-----------|------------|---------|
| **BLOCK_M/N/K** | `dram__throughput` + `sm__occupancy` | ✅ 完全覆盖 |
| **num_warps** | `sm__maximum_warps_per_active_cycle_pct` | ✅ 完全覆盖 |
| **num_stages** | `smsp__warp_issue_stalled_memory_dependency` | ✅ 完全覆盖 |
| **eviction_policy** | `lts__t_sector_hit_rate.pct` | ✅ 间接覆盖 |
| **Grid size** | `sm__maximum_warps_per_active_cycle_pct` | ✅ 间接覆盖 |

**结论**: 所有Triton核心参数都已被现有4个指标覆盖。

---

## 候选指标评估

### 1. 寄存器压力 `launch__registers_per_thread`

**Triton可操作性**: ⭐⭐⭐ 中等
- 可通过减小BLOCK_M/N/K来降低寄存器使用

**是否为优化目标**: ❌ 否
- 这是**诊断指标**，不是优化目标
- 寄存器少不一定好（可能性能更差）
- 真正的目标是**提高occupancy**（已有指标）

**诊断价值**: ⭐⭐⭐⭐ 高
- 当occupancy低时，可以诊断是否因为寄存器过多

**是否同等重要**: ❌ 否
- 这是辅助诊断指标，不如优化目标指标重要

**结论**: ❌ **不添加到核心4指标**
- 理由：诊断性而非目标性
- 建议：可作为可选的调试指标（在METRICS_FULL中已有）

---

### 2. Shared Memory使用 `launch__shared_mem_per_block_allocated`

**Triton可操作性**: ⭐⭐ 低
- Triton自动管理shared memory
- 手动控制有限

**是否为优化目标**: ❌ 否
- 同样是诊断指标

**结论**: ❌ **不重要**

---

### 3. 计算吞吐率 `sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active`

**Triton可操作性**: ⭐ 很低
- 算法级别的问题，LLM难以直接优化
- 不是通过调Triton参数能改善的

**是否为优化目标**: ✅ 是
- 理想情况下要最大化计算吞吐

**诊断价值**: ⭐⭐⭐ 中等
- 帮助区分 compute-bound vs memory-bound

**是否同等重要**: ❌ 否
- DRAM throughput已经能帮助判断是否memory-bound
- 如果DRAM低+occupancy高，就是compute-bound
- 不需要单独的计算吞吐指标

**结论**: ❌ **不添加**
- 理由：可操作性太低，现有指标已可推断

---

### 4. L1缓存命中率 `l1tex__t_sector_hit_rate.pct`

**Triton可操作性**: ⭐ 很低
- Triton难以控制L1缓存
- run_ncu.py注释已说明："L1 cache is hard to control in Triton"

**结论**: ❌ **不重要**

---

### 5. Warp执行效率 `smsp__thread_inst_executed_per_inst_executed.ratio`

**说明**: 反映分支分歧（branch divergence）

**Triton可操作性**: ⭐ 很低
- 依赖算法，不是Triton参数能优化的

**结论**: ❌ **不重要**

---

### 6. Tensor Core利用率 `sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active`

**Triton可操作性**: ⭐⭐ 低
- Triton通过使用16倍数的BLOCK自动使用Tensor Core
- LLM难以进一步优化

**是否为优化目标**: ❌ 否
- 这是**结果指标**，不是优化目标
- 真正的目标是性能（DRAM throughput等）

**结论**: ❌ **不添加**
- 理由：结果性而非可操作性

---

### 7. 其他Warp停顿指标

- `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct`
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct`
- `smsp__warp_issue_stalled_barrier_per_warp_active.pct`

**Triton可操作性**: ⭐ 很低
- 编译器级别的优化，LLM无法控制

**结论**: ❌ **不重要**

---

## 性能瓶颈覆盖检查

| 瓶颈类型 | 现有指标覆盖 | 是否需要补充 |
|---------|------------|-------------|
| **内存带宽** | ✅ `dram__throughput` | ❌ 不需要 |
| **内存延迟** | ✅ `memory_stalls` | ❌ 不需要 |
| **缓存效率** | ✅ `lts__t_sector_hit_rate` | ❌ 不需要 |
| **计算资源** | ✅ `sm__occupancy` | ❌ 不需要 |
| **寄存器压力** | ✅ `sm__occupancy` (间接) | ⚠️ 可选诊断 |
| **Shared Memory压力** | ✅ `sm__occupancy` (间接) | ❌ 不需要 |
| **分支分歧** | ❌ 无 | ❌ 不可操作，不需要 |

---

## 最终结论

### ✅ **现有4指标已经是最优配置**

**理由**:
1. ✅ 覆盖所有Triton核心参数（BLOCK, num_warps, num_stages, eviction_policy, grid）
2. ✅ 覆盖所有主要性能瓶颈（带宽、延迟、缓存、计算）
3. ✅ 全都是**可操作的优化目标指标**，不是诊断指标
4. ✅ 数量适中（4个），避免信息过载

### ❌ **没有同等或更重要的指标需要添加**

**可能的扩展方向**（优先级低）:

#### 如果需要更细粒度的诊断（仅用于debugging）:
```python
# 可选的诊断指标集（不是优化目标）
METRICS_DEBUG = ",".join([
    "launch__registers_per_thread",           # 诊断: 占用率低是否因为寄存器
    "launch__shared_mem_per_block_allocated", # 诊断: 占用率低是否因为shared memory
    "launch__occupancy_limit_registers",      # 诊断: 占用率被什么限制
    "launch__occupancy_limit_shared_mem",
])
```

**但这些不应该加入核心METRICS**，原因：
- 不是优化目标，是诊断工具
- Triton可操作性低
- 会让LLM分心，关注错误的东西

---

## 对比其他框架

### PyTorch Profiler
常用指标：
- GPU Utilization - 类似我们的 `sm__occupancy` ✅
- Memory Bandwidth - 类似我们的 `dram__throughput` ✅
- Kernel Time - 我们用speedup间接衡量 ✅

### NVIDIA Nsight Compute推荐的Top指标
高级用户常用：
1. SM Efficiency - 我们有 ✅
2. Memory Throughput - 我们有 ✅
3. L2 Hit Rate - 我们有 ✅
4. Warp Stall Reasons - 我们有最重要的memory stalls ✅

---

## 推荐决策

### 核心指标（当前）: 保持4个 ✅
```python
METRICS = [
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",           # 带宽
    "lts__t_sector_hit_rate.pct",                                   # 缓存
    "sm__maximum_warps_per_active_cycle_pct",                       # 占用率
    "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct", # 延迟
]
```

### 可选调试指标（在METRICS_FULL中已有）
仅在需要深入诊断时使用，不应作为核心指标。

### 不推荐添加
- ❌ 计算吞吐率（可操作性低）
- ❌ 寄存器/Shared Memory使用量（诊断性，非目标性）
- ❌ L1缓存（Triton难控制）
- ❌ 分支效率（算法级问题）

---

## 总结

**现有4个指标已经达到最优平衡**:
- ✅ 覆盖全面（所有Triton参数 + 所有主要瓶颈）
- ✅ 高度可操作（LLM可以通过Triton参数优化）
- ✅ 目标明确（优化目标，不是诊断指标）
- ✅ 简洁高效（4个指标，信息量刚好）

**建议**: 保持当前配置，不添加新的核心指标。

如果将来遇到特定问题需要诊断，可以临时启用METRICS_FULL，但不应改变核心的4指标配置。
