# 最终优化策略（包含轻微提升优化）

## 策略总览

**核心原则**: 除了显著提升（>15%）直接采用外，所有其他情况都尝试 1-2 次优化。

---

## 完整决策表

| 性能变化 | 判断 | 迭代次数 | Token 消耗 | 说明 |
|---------|------|---------|-----------|------|
| **提升 >15%** | ✅ 显著提升 | **0** | 0 | 直接采用 |
| **提升 0-15%** | ✅ 轻微提升 | **1** | ~5,500 | 尝试进一步优化 ⭐ |
| **下降 0-10%** | ⚠️ 轻微下降 | **1** | ~5,500 | 快速修复 |
| **下降 >10%** | ✗ 显著下降 | **2** | ~11,000 | 深度修复 + 早停 |

---

## 代码逻辑

```python
improvement_ratio = step_score / global_best_score

if improvement_ratio > 1.15:
    # 提升 >15% → 直接采用
    should_optimize = False
    profile_iters = 0

elif improvement_ratio >= 1.0:
    # 提升 0-15% → 1 次优化（尝试进一步提升）⭐
    should_optimize = True
    profile_iters = 1

elif improvement_ratio >= 0.90:
    # 下降 0-10% → 1 次优化（快速修复）
    should_optimize = True
    profile_iters = 1

else:
    # 下降 >10% → 2 次优化（深度修复）
    should_optimize = True
    profile_iters = 2
```

---

## 设计理由

### 为什么轻微提升也要优化？

**原因 1: 探索更优解**
```
初始代码: +8%
如果不优化 → 接受 +8%
如果优化 1 次:
  ├─ 成功: +15% (额外提升 7%)  ✓
  └─ 失败: +8% (保持原样)      -
风险低，收益可能高
```

**原因 2: 每个阶段都是优化机会**
```
Stage 3 (Block Tiling):
  初始: +5% (1.05x)
  优化后: +18% (1.18x)  ← 可能找到更优的 BLOCK size

如果不优化，可能错过该阶段的最优参数
```

**原因 3: Token 成本可控**
```
1 次优化 = ~5,500 tokens
vs 直接采用 = 0 tokens

差异: 5,500 tokens
成本: 可接受（相比 2-3 次的 11,000-16,500）
```

---

## Token 消耗分析

### 典型 4 阶段场景

**场景 1: 全部顺利（最好情况）**
```
Stage 1: +18% → 0 token (直接采用)
Stage 2: +20% → 0 token (直接采用)
Stage 3: +17% → 0 token (直接采用)
Stage 4: +16% → 0 token (直接采用)
总计: 0 tokens
```

**场景 2: 有轻微提升（典型情况）**
```
Stage 1: +12% → 5,500 token (1 次优化)
Stage 2: +8%  → 5,500 token (1 次优化)
Stage 3: +6%  → 5,500 token (1 次优化)
Stage 4: +10% → 5,500 token (1 次优化)
总计: 22,000 tokens
```

**场景 3: 部分需要修复（混合情况）**
```
Stage 1: +10% → 5,500 token (1 次优化)
Stage 2: -7%  → 5,500 token (1 次优化)
Stage 3: -18% → 11,000 token (2 次优化)
Stage 4: +5%  → 5,500 token (1 次优化)
总计: 27,500 tokens
```

**场景 4: 全部需要修复（最坏情况）**
```
Stage 1: -12% → 11,000 token (2 次优化)
Stage 2: -8%  → 5,500 token (1 次优化)
Stage 3: -15% → 11,000 token (2 次优化)
Stage 4: -6%  → 5,500 token (1 次优化)
总计: 33,000 tokens
```

### 对比旧策略

| 场景 | 旧策略 (固定3次) | 新策略 (自适应) | 节省 |
|------|----------------|----------------|------|
| 全部顺利 | 66,000 | **0** | **100%** |
| 轻微提升 | 66,000 | **22,000** | **67%** |
| 混合情况 | 66,000 | **27,500** | **58%** |
| 全部修复 | 66,000 | **33,000** | **50%** |

**平均节省**: 约 **60-70%**

---

## 实际案例

### 案例 1: 轻微提升后进一步优化成功

```
Stage 3: Block Tiling
  Baseline: 1.0x

  Initial attempt:
    Score: 1.08x (+8%)
    → ✓ Minor improvement
    → Attempting 1 optimization...

  Profile-based optimization:
    Current NCU:
      dram__throughput: 68.5%
      l1tex__t_sector_hit_rate: 72.3%
      occupancy: 65.2%

    LLM analysis:
      "DRAM throughput moderate, cache hit rate good.
       Suggest: Increase BLOCK_M/N to 128/256 for more data reuse"

    Optimized code:
      BLOCK_M: 64 → 128
      BLOCK_N: 64 → 256

    Result: 1.18x (+18% vs baseline, +10% vs initial)
    → ✓ Accepted optimized version

  Token used: ~5,500
  Benefit: +10% additional speedup
```

---

### 案例 2: 轻微提升后优化失败（无损）

```
Stage 2: Memory Access
  Baseline: 1.0x

  Initial attempt:
    Score: 1.12x (+12%)
    → ✓ Minor improvement
    → Attempting 1 optimization...

  Profile-based optimization:
    Current NCU: (good metrics)

    LLM analysis:
      "Already well optimized for memory access.
       Suggest: Try tl.trans() for different layout"

    Optimized code:
      Added tl.trans()

    Result: 1.10x (-2% vs initial)
    → ✗ Worse than initial
    → Rollback to initial 1.12x

  Token used: ~5,500
  Benefit: 0 (但没有损失，保持 1.12x)
```

---

### 案例 3: 显著下降后成功修复

```
Stage 3: Block Tiling
  Baseline: 1.20x

  Initial attempt:
    Score: 1.02x (-15% vs baseline)
    → ✗ Significant degradation
    → Attempting 2 optimizations...

  Iteration 1:
    NCU comparison:
      Baseline: occupancy 72%, cache hit 78%
      Current:  occupancy 28%, cache hit 35%
      → Problem: BLOCK size too large

    Optimized: Reduce BLOCK_M/N to 128
    Result: 1.15x (+13% vs failed, -4% vs baseline)
    → Better, but still below baseline
    → Continue...

  Iteration 2:
    Optimized: BLOCK_M=128, BLOCK_N=128, BLOCK_K=64
    Result: 1.38x (+20% vs iteration 1, +15% vs baseline)
    → ✓ Success! Above baseline

  Token used: ~11,000
  Benefit: Recovered from -15% to +15%
```

---

## 优势分析

### ✅ 优势

1. **更充分探索每个阶段的优化空间**
   - 即使初始有提升，也尝试找到更优解
   - 每个阶段都是独立的优化机会

2. **风险可控**
   - 只优化 1 次，token 成本较低
   - 如果失败，保留初始版本（无损）

3. **收益可能很高**
   - +5% → +15% (额外 10% 提升)
   - 对于关键阶段（如 Block Tiling），收益明显

4. **仍然节省大量 token**
   - vs 固定 3 次: 节省 60-70%
   - vs 完全不优化: 增加适度成本，但性能更优

### ⚠️ 成本增加

相比"轻微提升直接采用"的策略:
```
旧方案（提升直接采用）:
  Stage 1: +10% → 0 token
  Stage 2: +8%  → 0 token
  Total: 0 tokens

新方案（提升也优化）:
  Stage 1: +10% → 5,500 token (1 次)
  Stage 2: +8%  → 5,500 token (1 次)
  Total: 11,000 tokens

增加: 11,000 tokens
但可能收益: 额外 5-10% 性能提升
```

**判断**: 11,000 tokens 换取可能的 5-10% 性能提升 → **值得**

---

## 决策树（最终版）

```
性能测试
  ↓
计算 improvement_ratio
  ↓
┌────────────────────────────────────────┐
│ ratio > 1.15  (提升 >15%)               │
│ → 直接采用 (0 token)                    │
│ 理由: 已经很好，进一步优化收益有限        │
└────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────┐
│ 1.0 ≤ ratio ≤ 1.15  (提升 0-15%) ⭐     │
│ → 1 次优化 (~5,500 token)               │
│ 理由: 探索该阶段的更优参数              │
│ 风险: 低（失败则回退）                  │
│ 收益: 可能额外 5-10% 提升                │
└────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────┐
│ 0.90 ≤ ratio < 1.0  (下降 0-10%)        │
│ → 1 次优化 (~5,500 token)               │
│ 理由: 快速修复轻微问题                  │
└────────────────────────────────────────┘
  ↓
┌────────────────────────────────────────┐
│ ratio < 0.90  (下降 >10%)               │
│ → 2 次优化 (~11,000 token)              │
│ 理由: 深度修复严重问题                  │
│ 早停: 连续 2 次失败即停                 │
└────────────────────────────────────────┘
```

---

## 配置建议

### 默认配置（推荐）

```bash
python main.py task.py \
  --round 4 \
  --profile_iters_per_step 2  # 上限为 2
```

### 如果 Token 预算紧张

可以临时修改代码，让轻微提升直接采用:
```python
elif improvement_ratio >= 1.0:
    # 提升 0-15% → 直接采用（节省 token）
    should_optimize = False
    profile_iters = 0
```

**效果**: 可节省额外 20-30% token，但可能错过 5-10% 性能提升

---

## 性能 vs 成本权衡

### 方案对比

| 方案 | Token 成本 (4阶段) | 预期性能 | 推荐场景 |
|------|------------------|---------|---------|
| **固定 3 次优化** | 66,000 | 最高 | 学术研究 |
| **新策略（轻微提升也优化）** | 22,000-33,000 | **高** | **生产环境** ⭐ |
| **保守策略（提升直接采用）** | 11,000-22,000 | 中高 | Token 极度受限 |
| **不优化** | 0 | 低 | 不推荐 |

### 推荐使用

✅ **新策略（轻微提升也优化）** - 平衡点
- Token 成本: 中等（22k-33k）
- 性能: 高（充分探索每个阶段）
- 风险: 低（1-2 次迭代 + 早停）
- 适合: **大多数生产场景**

---

## 预期效果

### 4 阶段优化（典型）

```
Stage 0: Seed
  1.0x

Stage 1: Grid (+10%)
  Initial: 1.10x
  Optimize 1 iter: 1.14x
  → Accept: 1.14x

Stage 2: Memory (+8%)
  Initial: 1.23x (vs 1.14x)
  Optimize 1 iter: 1.28x
  → Accept: 1.28x

Stage 3: Block (-12%)
  Initial: 1.13x (vs 1.28x)
  Optimize 2 iter: 1.42x
  → Accept: 1.42x

Stage 4: Fine-tune (+6%)
  Initial: 1.50x (vs 1.42x)
  Optimize 1 iter: 1.56x
  → Accept: 1.56x

Final: 1.56x speedup
Token used: 27,500
vs 不优化提升: ~1.3x
vs 固定3次: 节省 58% token
```

---

## 总结

### 最终策略

- **提升 >15%**: 直接采用 (0 次)
- **提升 0-15%**: 优化 1 次 ⭐
- **下降 0-10%**: 优化 1 次
- **下降 >10%**: 优化 2 次

### 关键特点

✅ **充分探索**: 每个阶段都尝试找最优解
✅ **成本可控**: 平均 22k-33k tokens（节省 60-70%）
✅ **风险低**: 1-2 次迭代 + 早停机制
✅ **性能高**: 不错过任何优化机会

这是**性能与成本的最佳平衡**！🎯
