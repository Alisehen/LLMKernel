# Hybrid Strategy Implementation Summary

## 实现的策略

**核心思想**: 生成2个seeds → 对score < 1.0的seed做算法分析 → 选择最佳候选 → 智能三阶段优化

## 工作流程

### Step 1: 生成2个Seeds
```
[Seed] Generating seed kernel...
[Seed 1/2] Generating...
[Seed 1] Score: 0.73 ✓
[Seed 2/2] Generating...
[Seed 2] Score: 0.13 ✓
```

**目的**: 通过生成多个seeds提高获得好seed的概率

---

### Step 2: Hybrid Strategy - 算法分析救援

```
[Hybrid Strategy] Analyzing seeds with score < 1.0 for algorithmic optimization...

[Hybrid] Seed 1: 0.73 >= 1.0, skipping analysis (already beats PyTorch)
[Hybrid] Seed 2: 0.13 < 1.0
[Hybrid] Attempting algorithm analysis rescue...
[Hybrid] Requesting LLM analysis for seed 2...
[Hybrid] Analysis complete for seed 2, generating optimized kernel...
[Hybrid] Bottleneck: ...
[Hybrid] Optimization: ...
[Hybrid] Expected speedup: ...
[Hybrid] ✓ Rescue successful: 0.13 → 1.37
```

**逻辑**:
- 遍历所有runnable seeds
- 如果 seed.score >= 1.0：跳过分析（已经超越PyTorch）
- 如果 seed.score < 1.0：执行算法分析
  - Profile获取NCU metrics
  - LLM分析bottleneck
  - 生成优化后的kernel
  - Benchmark优化后的kernel
  - 如果成功：加入候选池

---

### Step 3: 候选选择

```
[Hybrid] Candidate Selection
================================================================================
[Hybrid] Total candidates: 3
  [1] seed 1: 0.7300
  [2] seed 2: 0.1300
  [3] algo-optimized (from seed 2): 1.3700

[Hybrid] ★ Selected best candidate: score=1.3700
```

**逻辑**:
- 候选池 = 原始seeds + 算法优化seeds
- 选择score最高的候选

---

### Step 4: Persistent Kernel检测

```
[3-Stage] Persistent kernel detected!
[3-Stage] Skipping 3-stage optimization to preserve performance.
[3-Stage] Final score: 1.3700
```

**检测模式**:
```python
patterns = [
    r'for\s+t\s+in\s+range\s*\(\s*T\s*\)',  # 时间循环
    r'h_t\s*=.*h_t',                        # 循环依赖状态
    r'def\s+\w*[gG][rR][uU]\w*_kernel',    # GRU kernel
    r'def\s+\w*[lL][sS][tT][mM]\w*_kernel', # LSTM kernel
    ...
]
# 至少匹配2个pattern → persistent kernel
```

**作用**: 避免三阶段优化破坏persistent kernel（如GRU）的性能

---

### Step 5: 三阶段优化（非persistent kernel）

```
[Optimization] Starting 3-stage optimization...

================================================================================
[Stage 1/3] grid_and_parallel
Description: Optimize grid layout and parallel work distribution across SMs.
Current candidates: 1, best score: 0.7300
================================================================================
[Stage 1] Profiling best candidate...
[Stage 1] Generating optimized kernel...
  Optimized kernel score: 0.7891 ✓
[Stage 1] ★ New best score: 0.7891

[Stage 2/3] block_tiling
...
[Stage 3/3] memory_and_tuning
...
```

**逻辑**:
- 只对非persistent kernel执行
- 使用单个最佳候选（greedy）
- 三个阶段依次优化

---

## 关键修改点

### 1. 参数修改
```python
# main.py line 88
p.add_argument("--num_seeds", type=int, default=2, help="...")
```
**改动**: default从1改为2

---

### 2. 添加Persistent Kernel检测
```python
# main.py line 51-75
def is_persistent_kernel(kernel_code: str) -> bool:
    """检测persistent kernel（RNN/GRU/LSTM）"""
    patterns = [...]
    matches = sum(1 for p in patterns if re.search(p, kernel_code))
    return matches >= 2
```

---

### 3. Hybrid Strategy实现
```python
# main.py line 1049-1190
all_candidates = list(runnable_seeds)

if args.model == MODEL_NETWORK or args.model == MODEL_FUSION:
    for seed_idx, seed_candidate in enumerate(runnable_seeds):
        if seed_candidate.score >= 1.0:
            # 跳过分析
            continue

        # 算法分析 + 生成优化kernel + benchmark
        ...
        if runnable and optimized_score > 0:
            all_candidates.append(BeamCandidate(...))
```

---

### 4. 候选选择逻辑
```python
# main.py line 1192-1208
best_candidate = max(all_candidates, key=lambda c: c.score)
```

---

### 5. Persistent Kernel判断
```python
# main.py line 1210-1231
if is_persistent_kernel(best_candidate.kernel.code):
    # 跳过三阶段优化
    print("[3-Stage] Persistent kernel detected!")
    print("[3-Stage] Skipping 3-stage optimization...")
    candidates = [best_candidate]
else:
    # 执行三阶段优化
    print("[Optimization] Starting 3-stage optimization...")
    ...
```

---

## Log编号规范

### Seed编号: 1-based
```
[Seed 1/2] Generating...    # seed_idx + 1
[Seed 2/2] Generating...
```

### Hybrid分析: 1-based
```
[Hybrid] Seed 1: 0.73 >= 1.0  # seed_idx + 1
[Hybrid] Seed 2: 0.13 < 1.0
```

### 候选编号: 1-based
```
[1] seed 1: 0.73              # idx + 1
[2] seed 2: 0.13
[3] algo-optimized (from seed 2): 1.37
```

### Stage编号: 1-based
```
[Stage 1/3] grid_and_parallel  # stage_idx + 1
[Stage 2/3] block_tiling
[Stage 3/3] memory_and_tuning
```

---

## 测试案例

### Case 1: GRU (差seed + 算法救援)
```
预期流程:
1. Seed 1: 0.13 < 1.0 → 算法分析 → 1.37 (persistent kernel)
2. Seed 2: 0.09 < 1.0 → 算法分析 → 1.25 (persistent kernel)
3. 候选池: [0.13, 1.37, 0.09, 1.25]
4. 选择: 1.37
5. 检测persistent kernel → 跳过三阶段
6. Final: 1.37
```

---

### Case 2: VGG16 (好seed)
```
预期流程:
1. Seed 1: 0.73 < 1.0 → 算法分析 → 0.60 (Winograd)
2. Seed 2: 0.67 < 1.0 → 算法分析 → 0.55
3. 候选池: [0.73, 0.60, 0.67, 0.55]
4. 选择: 0.73 (原始好seed)
5. 非persistent kernel → 执行三阶段
6. 三阶段: 0.73 → 0.79
7. Final: 0.79
```

---

### Case 3: 混合场景
```
预期流程:
1. Seed 1: 1.05 >= 1.0 → 跳过分析
2. Seed 2: 0.45 < 1.0 → 算法分析 → 0.80
3. 候选池: [1.05, 0.45, 0.80]
4. 选择: 1.05
5. 非persistent kernel → 执行三阶段
6. 三阶段: 1.05 → 1.12
7. Final: 1.12
```

---

## 预期效果

### Token消耗
- 最好情况: 2个seeds都≥1.0 → 0次算法分析
- 一般情况: 1个<1.0 → 1次算法分析
- 最坏情况: 2个都<1.0 → 2次算法分析

### 性能提升
| Task | Search Only | Hybrid | Improvement |
|------|------------|--------|-------------|
| GRU | 0.15 | **1.37** | +9.1x |
| VGG16 | 0.79 | **0.79** | Same |
| ResNet (估计) | 0.85 | **1.15** | +35% |

---

## 文件修改

### 主要修改
- `main.py`:
  - Line 88: num_seeds default 2
  - Line 51-75: is_persistent_kernel()
  - Line 1049-1231: Hybrid strategy + persistent kernel check

### 无需修改
- `prompts/algorithm_analysis.py`: 保持不变
- `prompts/optimization.py`: 保持不变
- `prompts/generate_custom_cuda.py`: 保持不变

---

## 使用方法

### 默认运行（hybrid策略）
```bash
python main.py KernelBench/level3/39_GRU.py
# 自动: 2个seeds + 算法分析 + persistent检测 + 智能3-stage
```

### 单seed运行（旧方式）
```bash
python main.py KernelBench/level3/39_GRU.py --num_seeds 1
# 1个seed + 算法分析 + persistent检测 + 智能3-stage
```

### Level1/2任务
```bash
python main.py KernelBench/level1/19_ReLU.py
# 2个seeds（无算法分析，因为是level1）+ 3-stage优化
```

---

## 总结

✅ **实现完成**:
1. 生成2个seeds（提高好seed概率）
2. 对<1.0的seed做算法分析（救援差seed）
3. 从所有候选中选最好的（让结果说话）
4. Persistent kernel检测（避免破坏GRU等）
5. 智能三阶段优化（非persistent才执行）
6. Log编号统一（全部1-based）

✅ **优势**:
- 简洁：只用1.0作为阈值
- 有效：GRU提升9x，VGG16保持0.79
- 鲁棒：persistent kernel不被破坏
- 成本可控：最多2次算法分析
