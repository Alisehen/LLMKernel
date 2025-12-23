# 算法分析早停机制

## 功能说明

在算法分析阶段添加了智能判断机制：**如果LLM判断优化空间很小或不值得优化，则不再调用大模型生成新的kernel**。

这可以显著减少无效的LLM调用，节省token成本。

---

## 实现细节

### 1. Prompt修改 (`prompts/algorithm_analysis.py`)

在JSON输出中添加了两个新字段：

```json
{
  "worth_optimizing": "yes/no",
  "reason": "<Why worth or not worth optimizing, 1 sentence>",
  "bottleneck": "...",
  "optimisation method": "...",
  "modification plan": "...",
  "expected_speedup": "..."
}
```

### 2. 判断标准

**不值得优化 (worth_optimizing: "no")** 的情况：
- 代码已经接近最优（预期提升 < 10%）
- 瓶颈无法解决（硬件限制、算法已经最优）
- 优化会增加复杂度但收益很小

**值得优化 (worth_optimizing: "yes")** 的情况：
- 存在明显的算法低效（多个kernel、次优算法）
- 预期提升 >= 20%
- 有具体的优化路径

### 3. 逻辑流程

```
For each seed with score < 1.0:
  ├─ Profile kernel (NCU)
  ├─ Call LLM for algorithm analysis
  ├─ Parse JSON result
  ├─ Check "worth_optimizing"
  │
  ├─ If "no":
  │   ├─ Print reason
  │   └─ Skip kernel generation (continue to next seed)
  │
  └─ If "yes":
      ├─ Print bottleneck/optimization/speedup
      ├─ Generate optimized kernel
      └─ Benchmark and add to candidates
```

---

## 代码修改

### prompts/algorithm_analysis.py (line 83-109)

添加了判断标准说明和新的JSON字段：

```python
## Should We Optimize?

Before proposing optimization, determine if it's worthwhile:
- **Not worth optimizing** if:
  - Code is already near-optimal (expected speedup < 10%)
  - Bottleneck cannot be addressed (hardware limited, already optimal algorithm)
  - Optimization would add significant complexity with minimal gain

- **Worth optimizing** if:
  - Clear algorithmic inefficiency exists (multiple kernels, suboptimal algorithm)
  - Expected speedup >= 20%
  - Concrete optimization path available

## Output (JSON)

```json
{
  "worth_optimizing": "yes/no",
  "reason": "<Why worth or not worth optimizing, 1 sentence>",
  ...
}
```
```

### main.py (line 1144-1158)

添加了早停检查：

```python
# Extract analysis and check if worth optimizing
try:
    analysis_json = extract_json(analysis_result)
    if analysis_json:
        worth_optimizing = analysis_json.get('worth_optimizing', 'yes').lower()
        reason = analysis_json.get('reason', 'N/A')

        print(f"[Hybrid] Worth optimizing: {worth_optimizing}")
        print(f"[Hybrid] Reason: {reason}")

        # Check if optimization is worthwhile
        if worth_optimizing == 'no':
            print(f"[Hybrid] ⊘ Skipping optimization for seed {seed_idx + 1} (not worth optimizing)")
            continue  # Skip to next seed

        # Continue with optimization if worthwhile
        ...
```

---

## 使用场景示例

### Case 1: 已接近最优（不值得优化）

```
[Hybrid] Seed 1: score=0.92 < 1.0
[Hybrid] Attempting algorithm analysis rescue...
[Hybrid] Requesting LLM analysis for seed 1...
[Hybrid] Worth optimizing: no
[Hybrid] Reason: Kernel already uses optimal Winograd algorithm, expected gain < 5%
[Hybrid] ⊘ Skipping optimization for seed 1 (not worth optimizing)
```

**Token节省**: 1次LLM调用（分析） vs 2次（分析+生成）→ 节省50%

---

### Case 2: 有优化空间（值得优化）

```
[Hybrid] Seed 1: score=0.13 < 1.0
[Hybrid] Attempting algorithm analysis rescue...
[Hybrid] Requesting LLM analysis for seed 1...
[Hybrid] Worth optimizing: yes
[Hybrid] Reason: Multiple separate kernels can be fused, expected 10x speedup
[Hybrid] Analysis complete for seed 1, generating optimized kernel...
[Hybrid] Bottleneck: Launching 64 separate kernels for time-step loop
[Hybrid] Optimization: Fuse into persistent kernel
[Hybrid] Expected speedup: 800-1000%
[Hybrid] ✓ Rescue successful: 0.13 → 1.37
```

**继续生成**: 值得调用LLM生成优化kernel

---

## 预期收益

### Token成本优化

| 场景 | 原流程 | 新流程 | 节省 |
|------|--------|--------|------|
| 2个seeds，都值得优化 | 2×分析 + 2×生成 | 2×分析 + 2×生成 | 0% |
| 2个seeds，1个不值得 | 2×分析 + 2×生成 | 2×分析 + 1×生成 | 25% |
| 2个seeds，都不值得 | 2×分析 + 2×生成 | 2×分析 + 0×生成 | 50% |

### 典型任务

- **VGG16** (好seed 0.79): LLM可能判断"Winograd复杂度高，提升<5%" → 不值得 → 节省1次生成
- **GRU** (差seed 0.13): LLM判断"多kernel可融合，提升10x" → 值得 → 继续生成
- **MatMul** (seed 0.95): LLM判断"已接近roofline，提升<10%" → 不值得 → 节省1次生成

**估算**: 约30-40%的cases可以跳过生成，节省约20-30%的总token成本

---

## 总结

✅ **实现完成**:
1. Prompt添加判断标准和`worth_optimizing`字段
2. main.py添加早停检查逻辑
3. 保持向后兼容（默认值为"yes"）

✅ **优势**:
- 智能：由LLM自己判断是否值得优化
- 节省成本：避免无效的kernel生成调用
- 不影响性能：不值得优化的cases本来提升也很小
- 保持灵活：LLM可以根据具体情况判断

✅ **适用范围**:
- 对所有Level2/3任务生效
- Level1任务不使用算法分析，不受影响
