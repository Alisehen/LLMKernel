# Optimization Prompt修复：添加PyTorch Reference

## 问题回顾

在优化`optimization_from_analysis` prompt以减少token消耗时，我们移除了完整的seed_prompt，但也意外地移除了**PyTorch reference code**。

这导致40_GRUHidden任务失败（0.20x vs 39_GRU的1.37x），因为LLM无法看到任务只需要返回`h_n`而不是完整的`output`。

---

## 修复方案

### 修改1: `prompts/optimization_from_analysis.py`

#### 添加PyTorch Reference段落（template line 20-32）

```python
# PyTorch Reference (Target Behavior)

```python
$pytorch_reference
```

**CRITICAL**: Study the PyTorch code carefully to understand:
- What does `forward()` return? (full output sequence vs final hidden state only)
- What is the computational pattern?
- What are the input/output shapes?

Your optimized kernel MUST match this exact behavior.
```

**重点强调**：
- 用 `**CRITICAL**` 标记，确保LLM注意到
- 明确指出需要关注的关键信息：`forward()`的返回值
- 特别提到"full output vs final hidden state"这个GRU任务的关键差异

#### 更新函数签名（line 84-92）

```python
def build_optimization_from_analysis_prompt(
    *,
    bottleneck: str,
    optimization_method: str,
    modification_plan: str,
    expected_speedup: str,
    current_kernel: str,
    pytorch_reference: str,  # ✅ 新增参数
) -> str:
```

#### 更新template.substitute（line 106-113）

```python
prompt = optimization_from_analysis_tmpl.substitute(
    bottleneck=bottleneck,
    optimization_method=optimization_method,
    modification_plan=modification_plan,
    expected_speedup=expected_speedup,
    current_kernel=current_kernel.strip(),
    pytorch_reference=pytorch_reference.strip(),  # ✅ 新增
)
```

---

### 修改2: `main.py`

#### 读取并传递PyTorch代码（line 1167-1177）

```python
# Continue with optimization if worthwhile
print(f"[Hybrid] Analysis complete for seed {seed_idx + 1}, generating optimized kernel...")
print(f"[Hybrid] Bottleneck: {analysis_json.get('bottleneck', 'N/A')[:80]}...")
print(f"[Hybrid] Optimization: {analysis_json.get('optimisation method', 'N/A')[:80]}...")
print(f"[Hybrid] Expected speedup: {analysis_json.get('expected_speedup', 'N/A')}")

# ✅ Read PyTorch reference code
pytorch_code = task_path.read_text(encoding="utf-8")

# Build prompt for generating optimized seed based on analysis
optimization_instruction = build_optimization_from_analysis_prompt(
    bottleneck=analysis_json.get('bottleneck', 'N/A'),
    optimization_method=analysis_json.get('optimisation method', 'N/A'),
    modification_plan=analysis_json.get('modification plan', 'N/A'),
    expected_speedup=analysis_json.get('expected_speedup', 'N/A'),
    current_kernel=seed_candidate.kernel.code,
    pytorch_reference=pytorch_code,  # ✅ 新增参数
)
```

---

## Token分析

### 修复前后对比

| 版本 | PyTorch代码 | 完整seed_prompt | 总Token（估算） |
|------|------------|----------------|----------------|
| **原始版本** | ✅ 有（在seed_prompt内） | ✅ 有 | ~10,000 |
| **优化版本（有bug）** | ❌ 无 | ❌ 无 | ~1,500 |
| **修复版本** | ✅ 有（单独添加） | ❌ 无 | ~2,500 |

### 详细breakdown

**修复版本的token组成**：
- Analysis results (bottleneck/optimization/plan): ~300 tokens
- **PyTorch reference code**: ~500-1000 tokens (典型任务)
- Current kernel code: ~800-1200 tokens
- Triton syntax rules (精简版): ~200 tokens
- Output format requirements: ~100 tokens
- **总计**: ~2,000-2,500 tokens

**节省效果**：
- vs 原始版本：节省 ~7,500-8,000 tokens (**75%节省**)
- vs 有bug版本：增加 ~1,000 tokens (**必要的代价**)

---

## 预期效果

### 对40_GRUHidden的影响

**修复前**：
```
LLM看到的信息：
- Bottleneck: "多个小kernel launches"
- Optimization: "Fuse into persistent kernel"
- Current kernel: (只有Triton代码)
❌ 不知道任务只需要h_n

生成的kernel：
- 计算所有时间步的输出（浪费）
- 性能：0.20x
```

**修复后**：
```
LLM看到的信息：
- PyTorch Reference: return h_n  ✅ 看到了！
- Bottleneck: "多个小kernel launches"
- Optimization: "Fuse into persistent kernel"
- Current kernel: (Triton代码)
✅ 知道只需要最后的h_n

预期生成的kernel：
- 只计算最后的h_n（高效）
- 预期性能：1.0x+ (接近或超越PyTorch)
```

---

## 其他任务的影响

### 不受影响的任务

**Level1/2任务**：
- 不使用算法分析
- 仍然使用seed_prompt
- 无影响

**Level3且seed好的任务**：
- 如果seed >=1.0，跳过算法分析
- 无影响

### 受益的任务

**所有需要算法分析的Level3任务**：
- 现在LLM能看到完整的任务定义
- 能够根据输出要求优化kernel
- 特别是对于：
  - RNN/GRU/LSTM variants (不同的output需求)
  - Attention variants (full attention matrix vs scores only)
  - Conv variants (different output shapes)

---

## 对比：修复前后的Prompt

### 修复前（有bug）

```
You are optimizing a Triton kernel based on algorithmic analysis.

# Analysis Results
Bottleneck: ...
Optimization Strategy: ...
...

# Current Kernel (needs optimization)
```python
[Triton code]
```

# Your Task
Implement the optimization strategy above.
...
```

**问题**：LLM只看到Triton代码，不知道任务的实际需求。

---

### 修复后

```
You are optimizing a Triton kernel based on algorithmic analysis.

# PyTorch Reference (Target Behavior)
```python
class Model(nn.Module):
    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        return h_n  # ✅ LLM看到这里！
```

**CRITICAL**: Study the PyTorch code carefully to understand:
- What does `forward()` return? (full output sequence vs final hidden state only)
...

# Analysis Results
Bottleneck: ...
Optimization Strategy: ...
...

# Current Kernel (needs optimization)
```python
[Triton code]
```

# Your Task
Implement the optimization strategy above.
...
```

**优势**：LLM能看到完整的任务定义，理解实际需求。

---

## 验证方法

### 测试1: 40_GRUHidden重新运行

```bash
python main.py KernelBench/level3/40_GRUHidden.py

# 预期结果：
# - Seed生成：~0.05-0.15
# - 算法分析：识别"只需要h_n"
# - 优化生成：只计算h_n的persistent kernel
# - 最终score：1.0x+ (vs 修复前的0.20x)
```

### 测试2: 检查生成的prompt

```bash
cat run/*/40_GRUHidden/evaluation/llm_io/optimization_from_analysis_prompt_seed*.txt

# 检查：
# - ✅ 包含PyTorch代码
# - ✅ 包含"return h_n"这一行
# - ✅ 有CRITICAL提示
```

### 测试3: 检查生成的kernel

```bash
cat run/*/40_GRUHidden/code/kernel_*.py

# 检查：
# - ✅ 是否只计算最后的h_n
# - ✅ 是否避免输出所有时间步
# - ✅ 性能是否接近1.0x
```

---

## 总结

✅ **修复完成**：
1. 在optimization prompt中添加PyTorch reference code
2. 强调需要关注的关键信息（return值、shapes等）
3. 保持token效率（仍然比原始版本节省75%）

✅ **预期效果**：
- 40_GRUHidden：0.20x → 1.0x+ (**5x提升**)
- 其他类似任务：更准确的优化策略
- Token成本可控：只增加~1000 tokens

✅ **关键教训**：
- Token优化很重要，但**不能丢失关键上下文**
- PyTorch reference虽然占token，但对理解任务至关重要
- 需要在效率和准确性之间找到平衡
- **完整性 > 简洁性**，当涉及到任务理解时

✅ **下一步**：
- 重新运行40_GRUHidden验证修复
- 在其他Level3任务上测试
- 监控token消耗和性能提升
