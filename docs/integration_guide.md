# 算子分类系统集成指南

## 概述

将`config/operator_categories_v2.py`的分类系统集成到现有的优化流程中，实现：
1. **自动分类**：根据算子类型选择优化策略
2. **动态stage**：不同类别使用不同的优化阶段
3. **智能early exit**：基于分类的早退机制
4. **定制化prompt**：针对类别的优化建议

---

## 集成步骤

### Step 1: 修改 `main.py` - 算子分类

在main函数开始处，对算子进行分类：

```python
# main.py (在 run_one_task 函数开头添加)

from config.operator_categories_v2 import (
    classify_operator,
    OPERATOR_CATEGORIES,
    get_stage_config,
    build_stage_prompt_section,
    get_key_ncu_metrics,
    check_early_exit,
)

def run_one_task(task_path: Path, args, ...):
    # ... 现有代码 ...

    # 【新增】分类算子
    task_name = task_path.stem  # 例如: "56_Matmul_Sigmoid_Sum"
    level = "level2" if "level2" in str(task_path) else "level1"

    category = classify_operator(task_name, level)
    category_config = OPERATOR_CATEGORIES[category]

    print(f"\n{'='*80}")
    print(f"Operator Category: {category}")
    print(f"Description: {category_config['description']}")
    print(f"Total Stages: {len(category_config['stages'])}")
    print(f"{'='*80}\n")

    # ... 继续seed阶段 ...
```

---

### Step 2: 使用分类特定的 Stages

替换硬编码的 `OPTIMIZATION_STAGES`：

```python
# main.py (替换原来的 OPTIMIZATION_STAGES 循环)

def run_one_task(task_path: Path, args, ...):
    # ... seed阶段代码 ...

    # 【修改】使用分类特定的stages
    optimization_stages = category_config["stages"]  # 从分类配置获取

    for stage_idx, stage_config in enumerate(optimization_stages):
        stage_name = stage_config["name"]
        stage_description = stage_config["description"]

        print(f"\n{'='*80}")
        print(f"[Stage {stage_idx + 1}/{len(optimization_stages)}] {stage_name}")
        print(f"Category: {category}")
        print(f"Focus: {stage_config['focus']}")
        print(f"{'='*80}")

        # ... NCU profiling ...

        # 【新增】Early exit检查
        should_exit, exit_reason = check_early_exit(
            category=category,
            stage_id=stage_idx,
            performance_score=best_score,
            op_metadata={
                "op_type": task_name,
                "kernel_size": extract_kernel_size(task_path),  # 需要实现
                "score": best_score,
            }
        )

        if should_exit:
            print(f"\n⛔ [Early Exit] {exit_reason}")
            print(f"   Skipping remaining stages and using current best kernel.\n")
            break

        # ... 继续优化 ...
```

---

### Step 3: 修改 `prompts/optimization.py` - 集成分类指导

更新 `build_optimization_prompt` 函数：

```python
# prompts/optimization.py

from pathlib import Path
from typing import Optional
from config.operator_categories_v2 import build_stage_prompt_section

def build_optimization_prompt(
    arch_path: Path,
    gpu_name: Optional[str] = None,
    *,
    ncu_metrics: str = "",
    history_block: str = "",
    stage_name: str = "",
    stage_description: str = "",
    failure_analysis: str = "",
    # 【新增】分类相关参数
    category: str = "Memory-Intensive",  # 默认值
    stage_id: int = 0,
) -> str:
    """Build optimization prompt with category-specific guidance."""

    gpu_info = _load_gpu_spec()
    # ... GPU信息处理 ...

    # 【新增】构建分类特定的stage context
    category_stage_context = build_stage_prompt_section(category, stage_id)

    # 【保留】原有的通用stage_focus_map（作为fallback）
    # 如果category配置没有，使用原来的
    if category_stage_context:
        stage_context = category_stage_context
    else:
        # Fallback to original stage_focus_map
        stage_focus_map = {
            "grid_and_parallel": """...""",
            # ... 原有的配置 ...
        }
        focus = stage_focus_map.get(stage_name, "")
        stage_context = f"""
## Current Optimization Stage
**Stage**: {stage_description}
{focus}
"""

    # ... 构建最终prompt ...

    return _OPTIMIZATION_PROMPT_TEMPLATE.substitute(
        gpu_name=gpu_name,
        gpu_arch=gpu_arch,
        gpu_items=gpu_items,
        arch_src=arch_src,
        history_block=hist,
        STAGE_CONTEXT=stage_context,
        NCU_METRICS=ncu_section,
        FAILURE_ANALYSIS=failure_context,
    )
```

---

### Step 4: 调用时传入分类信息

在 `main.py` 中调用 `build_optimization_prompt` 时：

```python
# main.py (optimization loop)

opt_prompt = build_optimization_prompt(
    arch_path=best_kernel.code_path,
    gpu_name=args.gpu,
    ncu_metrics=metrics_block,
    history_block=None,
    stage_name=stage_name,
    stage_description=stage_description,
    failure_analysis="",
    # 【新增】传入分类信息
    category=category,
    stage_id=stage_idx,
)
```

---

### Step 5: 选择性提取 NCU 指标

只提取当前stage关注的核心指标：

```python
# main.py (NCU profiling部分)

# 原来：提取所有指标
# metrics_block = metrics_to_prompt(metrics_df)

# 【修改】只提取关键指标
from config.operator_categories_v2 import get_key_ncu_metrics

key_metrics = get_key_ncu_metrics(category, stage_idx)
print(f"Key metrics for this stage: {list(key_metrics.keys())}")

# 过滤NCU指标
filtered_metrics_df = metrics_df[
    metrics_df["Metric Name"].isin(key_metrics.values())
]

metrics_block = metrics_to_prompt(filtered_metrics_df)
print(f"\n[NCU] Extracted {len(filtered_metrics_df)} key metrics")
```

---

### Step 6: Skip Stage 检查

基于分类配置的skip条件：

```python
# main.py (stage loop内部)

from config.operator_categories_v2 import should_skip_stage

# 在NCU profiling之后，优化之前
should_skip, skip_reason = should_skip_stage(
    category=category,
    stage_id=stage_idx,
    op_metadata={
        "op_type": task_name,
        "score": best_score,
    }
)

if should_skip:
    print(f"\n⏩ [Stage {stage_idx + 1}] SKIPPED: {skip_reason}")
    continue
```

---

## 完整集成示例

```python
# main.py - 完整的优化循环

from config.operator_categories_v2 import (
    classify_operator,
    OPERATOR_CATEGORIES,
    build_stage_prompt_section,
    get_key_ncu_metrics,
    check_early_exit,
    should_skip_stage,
)

def run_one_task(task_path: Path, args, ...):
    # ========== 1. 分类算子 ==========
    task_name = task_path.stem
    level = "level2" if "level2" in str(task_path) else "level1"
    category = classify_operator(task_name, level)
    category_config = OPERATOR_CATEGORIES[category]

    print(f"📂 Category: {category} ({category_config['count']} operators)")

    # ========== 2. Seed阶段 ==========
    # ... 原有seed代码 ...

    # ========== 3. 优化循环 ==========
    optimization_stages = category_config["stages"]

    for stage_idx, stage_config in enumerate(optimization_stages):
        stage_name = stage_config["name"]

        print(f"\n{'='*80}")
        print(f"📍 Stage {stage_idx + 1}/{len(optimization_stages)}: {stage_name}")
        print(f"   Focus: {stage_config['focus']}")
        print(f"{'='*80}")

        # ========== 3a. Early Exit检查 ==========
        should_exit, exit_reason = check_early_exit(
            category, stage_idx, best_score,
            {"op_type": task_name, "kernel_size": 3}
        )
        if should_exit:
            print(f"⛔ Early Exit: {exit_reason}")
            break

        # ========== 3b. NCU Profiling (只提取关键指标) ==========
        key_metrics = get_key_ncu_metrics(category, stage_idx)

        csv_path = profile_bench(...)
        metrics_df = load_ncu_metrics(csv_path, ...)

        # 过滤
        filtered_df = metrics_df[
            metrics_df["Metric Name"].isin(key_metrics.values())
        ]
        metrics_block = metrics_to_prompt(filtered_df)

        print(f"📊 Monitoring {len(key_metrics)} key metrics:")
        for name, metric in key_metrics.items():
            print(f"   • {name}: {metric}")

        # ========== 3c. Skip Stage检查 ==========
        should_skip, skip_reason = should_skip_stage(
            category, stage_idx, {"op_type": task_name, "score": best_score}
        )
        if should_skip:
            print(f"⏩ Skipped: {skip_reason}")
            continue

        # ========== 3d. 生成优化prompt ==========
        opt_prompt = build_optimization_prompt(
            arch_path=best_kernel.code_path,
            gpu_name=args.gpu,
            ncu_metrics=metrics_block,
            stage_name=stage_name,
            stage_description=stage_config["description"],
            category=category,
            stage_id=stage_idx,
        )

        # ========== 3e. LLM生成 + benchmark ==========
        current_kernel = _llm_to_kernel(opt_prompt, ...)
        _bench_and_score(current_kernel, ...)

        # ========== 3f. 评估结果 ==========
        if current_kernel.score > best_score:
            print(f"✅ Improved: {best_score:.4f} → {current_kernel.score:.4f}")
            best_kernel = current_kernel
            best_score = current_kernel.score
        else:
            print(f"❌ No improvement: {current_kernel.score:.4f} <= {best_score:.4f}")
            # 继续下一个stage

    # ========== 4. 输出最终结果 ==========
    print(f"\n🏁 Final Best Score: {best_score:.4f}")
    print(f"   Category: {category}")
    return best_kernel
```

---

## 关键改进点

### 1. **动态Stage数量**
- Compute-Intensive: 3个stage
- Memory-Intensive: 3个stage (可能early exit)
- Fusion-Compute: 3个stage
- Fusion-Memory: 3个stage (可能early exit)

### 2. **精简NCU指标**
- 每个stage只看2-3个核心指标
- 减少prompt长度，提高LLM理解

### 3. **智能Early Exit**
- Conv小kernel在stage1后可能退出
- Matmul baseline差在stage1后退出
- 基于分类的退出条件

### 4. **定制化Guidance**
- 每个类别有专门的优化建议
- 代码示例更具体
- 条件判断更清晰

---

## 测试建议

1. **选择4个代表性算子测试**:
   ```bash
   # Compute-Intensive
   python main.py KernelBench/level1/1_Square_matrix_multiplication_.py

   # Memory-Intensive
   python main.py KernelBench/level1/67_conv_standard_1D.py

   # Fusion-Compute
   python main.py KernelBench/level2/56_Matmul_Sigmoid_Sum.py

   # Fusion-Memory
   python main.py KernelBench/level2/1_Conv2D_ReLU_BiasAdd.py
   ```

2. **观察输出**:
   - 是否正确分类
   - Stage guidance是否合理
   - NCU指标是否精简
   - Early exit是否触发

---

## 预期效果

| 类别 | 原来4 stages | 现在stages | 预期改进 |
|------|-------------|-----------|---------|
| Compute-Intensive | 全部执行 | 3个 | 更聚焦计算优化 |
| Memory-Intensive | 全部执行 | 1-3个 (early exit) | Conv早退，节省时间 |
| Fusion-Compute | 全部执行 | 3个 | 渐进式融合 |
| Fusion-Memory | 全部执行 | 1-3个 (early exit) | Conv差时早退 |

---

## 下一步

1. 实现`extract_kernel_size()`辅助函数
2. 在`prompts/optimization.py`中集成分类系统
3. 测试4个代表算子
4. 根据测试结果调整配置
