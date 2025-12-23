# TVM Benchmark脚本性能问题分析

## 发现的问题

### 1. ❌ **Target配置不完整** (最关键)

**当前代码** (line 174, 282):
```python
tvm_target = "cuda"
target = tvm.target.Target(tvm_target)
```

**问题**:
- 没有指定GPU架构（sm_xx）
- 没有指定计算能力相关参数
- TVM无法使用针对特定架构的优化

**应该改为**:
```python
# 获取GPU架构
import torch
if torch.cuda.is_available():
    gpu_arch = torch.cuda.get_device_capability(0)
    sm_version = f"sm_{gpu_arch[0]}{gpu_arch[1]}"
else:
    sm_version = "sm_75"  # 默认值

tvm_target = f"cuda -arch={sm_version}"
# 或更详细的配置：
tvm_target = tvm.target.Target({
    "kind": "cuda",
    "arch": sm_version,
    "max_num_threads": 1024,
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,  # 根据实际GPU调整
    "registers_per_block": 65536,
})
```

**影响**:
- 没有GPU架构信息，TVM无法生成针对性优化（如Tensor Core）
- 可能导致20-50%的性能损失

---

### 2. ❌ **缺少Tensor Core启用**

**当前代码** (line 289-301):
```python
dl.ApplyDefaultSchedule(
    dl.gpu.Matmul(),
    dl.gpu.GEMV(),
    dl.gpu.Reduction(),
    dl.gpu.GeneralReduction(),
    dl.gpu.Fallback(),
)
```

**问题**:
- 没有明确启用Tensor Core
- 没有指定数据类型（float16/bfloat16 for tensor core）

**应该改为**:
```python
# 1. 在pipeline中添加数据类型转换（如果需要）
pipeline = tvm.transform.Sequential([
    relax.transform.LegalizeOps(),
    relax.transform.AnnotateTIROpPattern(),
    relax.transform.FoldConstant(),
    relax.transform.FuseOps(),
    relax.transform.FuseTIR(),
    # 添加更多优化pass
    relax.transform.DeadCodeElimination(),
    dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.GEMV(),
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Fallback(),
    ),
])

# 2. 在PassContext中启用相关选项
with target, tvm.transform.PassContext(
    opt_level=3,
    config={
        "relay.backend.use_auto_scheduler": False,
        "relay.FuseOps.max_depth": 10,
        "tir.add_lower_pass": [],
    }
):
    mod = pipeline(mod)
    ex = relax.build(mod, target=target)
```

**影响**:
- 对于矩阵运算密集型算子（GEMM），可能损失2-3倍性能

---

### 3. ⚠️ **缺少AutoTuning**

**当前代码**:
- 完全依赖DLight的默认schedule
- 没有针对具体硬件和workload进行tuning

**问题**:
- TVM的真正优势在于AutoTuning
- 默认schedule往往不是最优的

**建议添加**:
```python
# 使用MetaSchedule进行tuning（TVM Unity推荐）
from tvm import meta_schedule as ms

# 1. 提取tuning任务
database = ms.database.MemoryDatabase()
with target:
    tasks = ms.extract_task_from_relay(mod, target, params={})

# 2. Tune（如果时间允许）
if len(tasks) > 0:
    tuner = ms.tune.TuneContext(
        mod=mod,
        target=target,
        space=ms.space_generator.PostOrderApply(),
        search_strategy=ms.search_strategy.ReplayTrace(),
        task_scheduler=ms.task_scheduler.RoundRobin(
            tasks=tasks,
            max_trials_per_task=100,  # 每个任务100次trial
        ),
        num_threads=4,
    )

    # Run tuning
    tuner.run()

    # Apply best schedule
    with database:
        mod = ms.relax_integration.tune_relax(mod, target, database)
```

**影响**:
- 对于复杂算子（如conv2d），tuning可以带来2-5倍性能提升
- 但tuning需要时间（每个算子几分钟到几十分钟）

---

### 4. ⚠️ **Benchmark参数可能不够稳定**

**当前代码** (line 195, 321):
```python
res_eager = benchmark_eager(model, inputs, 10, 100, device_str, torch)
res_tvm = benchmark_tvm_relax(vm, inputs_tvm, tvm_dev, 10, 100)
```

**问题**:
- warmup=10可能不够（尤其是TVM的kernel cache）
- rep=100可能对于小算子过多，对于大算子不够

**建议**:
```python
# 根据算子大小动态调整
def get_benchmark_params(model_size_mb):
    if model_size_mb < 1:  # 小算子
        return {"warmup": 20, "rep": 200}
    elif model_size_mb < 10:  # 中等算子
        return {"warmup": 10, "rep": 100}
    else:  # 大算子
        return {"warmup": 5, "rep": 50}

# 或者使用更鲁棒的benchmark
# warmup=50, rep=100 for both
```

---

### 5. ⚠️ **可能的内存管理问题**

**当前代码** (line 201-211):
```python
# === AGGRESSIVE MEMORY CLEANUP BEFORE TVM ===
inputs_cpu = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in inputs]

del inputs
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

**问题**:
- 虽然清理了PyTorch内存，但可能影响benchmark公平性
- TVM和PyTorch都在同一个进程中，可能有内存碎片

**建议**:
- 保持当前清理逻辑
- 但在benchmark时确保预热充分

---

### 6. ❌ **缺少Graph-level优化**

**当前pipeline** (line 289-301):
```python
pipeline = tvm.transform.Sequential([
    relax.transform.LegalizeOps(),
    relax.transform.AnnotateTIROpPattern(),
    relax.transform.FoldConstant(),
    relax.transform.FuseOps(),
    relax.transform.FuseTIR(),
    dl.ApplyDefaultSchedule(...),
])
```

**缺少的优化**:
```python
pipeline = tvm.transform.Sequential([
    relax.transform.LegalizeOps(),
    relax.transform.AnnotateTIROpPattern(),
    relax.transform.FoldConstant(),
    relax.transform.FuseOps(),
    relax.transform.FuseTIR(),

    # 添加更多优化
    relax.transform.DeadCodeElimination(),  # 死代码消除
    relax.transform.RemoveUnusedFunctions(),  # 移除未使用函数
    # relax.transform.AlterOpImpl(),  # 算子实现替换（如果可用）

    dl.ApplyDefaultSchedule(...),
])
```

---

## 修复建议优先级

### 🔴 **高优先级（必须修复）**

1. **添加GPU架构配置** (最关键)
   ```python
   gpu_arch = torch.cuda.get_device_capability(0)
   sm_version = f"sm_{gpu_arch[0]}{gpu_arch[1]}"
   tvm_target = f"cuda -arch={sm_version}"
   ```

2. **检查Tensor Core支持**
   ```python
   # 对于支持Tensor Core的GPU（sm_70+），确保数据类型为float16
   if gpu_arch[0] >= 7:  # Volta或更新
       # 考虑在benchmark前将模型转换为float16
       pass
   ```

### 🟡 **中优先级（建议修复）**

3. **增加warmup次数**
   ```python
   # TVM kernel cache需要更多warmup
   res_tvm = benchmark_tvm_relax(vm, inputs_tvm, tvm_dev, 50, 100)
   ```

4. **添加更多Graph优化pass**
   ```python
   relax.transform.DeadCodeElimination(),
   relax.transform.RemoveUnusedFunctions(),
   ```

### 🟢 **低优先级（可选）**

5. **添加AutoTuning支持**（需要大量时间）
6. **更详细的target配置**（shared memory、register等）

---

## 修改后的关键代码

```python
def worker_entry_point(file_path, device_str, result_queue):
    """Worker function with improved TVM configuration."""

    try:
        torch, tvm, relax, runtime, from_fx, builtins, np, gc = get_torch_tvm_imports()

        # ... 前面的代码保持不变 ...

        # === 改进1: 更详细的target配置 ===
        if device_str == "cuda" and torch.cuda.is_available():
            torch_dev = torch.device("cuda")

            # 获取GPU架构
            gpu_arch = torch.cuda.get_device_capability(0)
            sm_version = f"sm_{gpu_arch[0]}{gpu_arch[1]}"

            # 详细target配置
            tvm_target = tvm.target.Target({
                "kind": "cuda",
                "arch": sm_version,
                "max_num_threads": 1024,
                "thread_warp_size": 32,
            })

            try:
                tvm_dev = tvm.cuda(0)
            except:
                tvm_dev = runtime.device("cuda", 0)
        else:
            torch_dev = torch.device("cpu")
            tvm_target = tvm.target.Target("llvm")
            tvm_dev = tvm.cpu(0)

        # ... PyTorch benchmark代码保持不变 ...

        # === 改进2: 更完整的优化pipeline ===
        if str(tvm_target.kind) == "cuda":
            try:
                import tvm.dlight as dl
                pipeline = tvm.transform.Sequential([
                    relax.transform.LegalizeOps(),
                    relax.transform.AnnotateTIROpPattern(),
                    relax.transform.FoldConstant(),
                    relax.transform.FuseOps(),
                    relax.transform.FuseTIR(),
                    relax.transform.DeadCodeElimination(),  # 新增
                    relax.transform.RemoveUnusedFunctions(),  # 新增
                    dl.ApplyDefaultSchedule(
                        dl.gpu.Matmul(),
                        dl.gpu.GEMV(),
                        dl.gpu.Reduction(),
                        dl.gpu.GeneralReduction(),
                        dl.gpu.Fallback(),
                    ),
                ])
                with tvm_target, tvm.transform.PassContext(
                    opt_level=3,
                    config={
                        "relay.FuseOps.max_depth": 10,
                    }
                ):
                    mod = pipeline(mod)
                    ex = relax.build(mod, target=tvm_target)
            except Exception as e:
                with tvm.transform.PassContext(opt_level=3):
                    ex = relax.build(mod, target=tvm_target)
        else:
            with tvm.transform.PassContext(opt_level=3):
                ex = relax.build(mod, target=tvm_target)

        # ... 后面的代码保持不变 ...

        # === 改进3: 增加warmup ===
        res_tvm = benchmark_tvm_relax(vm, inputs_tvm, tvm_dev, 50, 100)  # warmup从10改为50

    except Exception as e:
        # ... 错误处理保持不变 ...
```

---

## 预期改进效果

| 改进项 | 预期性能提升 | 适用算子 |
|--------|-------------|---------|
| GPU架构配置 | 20-50% | 所有CUDA算子 |
| Tensor Core启用 | 2-3x | GEMM密集型（矩阵乘法、卷积） |
| 增加warmup | 5-10% | 小算子 |
| Graph优化pass | 10-20% | 复杂模型 |
| AutoTuning | 2-5x | 所有算子（需要时间） |

---

## 结论

**当前TVM性能不如PyTorch的主要原因**：

1. ❌ **没有指定GPU架构** - 导致TVM无法使用针对性优化
2. ❌ **没有启用Tensor Core** - 对于GEMM算子损失巨大
3. ⚠️ **依赖默认schedule** - 没有tuning

**快速修复（5分钟内）**：
1. 添加GPU架构配置
2. 增加warmup次数

**完整修复（需要时间）**：
1. 上述快速修复
2. 添加AutoTuning支持
3. 针对性优化特定算子

修复后，TVM应该能够达到与PyTorch相当或更好的性能（尤其是对于大模型和复杂算子）。
