# Bench Script修复

## 问题
NCU profiling失败，错误：
```
TypeError: ModelNew.__init__() missing 3 required positional arguments
```

## 根本原因
`bench_ref_inputs_0.py`假设`ModelNew()`无参数实例化，但许多kernel需要初始化参数（如Conv2D的in_channels, out_channels等）。

## 解决方案

**文件**: `bench_ref_inputs_0.py` (lines 53-62)

**修复前**:
```python
model = ModelNew().to(device).eval()  # ❌ 假设无参数
```

**修复后**:
```python
# Get init inputs from reference module (if available)
get_init_inputs = getattr(ref_mod, "get_init_inputs", None)
if get_init_inputs is not None:
    init_args = get_init_inputs()
    if not isinstance(init_args, (list, tuple)):
        init_args = [init_args]
    model = ModelNew(*init_args).to(device).eval()
else:
    # Fallback: try to instantiate without args
    model = ModelNew().to(device).eval()
```

## 工作原理
1. 尝试从reference模块获取`get_init_inputs()`函数
2. 如果存在，使用返回的参数初始化`ModelNew`
3. 如果不存在，回退到无参数初始化（向后兼容）

## 影响
- ✅ 修复所有需要参数的kernel（Conv2D, Linear等）
- ✅ 保持向后兼容（无参数的kernel仍然工作）
- ✅ 适用于所有4个阶段的NCU profiling

## 测试
重新运行你的Conv2D测试，NCU profiling应该成功。
