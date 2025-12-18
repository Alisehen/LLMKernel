# Kernel Benchmark Summary - level2
Generated at: 2025-12-16 13:06:54

| # | Kernel Name | Speedup | Status | Ref (ms) | Triton (ms) |
|---|-------------|---------|--------|----------|-------------|
| 1 | 1_Conv2D_ReLU_BiasAdd | 0.0833 | ✅ | 12.7310 | 152.7776 |
| 2 | 2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide | 1.7046 | ✅ | 39.8727 | 23.3911 |
| 3 | 3_ConvTranspose3d_Sum_LayerNorm_AvgPool_GELU | 0.9391 | ✅ | 18.3591 | 19.5492 |
| 4 | 4_Conv2d_Mish_Mish | N/A | ❌ | N/A | N/A |
| 5 | 5_ConvTranspose2d_Subtract_Tanh | N/A | ❌ | N/A | N/A |
| 6 | 6_Conv3d_Softmax_MaxPool_MaxPool | 0.2023 | ✅ | 1.3010 | 6.4315 |
| 7 | 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd | N/A | ❌ | N/A | N/A |
| 8 | 8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum | 1.0000 | ✅ | 7.8677 | 7.8674 |
| 9 | 9_Matmul_Subtract_Multiply_ReLU | N/A | ❌ | N/A | N/A |
| 10 | 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh | N/A | ❌ | N/A | N/A |
| 11 | 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm | N/A | ❌ | N/A | N/A |
| 12 | 12_Gemm_Multiply_LeakyReLU | 0.9923 | ✅ | 3.2344 | 3.2594 |
| 13 | 13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling | N/A | ❌ | N/A | N/A |
| 14 | 14_Gemm_Divide_Sum_Scaling | N/A | ❌ | N/A | N/A |
| 15 | 15_ConvTranspose3d_BatchNorm_Subtract | 0.9292 | ✅ | 3.8586 | 4.1526 |
| 16 | 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling | 1.0002 | ✅ | 37.4032 | 37.3955 |
| 17 | 17_Conv2d_InstanceNorm_Divide | 0.1477 | ✅ | 15.0184 | 101.6866 |
| 18 | 18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp | N/A | ❌ | N/A | N/A |
| 19 | 19_ConvTranspose2d_GELU_GroupNorm | 0.9383 | ✅ | 34.0497 | 36.2891 |
| 20 | 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd | 0.9978 | ✅ | 12.0790 | 12.1054 |
| 21 | 21_Conv2d_Add_Scale_Sigmoid_GroupNorm | 1.3916 | ✅ | 16.4137 | 11.7952 |
| 22 | 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish | N/A | ❌ | N/A | N/A |
| 23 | 23_Conv3d_GroupNorm_Mean | N/A | ❌ | N/A | N/A |
| 24 | 24_Conv3d_Min_Softmax | N/A | ❌ | N/A | N/A |
| 25 | 25_Conv2d_Min_Tanh_Tanh | 0.1709 | ✅ | 14.6959 | 86.0051 |
| 26 | 26_ConvTranspose3d_Add_HardSwish | 1.4776 | ✅ | 18.2184 | 12.3293 |
| 27 | 27_Conv3d_HardSwish_GroupNorm_Mean | 0.9992 | ✅ | 15.0674 | 15.0793 |
| 28 | 28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply | N/A | ❌ | N/A | N/A |
| 29 | 29_Matmul_Mish_Mish | N/A | ❌ | N/A | N/A |
| 30 | 30_Gemm_GroupNorm_Hardtanh | 0.0009 | ✅ | 3.3653 | 3845.8422 |
| 31 | 31_Conv2d_Min_Add_Multiply | 1.2969 | ✅ | 15.0479 | 11.6026 |
| 32 | 32_Conv2d_Scaling_Min | 1.2437 | ✅ | 23.5897 | 18.9670 |
| 33 | 33_Gemm_Scale_BatchNorm | 0.9940 | ✅ | 3.2779 | 3.2978 |
| 34 | 34_ConvTranspose3d_LayerNorm_GELU_Scaling | N/A | ❌ | N/A | N/A |
| 35 | 35_Conv2d_Subtract_HardSwish_MaxPool_Mish | N/A | ❌ | N/A | N/A |
| 36 | 36_ConvTranspose2d_Min_Sum_GELU_Add | N/A | ❌ | N/A | N/A |
| 37 | 37_Matmul_Swish_Sum_GroupNorm | 1.0475 | ✅ | 13.2257 | 12.6260 |
| 38 | 38_ConvTranspose3d_AvgPool_Clamp_Softmax_Multiply | 1.0009 | ✅ | 17.1279 | 17.1128 |
| 39 | 39_Gemm_Scale_BatchNorm | 0.9916 | ✅ | 14.0088 | 14.1271 |
| 40 | 40_Matmul_Scaling_ResidualAdd | N/A | ❌ | N/A | N/A |
| 41 | 41_Gemm_BatchNorm_GELU_ReLU | N/A | ❌ | N/A | N/A |
| 42 | 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply | 0.9975 | ✅ | 22.1727 | 22.2280 |
| 43 | 43_Conv3d_Max_LogSumExp_ReLU | 0.7077 | ✅ | 6.8473 | 9.6754 |
| 44 | 44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean | N/A | ❌ | N/A | N/A |
| 45 | 45_Gemm_Sigmoid_LogSumExp | N/A | ❌ | N/A | N/A |
| 46 | 46_Conv2d_Subtract_Tanh_Subtract_AvgPool | N/A | ❌ | N/A | N/A |
| 47 | 47_Conv3d_Mish_Tanh | N/A | ❌ | N/A | N/A |
| 48 | 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid | N/A | ❌ | N/A | N/A |
| 49 | 49_ConvTranspose3d_Softmax_Sigmoid | 1.0000 | ✅ | 6.7604 | 6.7601 |
| 50 | 50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling | N/A | ❌ | N/A | N/A |
| 51 | 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd | N/A | ❌ | N/A | N/A |
| 52 | 52_Conv2d_Activation_BatchNorm | N/A | ❌ | N/A | N/A |
| 53 | 53_Gemm_Scaling_Hardtanh_GELU | 1.0364 | ✅ | 6.7708 | 6.5333 |
| 54 | 54_Conv2d_Multiply_LeakyReLU_GELU | 1.2632 | ✅ | 16.8717 | 13.3565 |
| 55 | 55_Matmul_MaxPool_Sum_Scale | N/A | ❌ | N/A | N/A |
| 56 | 56_Matmul_Sigmoid_Sum | N/A | ❌ | N/A | N/A |
| 57 | 57_Conv2d_ReLU_HardSwish | 1.7245 | ✅ | 8.1306 | 4.7149 |
| 58 | 58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp | N/A | ❌ | N/A | N/A |
| 59 | 59_Matmul_Swish_Scaling | 1.0025 | ✅ | 6.2578 | 6.2420 |
| 60 | 60_ConvTranspose3d_Swish_GroupNorm_HardSwish | 0.9989 | ✅ | 24.5578 | 24.5856 |
| 61 | 61_ConvTranspose3d_ReLU_GroupNorm | N/A | ❌ | N/A | N/A |
| 62 | 62_Matmul_GroupNorm_LeakyReLU_Sum | 0.7608 | ✅ | 3.5666 | 4.6878 |
| 63 | 63_Gemm_ReLU_Divide | 0.9624 | ✅ | 3.2511 | 3.3781 |
| 64 | 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU | N/A | ❌ | N/A | N/A |
| 65 | 65_Conv2d_AvgPool_Sigmoid_Sum | 0.9881 | ✅ | 26.8020 | 27.1245 |
| 66 | 66_Matmul_Dropout_Softmax | N/A | ❌ | N/A | N/A |
| 67 | 67_Conv2d_GELU_GlobalAvgPool | N/A | ❌ | N/A | N/A |
| 68 | 68_Matmul_Min_Subtract | 1.0010 | ✅ | 1.6664 | 1.6648 |
| 69 | 69_Conv2d_HardSwish_ReLU | 0.8953 | ✅ | 4.6931 | 5.2417 |
| 70 | 70_Gemm_Sigmoid_Scaling_ResidualAdd | 1.0009 | ✅ | 3.3039 | 3.3008 |
| 71 | 71_Conv2d_Divide_LeakyReLU | 1.1667 | ✅ | 4.1496 | 3.5566 |
| 72 | 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool | N/A | ❌ | N/A | N/A |
| 73 | 73_Conv2d_BatchNorm_Scaling | 0.8979 | ✅ | 4.7367 | 5.2753 |
| 74 | 74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max | 0.9400 | ✅ | 4.5166 | 4.8048 |
| 75 | 75_Gemm_GroupNorm_Min_BiasAdd | N/A | ❌ | N/A | N/A |
| 76 | 76_Gemm_Add_ReLU | 0.9820 | ✅ | 3.2982 | 3.3586 |
| 77 | 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool | 0.9885 | ✅ | 30.4087 | 30.7633 |
| 78 | 78_ConvTranspose3d_Max_Max_Sum | 1.0000 | ✅ | 1077.3145 | 1077.2676 |
| 79 | 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max | 0.9473 | ✅ | 2.0450 | 2.1588 |
| 80 | 80_Gemm_Max_Subtract_GELU | 1.0017 | ✅ | 3.1927 | 3.1872 |
| 81 | 81_Gemm_Swish_Divide_Clamp_Tanh_Clamp | N/A | ❌ | N/A | N/A |
| 82 | 82_Conv2d_Tanh_Scaling_BiasAdd_Max | N/A | ❌ | N/A | N/A |
| 83 | 83_Conv3d_GroupNorm_Min_Clamp_Dropout | N/A | ❌ | N/A | N/A |
| 84 | 84_Gemm_BatchNorm_Scaling_Softmax | 0.9857 | ✅ | 3.2712 | 3.3188 |
| 85 | 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp | 0.9227 | ✅ | 6.0333 | 6.5385 |
| 86 | 86_Matmul_Divide_GELU | N/A | ❌ | N/A | N/A |
| 87 | 87_Conv2d_Subtract_Subtract_Mish | 0.9076 | ✅ | 21.4003 | 23.5785 |
| 88 | 88_Gemm_GroupNorm_Swish_Multiply_Swish | 1.0154 | ✅ | 3.5835 | 3.5291 |
| 89 | 89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max | 0.4599 | ✅ | 15.6597 | 34.0525 |
| 90 | 90_Conv3d_LeakyReLU_Sum_Clamp_GELU | 1.4910 | ✅ | 29.5024 | 19.7864 |
| 91 | 91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid | 1.2461 | ✅ | 24.4503 | 19.6219 |
| 92 | 92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp | N/A | ❌ | N/A | N/A |
| 93 | 93_ConvTranspose2d_Add_Min_GELU_Multiply | N/A | ❌ | N/A | N/A |
| 94 | 94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm | N/A | ❌ | N/A | N/A |
| 95 | 95_Matmul_Add_Swish_Tanh_GELU_Hardtanh | 1.0548 | ✅ | 3.4186 | 3.2410 |
| 96 | 96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp | 1.0057 | ✅ | 15.4780 | 15.3901 |
| 97 | 97_Matmul_BatchNorm_BiasAdd_Divide_Swish | N/A | ❌ | N/A | N/A |
| 98 | 98_Matmul_AvgPool_GELU_Scale_Max | 0.8882 | ✅ | 3.1600 | 3.5578 |
| 99 | 99_Matmul_GELU_Softmax | N/A | ❌ | N/A | N/A |
| 100 | 100_ConvTranspose3d_Clamp_Min_Divide | 1.1117 | ✅ | 39.5074 | 35.5370 |

## Summary
- Total kernels: 100
- Successful: 55
- Failed: 45
- Average speedup (successful only): 0.9619
- Success rate: 55.0%

## Failed Kernels
- **4_Conv2d_Mish_Mish**: CompilationError: unterminated string literal (detected at line 5) (4_Conv2d_Mish_Mish.py, line 5)...
- **5_ConvTranspose2d_Subtract_Tanh**: RuntimeError:     b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    result = x - b
    result = tl...
- **7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd**: RuntimeError:                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
- **9_Matmul_Subtract_Multiply_ReLU**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_kernels/level2/9_...
- **10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh**: RuntimeError: 
    shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    shared = tl.store(shared +...
- **11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **14_Gemm_Divide_Sum_Scaling**: CompilationError: unterminated string literal (detected at line 5) (14_Gemm_Divide_Sum_Scaling.py, l...
- **18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp**: CompilationError: unterminated string literal (detected at line 2) (18_Matmul_Sum_Max_AvgPool_LogSum...
- **22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish**: RuntimeError:     gt_threshold = x > threshold
    mish_x = tl.where(gt_threshold, x + log1p_exp_x, ...
- **23_Conv3d_GroupNorm_Mean**: RuntimeError:     tid = tl.program_id(1)

    if pid >= batch_size:
              ^
NameError('batch...
- **24_Conv3d_Min_Softmax**: RuntimeError: torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search fo...
- **28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **29_Matmul_Mish_Mish**: RuntimeError: 
    exp_x = tl.math.exp(x)
    log1p_exp_x = tl.math.log1p(exp_x)
                  ^...
- **34_ConvTranspose3d_LayerNorm_GELU_Scaling**: RuntimeError:         val = tl.load(x_ptr + offset, mask=mask, other=0.0)
        val = (val - mean)...
- **35_Conv2d_Subtract_HardSwish_MaxPool_Mish**: RuntimeError:     exp_x = tl.math.exp(x)
    log1p_exp_x = tl.math.log(1.0 + exp_x)
    tanh_log = t...
- **36_ConvTranspose2d_Min_Sum_GELU_Add**: RuntimeError:     x = sum_acc
    tanh_input = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    g...
- **40_Matmul_Scaling_ResidualAdd**: RuntimeError:   File "/home/hyc/miniconda3/envs/sglang/lib/python3.11/site-packages/triton/compiler/...
- **41_Gemm_BatchNorm_GELU_ReLU**: RuntimeError:     x_cubic = x_squared * x
    gelu_input = 0.7978845608028654 * x + 0.044715 * x_cub...
- **44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean**: RuntimeError: 
    local_sum = 0.0
    for i in range(0, elements_per_thread):
    ^
AssertionError(...
- **45_Gemm_Sigmoid_LogSumExp**: RuntimeError:     val = tl.load(offset, mask=tid < row_size, other=-float('inf'))

    max_val = tl....
- **46_Conv2d_Subtract_Tanh_Subtract_AvgPool**: RuntimeError:     pid_channel = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program...
- **47_Conv3d_Mish_Tanh**: RuntimeError:     x = tl.load(x_ptr + offsets, mask=mask)
    exp_x = tl.math.exp(x)
    log1p_exp_x...
- **48_Conv3d_Scaling_Tanh_Multiply_Sigmoid**: RuntimeError: 
    y = x * scale
    y = tl.math.tanh(y)
        ^
AttributeError("module 'triton.la...
- **50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling**: RuntimeError:     num_elements = batch * out_channels * depth * height * width
    for idx in range(...
- **51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_kernels/level2/51...
- **52_Conv2d_Activation_BatchNorm**: RuntimeError:     softplus_x = max_val + log1p_exp

    tanh_x = tl.tanh(softplus_x)
             ^
...
- **55_Matmul_MaxPool_Sum_Scale**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **56_Matmul_Sigmoid_Sum**: RuntimeError:     if ACC_TYPE == tl.float16:
        acc = acc.to(tl.float16)
    tl.store(out_ptrs ...
- **58_ConvTranspose3d_LogSumExp_HardSwish_Subtract_Clamp**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **61_ConvTranspose3d_ReLU_GroupNorm**: CompilationError: invalid decimal literal (61_ConvTranspose3d_ReLU_GroupNorm.py, line 4)...
- **64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU**: CompilationError: invalid syntax (64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU.py, line 1)...
- **66_Matmul_Dropout_Softmax**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **67_Conv2d_GELU_GlobalAvgPool**: RuntimeError:     mask = offsets < num_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    ...
- **72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool**: CompilationError: duplicate argument 'OW' in function definition (72_ConvTranspose3d_BatchNorm_AvgPo...
- **75_Gemm_GroupNorm_Min_BiasAdd**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **81_Gemm_Swish_Divide_Clamp_Tanh_Clamp**: RuntimeError:     swish = swish * 0.5
    swish = tl.minimum(tl.maximum(swish, -1.0), 1.0)
    tanh_...
- **82_Conv2d_Tanh_Scaling_BiasAdd_Max**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **83_Conv3d_GroupNorm_Min_Clamp_Dropout**: RuntimeError:     BLOCK_SIZE: tl.constexpr,
):
    min_val = tl.load(min_val_ptr)
              ^
Un...
- **86_Matmul_Divide_GELU**: RuntimeError:     term = x_div + 0.044715 * x_div_cubic
    term = term * 0.7978845608028654
    ter...
- **92_Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **93_ConvTranspose2d_Add_Min_GELU_Multiply**: RuntimeError:     tmp = x + tmp
    tmp = c0 * tmp
    tmp = tl.math.tanh(tmp)
          ^
Attribute...
- **94_Gemm_BiasAdd_Hardtanh_Mish_GroupNorm**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **97_Matmul_BatchNorm_BiasAdd_Divide_Swish**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **99_Matmul_GELU_Softmax**: RuntimeError:     x_cubed = x_squared * x
    tanh_input = sqrt2_over_pi * (x + c * x_cubed)
    tan...