# Kernel Benchmark Summary - level3
Generated at: 2025-12-16 13:19:35

| # | Kernel Name | Speedup | Status | Ref (ms) | Triton (ms) |
|---|-------------|---------|--------|----------|-------------|
| 1 | 1_MLP | 0.0112 | ✅ | 4.0294 | 359.2445 |
| 2 | 2_ShallowWideMLP | 0.0143 | ✅ | 12.3778 | 864.7576 |
| 3 | 3_DeepNarrowMLP | 0.8746 | ✅ | 2.8044 | 3.2064 |
| 4 | 4_LeNet5 | 0.0691 | ✅ | 1.0235 | 14.8105 |
| 5 | 5_AlexNet | 1.0000 | ✅ | 58.8699 | 58.8698 |
| 6 | 6_GoogleNetInceptionModule | N/A | ❌ | N/A | N/A |
| 7 | 7_GoogleNetInceptionV1 | 1.0177 | ✅ | 2.3740 | 2.3328 |
| 8 | 8_ResNetBasicBlock | 1.0826 | ✅ | 3.6876 | 3.4063 |
| 9 | 9_ResNet18 | 0.8812 | ✅ | 2.1196 | 2.4054 |
| 10 | 10_ResNet101 | 0.9341 | ✅ | 9.4379 | 10.1041 |
| 11 | 11_VGG16 | 0.9215 | ✅ | 8.3069 | 9.0142 |
| 12 | 12_VGG19 | N/A | ❌ | N/A | N/A |
| 13 | 13_DenseNet121TransitionLayer | N/A | ❌ | N/A | N/A |
| 14 | 14_DenseNet121DenseBlock | 1.0020 | ✅ | 16.8523 | 16.8185 |
| 15 | 15_DenseNet121 | N/A | ❌ | N/A | N/A |
| 16 | 16_DenseNet201 | 0.9875 | ✅ | 13.0865 | 13.2516 |
| 17 | 17_SqueezeNetFireModule | 0.0769 | ✅ | 35.2843 | 458.6607 |
| 18 | 18_SqueezeNet | N/A | ❌ | N/A | N/A |
| 19 | 19_MobileNetV1 | 1.0150 | ✅ | 2.3623 | 2.3273 |
| 20 | 20_MobileNetV2 | N/A | ❌ | N/A | N/A |
| 21 | 21_EfficientNetMBConv | 0.0477 | ✅ | 19.8632 | 416.2070 |
| 22 | 22_EfficientNetB0 | 1.0140 | ✅ | 3.2575 | 3.2127 |
| 23 | 23_EfficientNetB1 | N/A | ❌ | N/A | N/A |
| 24 | 24_EfficientNetB2 | N/A | ❌ | N/A | N/A |
| 25 | 25_ShuffleNetUnit | N/A | ❌ | N/A | N/A |
| 26 | 26_ShuffleNet | 1.0057 | ✅ | 14.9292 | 14.8452 |
| 27 | 27_RegNet | 0.9623 | ✅ | 5.2454 | 5.4510 |
| 28 | 28_VisionTransformer | 0.9902 | ✅ | 4.6904 | 4.7369 |
| 29 | 29_SwinMLP | 0.9814 | ✅ | 7.7893 | 7.9369 |
| 30 | 30_SwinTransformerV2 | 1.1392 | ✅ | 21.9947 | 19.3064 |
| 31 | 31_VisionAttention | N/A | ❌ | N/A | N/A |
| 32 | 32_ConvolutionalVisionTransformer | 1.0044 | ✅ | 4.9973 | 4.9754 |
| 33 | 33_VanillaRNN | N/A | ❌ | N/A | N/A |
| 34 | 34_VanillaRNNHidden | 1.1976 | ✅ | 58.0834 | 48.4989 |
| 35 | 35_LSTM | 1.0016 | ✅ | 33.9688 | 33.9130 |
| 36 | 36_LSTMHn | 1.0037 | ✅ | 26.5712 | 26.4722 |
| 37 | 37_LSTMCn | 1.0084 | ✅ | 26.7971 | 26.5745 |
| 38 | 38_LSTMBidirectional | N/A | ❌ | N/A | N/A |
| 39 | 39_GRU | N/A | ❌ | N/A | N/A |
| 40 | 40_GRUHidden | N/A | ❌ | N/A | N/A |
| 41 | 41_GRUBidirectional | N/A | ❌ | N/A | N/A |
| 42 | 42_GRUBidirectionalHidden | N/A | ❌ | N/A | N/A |
| 43 | 43_MinGPTCausalAttention | N/A | ❌ | N/A | N/A |
| 44 | 44_MiniGPTBlock | 1.0000 | ✅ | 55.8788 | 55.8770 |
| 45 | 45_UNetSoftmax | N/A | ❌ | N/A | N/A |
| 46 | 46_NetVladWithGhostClusters | 1.0001 | ✅ | 3.4053 | 3.4051 |
| 47 | 47_NetVladNoGhostClusters | N/A | ❌ | N/A | N/A |
| 48 | 48_Mamba2ReturnY | N/A | ❌ | N/A | N/A |
| 49 | 49_Mamba2ReturnFinalState | N/A | ❌ | N/A | N/A |
| 50 | 50_ReLUSelfAttention | N/A | ❌ | N/A | N/A |

## Summary
- Total kernels: 50
- Successful: 28
- Failed: 22
- Average speedup (successful only): 0.8301
- Success rate: 56.0%

## Failed Kernels
- **6_GoogleNetInceptionModule**: RuntimeError:         pid_batch * input_stride_0 + pid_channel * input_stride_1,
        pid_batch *...
- **12_VGG19**: RuntimeError:         x = tl.load(x_ptrs, mask=mask, other=0.0)
        w = tl.load(w_ptrs, mask=mas...
- **13_DenseNet121TransitionLayer**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **15_DenseNet121**: RuntimeError:                 in_h = h * INPUT_STRIDE_H if h >= 0 and h < in_height else 0
         ...
- **18_SqueezeNet**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/minicon...
- **20_MobileNetV2**: RuntimeError:                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
- **23_EfficientNetB1**: RuntimeError:         element_idx = pid * BLOCK_SIZE + idx
        if element_idx >= num_elements:
 ...
- **24_EfficientNetB2**: RuntimeError:   File "/home/hyc/miniconda3/envs/sglang/lib/python3.11/site-packages/torch/nn/modules...
- **25_ShuffleNetUnit**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **31_VisionAttention**: RuntimeError:                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/h...
- **33_VanillaRNN**: RuntimeError:     mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output...
- **38_LSTMBidirectional**: RuntimeError:     b_hh = tl.load(b_hh_ptr + b_hh_offs, mask=b_hh_offs < 4, other=0.0)

    gates = (...
- **39_GRU**: RuntimeError:         n_j += acc_hn + bias_hn_val
        n_j = n_j * tl.load(r_ptr + r_batch_offset...
- **40_GRUHidden**: CompilationError: invalid syntax (40_GRUHidden.py, line 1)...
- **41_GRUBidirectional**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **42_GRUBidirectionalHidden**: CompilationError: No module named 'triton.ops'...
- **43_MinGPTCausalAttention**: RuntimeError:         exp_a = tl.exp(a_stable)
        softmax_val = exp_a / row_sum
        tl.stor...
- **45_UNetSoftmax**: Timeout: kernel took longer than 120s...
- **47_NetVladNoGhostClusters**: RuntimeError:         _tmp5 = tl.where(rmask & xmask, tmp8, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, No...
- **48_Mamba2ReturnY**: RuntimeError:                   ^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_kernels/level3/48_...
- **49_Mamba2ReturnFinalState**: RuntimeError:     return triton_segsum(x)
           ^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_ke...
- **50_ReLUSelfAttention**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...