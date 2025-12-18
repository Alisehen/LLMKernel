# Kernel Benchmark Summary - level1
Generated at: 2025-12-16 12:54:19

| # | Kernel Name | Speedup | Status | Ref (ms) | Triton (ms) |
|---|-------------|---------|--------|----------|-------------|
| 1 | 1_Square_matrix_multiplication_ | N/A | ❌ | N/A | N/A |
| 2 | 2_Standard_matrix_multiplication_ | 1.2138 | ✅ | 2.6490 | 2.1824 |
| 3 | 3_Batched_matrix_multiplication | N/A | ❌ | N/A | N/A |
| 4 | 4_Matrix_vector_multiplication_ | 0.0851 | ✅ | 9.0491 | 106.2913 |
| 5 | 5_Matrix_scalar_multiplication | 0.6781 | ✅ | 9.4395 | 13.9208 |
| 6 | 6_Matmul_with_large_K_dimension_ | 0.1740 | ✅ | 1.6740 | 9.6200 |
| 7 | 7_Matmul_with_small_K_dimension_ | N/A | ❌ | N/A | N/A |
| 8 | 8_Matmul_with_irregular_shapes_ | 0.5211 | ✅ | 6.3480 | 12.1827 |
| 9 | 9_Tall_skinny_matrix_multiplication_ | 0.5465 | ✅ | 5.8016 | 10.6154 |
| 10 | 10_3D_tensor_matrix_multiplication | 0.0902 | ✅ | 1.1233 | 12.4575 |
| 11 | 11_4D_tensor_matrix_multiplication | 0.0612 | ✅ | 9.9564 | 162.6961 |
| 12 | 12_Matmul_with_diagonal_matrices_ | 12.0545 | ✅ | 2.7792 | 0.2306 |
| 13 | 13_Matmul_for_symmetric_matrices | N/A | ❌ | N/A | N/A |
| 14 | 14_Matmul_for_upper_triangular_matrices | 0.6648 | ✅ | 2.8013 | 4.2138 |
| 15 | 15_Matmul_for_lower_triangular_matrices | 0.4880 | ✅ | 2.7925 | 5.7223 |
| 16 | 16_Matmul_with_transposed_A | 0.0012 | ✅ | 2.4333 | 2043.8603 |
| 17 | 17_Matmul_with_transposed_B | N/A | ❌ | N/A | N/A |
| 18 | 18_Matmul_with_transposed_both | N/A | ❌ | N/A | N/A |
| 19 | 19_ReLU | 0.6769 | ✅ | 14.1558 | 20.9120 |
| 20 | 20_LeakyReLU | 0.6774 | ✅ | 14.1527 | 20.8921 |
| 21 | 21_Sigmoid | 0.6768 | ✅ | 14.1373 | 20.8882 |
| 22 | 22_Tanh | 0.6774 | ✅ | 14.1527 | 20.8931 |
| 23 | 23_Softmax | 0.0179 | ✅ | 27.8617 | 1555.6943 |
| 24 | 24_LogSoftmax | 0.7882 | ✅ | 27.8289 | 35.3063 |
| 25 | 25_Swish | 1.6811 | ✅ | 35.1085 | 20.8846 |
| 26 | 26_GELU_ | 0.6774 | ✅ | 14.1519 | 20.8903 |
| 27 | 27_SELU_ | 0.6781 | ✅ | 14.1678 | 20.8932 |
| 28 | 28_HardSigmoid | N/A | ❌ | N/A | N/A |
| 29 | 29_Softplus | 0.6778 | ✅ | 14.1673 | 20.9013 |
| 30 | 30_Softsign | 2.6806 | ✅ | 55.9547 | 20.8737 |
| 31 | 31_ELU | 0.6775 | ✅ | 14.1556 | 20.8941 |
| 32 | 32_HardTanh | 0.9999 | ✅ | 20.8898 | 20.8919 |
| 33 | 33_BatchNorm | 0.0240 | ✅ | 14.5614 | 605.6128 |
| 34 | 34_InstanceNorm | N/A | ❌ | N/A | N/A |
| 35 | 35_GroupNorm_ | N/A | ❌ | N/A | N/A |
| 36 | 36_RMSNorm_ | N/A | ❌ | N/A | N/A |
| 37 | 37_FrobeniusNorm_ | 0.9975 | ✅ | 24.2318 | 24.2920 |
| 38 | 38_L1Norm_ | N/A | ❌ | N/A | N/A |
| 39 | 39_L2Norm_ | N/A | ❌ | N/A | N/A |
| 40 | 40_LayerNorm | 0.0331 | ✅ | 6.4409 | 194.5886 |
| 41 | 41_Max_Pooling_1D | N/A | ❌ | N/A | N/A |
| 42 | 42_Max_Pooling_2D | N/A | ❌ | N/A | N/A |
| 43 | 43_Max_Pooling_3D | 1.0001 | ✅ | 7.5644 | 7.5638 |
| 44 | 44_Average_Pooling_1D | N/A | ❌ | N/A | N/A |
| 45 | 45_Average_Pooling_2D | N/A | ❌ | N/A | N/A |
| 46 | 46_Average_Pooling_3D | N/A | ❌ | N/A | N/A |
| 47 | 47_Sum_reduction_over_a_dimension | N/A | ❌ | N/A | N/A |
| 48 | 48_Mean_reduction_over_a_dimension | 0.0876 | ✅ | 10.0165 | 114.2870 |
| 49 | 49_Max_reduction_over_a_dimension | N/A | ❌ | N/A | N/A |
| 50 | 50_conv_standard_2D__square_input__square_kernel | N/A | ❌ | N/A | N/A |
| 51 | 51_Argmax_over_a_dimension | N/A | ❌ | N/A | N/A |
| 52 | 52_Argmin_over_a_dimension | N/A | ❌ | N/A | N/A |
| 53 | 53_Min_reduction_over_a_dimension | 0.1854 | ✅ | 9.9717 | 53.7847 |
| 54 | 54_conv_standard_3D__square_input__square_kernel | N/A | ❌ | N/A | N/A |
| 55 | 55_conv_standard_2D__asymmetric_input__square_kernel | N/A | ❌ | N/A | N/A |
| 56 | 56_conv_standard_2D__asymmetric_input__asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 57 | 57_conv_transposed_2D__square_input__square_kernel | N/A | ❌ | N/A | N/A |
| 58 | 58_conv_transposed_3D__asymmetric_input__asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 59 | 59_conv_standard_3D__asymmetric_input__square_kernel | N/A | ❌ | N/A | N/A |
| 60 | 60_conv_standard_3D__square_input__asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 61 | 61_conv_transposed_3D__square_input__square_kernel | N/A | ❌ | N/A | N/A |
| 62 | 62_conv_standard_2D__square_input__asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 63 | 63_conv_standard_2D__square_input__square_kernel | N/A | ❌ | N/A | N/A |
| 64 | 64_conv_transposed_1D | N/A | ❌ | N/A | N/A |
| 65 | 65_conv_transposed_2D__square_input__asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 66 | 66_conv_standard_3D__asymmetric_input__asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 67 | 67_conv_standard_1D | N/A | ❌ | N/A | N/A |
| 68 | 68_conv_transposed_3D__square_input__asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 69 | 69_conv_transposed_2D__asymmetric_input__asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 70 | 70_conv_transposed_3D__asymmetric_input__square_kernel | N/A | ❌ | N/A | N/A |
| 71 | 71_conv_transposed_2D__asymmetric_input__square_kernel | N/A | ❌ | N/A | N/A |
| 72 | 72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_ | N/A | ❌ | N/A | N/A |
| 73 | 73_conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped | N/A | ❌ | N/A | N/A |
| 74 | 74_conv_transposed_1D_dilated | N/A | ❌ | N/A | N/A |
| 75 | 75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__ | N/A | ❌ | N/A | N/A |
| 76 | 76_conv_standard_1D_dilated_strided__ | N/A | ❌ | N/A | N/A |
| 77 | 77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__ | N/A | ❌ | N/A | N/A |
| 78 | 78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__ | N/A | ❌ | N/A | N/A |
| 79 | 79_conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__ | N/A | ❌ | N/A | N/A |
| 80 | 80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__ | N/A | ❌ | N/A | N/A |
| 81 | 81_conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__ | N/A | ❌ | N/A | N/A |
| 82 | 82_conv_depthwise_2D_square_input_square_kernel | 1.1679 | ✅ | 4.2152 | 3.6092 |
| 83 | 83_conv_depthwise_2D_square_input_asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 84 | 84_conv_depthwise_2D_asymmetric_input_square_kernel | N/A | ❌ | N/A | N/A |
| 85 | 85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel | N/A | ❌ | N/A | N/A |
| 86 | 86_conv_depthwise_separable_2D | N/A | ❌ | N/A | N/A |
| 87 | 87_conv_pointwise_2D | N/A | ❌ | N/A | N/A |
| 88 | 88_MinGPTNewGelu | N/A | ❌ | N/A | N/A |
| 89 | 89_cumsum | N/A | ❌ | N/A | N/A |
| 90 | 90_cumprod | N/A | ❌ | N/A | N/A |
| 91 | 91_cumsum_reverse | N/A | ❌ | N/A | N/A |
| 92 | 92_cumsum_exclusive | N/A | ❌ | N/A | N/A |
| 93 | 93_masked_cumsum | 0.2171 | ✅ | 22.5525 | 103.8681 |
| 94 | 94_MSELoss | N/A | ❌ | N/A | N/A |
| 95 | 95_CrossEntropyLoss | N/A | ❌ | N/A | N/A |
| 96 | 96_HuberLoss | 2.5527 | ✅ | 23.0252 | 9.0200 |
| 97 | 97_ScaledDotProductAttention | N/A | ❌ | N/A | N/A |
| 98 | 98_KLDivLoss | 5.5724 | ✅ | 12.8410 | 2.3044 |
| 99 | 99_TripletMarginLoss | 3.0625 | ✅ | 13.9852 | 4.5666 |
| 100 | 100_HingeLoss | N/A | ❌ | N/A | N/A |

## Summary
- Total kernels: 100
- Successful: 36
- Failed: 64
- Average speedup (successful only): 1.1963
- Success rate: 36.0%

## Failed Kernels
- **1_Square_matrix_multiplication_**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_kernels/level1/1_...
- **3_Batched_matrix_multiplication**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **7_Matmul_with_small_K_dimension_**: RuntimeError:                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
- **13_Matmul_for_symmetric_matrices**: RuntimeError:         k_offs = k + tl.arange(0, BLOCK_SIZE_K)

        a_mask = (rm + tl.arange(0, B...
- **17_Matmul_with_transposed_B**: RuntimeError:                                tl.full((BLOCK_K, BLOCK_N), j < BLOCK_K, dtype=tl.int1)...
- **18_Matmul_with_transposed_both**: CompilationError: unterminated string literal (detected at line 30) (18_Matmul_with_transposed_both....
- **28_HardSigmoid**: RuntimeError: 
    zero = tl.zeros_like(x)
    one = tl.full_like(x, 1.0)
          ^
AttributeError...
- **34_InstanceNorm**: RuntimeError:         w_mask = offs_w + tl.arange(0, BLOCK_SIZE) < W
        mask = h_mask[:, None] ...
- **35_GroupNorm_**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_kernels/level1/35...
- **36_RMSNorm_**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **38_L1Norm_**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_kernels/level1/38...
- **39_L2Norm_**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_kernels/level1/39...
- **41_Max_Pooling_1D**: RuntimeError: torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search fo...
- **42_Max_Pooling_2D**: RuntimeError:             h = h_start + kh * dilation
            w = w_start + kw * dilation
      ...
- **44_Average_Pooling_1D**: RuntimeError:   File "/home/hyc/miniconda3/envs/sglang/lib/python3.11/site-packages/triton/runtime/j...
- **45_Average_Pooling_2D**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **46_Average_Pooling_3D**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **47_Sum_reduction_over_a_dimension**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **49_Max_reduction_over_a_dimension**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_kernels/level1/49...
- **50_conv_standard_2D__square_input__square_kernel**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **51_Argmax_over_a_dimension**: RuntimeError:         block_max_idx = 0
        for i in range(BLOCK_SIZE):
            if mask[i] a...
- **52_Argmin_over_a_dimension**: RuntimeError:             min_value = float("inf")
            min_index = 0
            for i in ra...
- **54_conv_standard_3D__square_input__square_kernel**: RuntimeError:   File "/home/hyc/generated_kernels/level1/54_conv_standard_3D__square_input__square_k...
- **55_conv_standard_2D__asymmetric_input__square_kernel**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **56_conv_standard_2D__asymmetric_input__asymmetric_kernel**: CompilationError: invalid decimal literal (56_conv_standard_2D__asymmetric_input__asymmetric_kernel....
- **57_conv_transposed_2D__square_input__square_kernel**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **58_conv_transposed_3D__asymmetric_input__asymmetric_kernel**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **59_conv_standard_3D__asymmetric_input__square_kernel**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **60_conv_standard_3D__square_input__asymmetric_kernel**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/minicon...
- **61_conv_transposed_3D__square_input__square_kernel**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **62_conv_standard_2D__square_input__asymmetric_kernel**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **63_conv_standard_2D__square_input__square_kernel**: RuntimeError:   File "/home/hyc/generated_kernels/level1/63_conv_standard_2D__square_input__square_k...
- **64_conv_transposed_1D**: RuntimeError:   File "/home/hyc/generated_kernels/level1/64_conv_transposed_1D.py", line 69, in __in...
- **65_conv_transposed_2D__square_input__asymmetric_kernel**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **66_conv_standard_3D__asymmetric_input__asymmetric_kernel**: RuntimeError:   File "/home/hyc/generated_kernels/level1/66_conv_standard_3D__asymmetric_input__asym...
- **67_conv_standard_1D**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **68_conv_transposed_3D__square_input__asymmetric_kernel**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **69_conv_transposed_2D__asymmetric_input__asymmetric_kernel**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **70_conv_transposed_3D__asymmetric_input__square_kernel**: RuntimeError:     start_idx = pid0 * BLOCK_SIZE
    for idx in range(start_idx, start_idx + BLOCK_SI...
- **71_conv_transposed_2D__asymmetric_input__square_kernel**: RuntimeError:                 w_in = w + padding - kw
                if h_in % stride != 0 or w_in ...
- **72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_**: RuntimeError:     test_model = ModelNew(*init_args, **init_kwargs)
                 ^^^^^^^^^^^^^^^^...
- **73_conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped**: CompilationError: invalid decimal literal (73_conv_transposed_3D_asymmetric_input_square_kernel__str...
- **74_conv_transposed_1D_dilated**: RuntimeError: Traceback (most recent call last):
  File "/home/hyc/LLMKernel/utils/compile_and_run.p...
- **75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__**: RuntimeError:                 h_in = h_out + padding_h - k_h * dilation_h
                w_in = w_o...
- **76_conv_standard_1D_dilated_strided__**: RuntimeError: 
    output_val = 0.0
    if has_bias:
    ^
AssertionError('initial value for `output...
- **77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__**: RuntimeError:                         h_out = h_in * stride + kw - padding

                        ...
- **78_conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__**: RuntimeError:     pid1 = tl.program_id(1)  # output channel
    pid2 = tl.program_id(2)  # output he...
- **79_conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__**: RuntimeError: torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaError...
- **80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__**: RuntimeError:                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
...
- **81_conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__**: RuntimeError:     pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)...
- **83_conv_depthwise_2D_square_input_asymmetric_kernel**: CompilationError: invalid decimal literal (83_conv_depthwise_2D_square_input_asymmetric_kernel.py, l...
- **84_conv_depthwise_2D_asymmetric_input_square_kernel**: RuntimeError:     if bias_ptr != 0:
        bias_offset = c_out * bias_channel_stride
        bias_v...
- **85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **86_conv_depthwise_separable_2D**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **87_conv_pointwise_2D**: RuntimeError:   File "/home/hyc/generated_kernels/level1/87_conv_pointwise_2D.py", line 68, in __ini...
- **88_MinGPTNewGelu**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/LLMKernel/KernelBench/level...
- **89_cumsum**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **90_cumprod**: RuntimeError:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home...
- **91_cumsum_reverse**: RuntimeError:     vec_start = x_ptr + pid * vec_len
    out_vec_start = output_ptr + pid * vec_len
 ...
- **92_cumsum_exclusive**: RuntimeError:     current = 0
    for i in range(0, inner_dim):
        if i == 0:
        ^
Asserti...
- **94_MSELoss**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/LLMKernel/KernelBench/level...
- **95_CrossEntropyLoss**: RuntimeError:     offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size

...
- **97_ScaledDotProductAttention**: RuntimeError:     max_score = tl.full((128,), float('-inf'), dtype=tl.float32)

    for k_block in r...
- **100_HingeLoss**: RuntimeError:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/hyc/generated_kernels/level1/10...