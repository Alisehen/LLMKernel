import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_full_kernel(
    x_ptr,        # *f32, [N, C_in, H_in, W_in]
    w_ptr,        # *f32, [C_in, C_out, KH, KW]
    y_ptr,        # *f32, [N, C_out, H_out, W_out]
    bias_ptr,     # *f32, [C_out] (unused if HAS_BIAS == False)
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH, KW,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Implements ConvTranspose2d for the special case:
        stride = 1, padding = 0, output_padding = 0, dilation = 1, groups = 1
    which is equivalent to a 2D full convolution:
        y[n, co, ho, wo] = sum_{ci, kh, kw} x[n, ci, ho - kh, wo - kw] * w[ci, co, kh, kw]
    with appropriate boundary checks.

    Tiling:
      - M dimension: N * H_out * W_out (output positions)
      - N dimension: C_out (output channels)
      - K dimension (reduced with tl.dot): C_in, iterated in BLOCK_K chunks
      - KH, KW are iterated explicitly in small loops (kernel-size loops)
    """
    pid_m = tl.program_id(axis=0)  # over output positions (N * H_out * W_out)
    pid_n = tl.program_id(axis=1)  # over output channels (C_out)

    # ----- Compute tile of output indices -----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    total_positions = N * H_out * W_out
    mask_m = offs_m < total_positions
    mask_n = offs_n < C_out

    # Decompose offs_m into (n_idx, ho, wo)
    hw_out = H_out * W_out
    n_idx = offs_m // hw_out                  # [BLOCK_M]
    rem = offs_m % hw_out
    ho = rem // W_out                         # [BLOCK_M]
    wo = rem % W_out                          # [BLOCK_M]

    # Prepare accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pre-broadcast n_idx, ho, wo for pointer arithmetic
    n_b = n_idx[:, None]                      # [BLOCK_M, 1]
    ho_b = ho[:, None]                        # [BLOCK_M, 1]
    wo_b = wo[:, None]                        # [BLOCK_M, 1]
    co_idx = offs_n                           # [BLOCK_N]
    co_b = co_idx[None, :]                    # [1, BLOCK_N]

    # ----- Main loops over kernel spatial dims and input channels -----
    for kh in range(0, KH):
        hi = ho - kh                          # [BLOCK_M]
        valid_hi = (hi >= 0) & (hi < H_in)

        hi_b = hi[:, None]                    # [BLOCK_M, 1]

        for kw in range(0, KW):
            wi = wo - kw                      # [BLOCK_M]
            valid_wi = (wi >= 0) & (wi < W_in)

            wi_b = wi[:, None]                # [BLOCK_M, 1]
            mask_hw = mask_m & valid_hi & valid_wi  # [BLOCK_M]

            # Skip computation if no valid positions in this tile (masking handles it anyway)
            # We don't branch on tl.tensors; we just keep masks.
            for c_start in range(0, C_in, BLOCK_K):
                ci = c_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                mask_ci = ci < C_in                   # [BLOCK_K]

                ci_b_row = ci[None, :]                # [1, BLOCK_K]
                ci_b_col = ci[:, None]                # [BLOCK_K, 1]

                # Load input tile X: shape [BLOCK_M, BLOCK_K]
                # x[n, ci, hi, wi]
                x_ptrs = (
                    ((n_b * C_in + ci_b_row) * H_in + hi_b) * W_in + wi_b
                )  # [BLOCK_M, BLOCK_K]
                mask_x = mask_hw[:, None] & mask_ci[None, :]  # [BLOCK_M, BLOCK_K]
                x_tile = tl.load(x_ptr + x_ptrs, mask=mask_x, other=0.0)

                # Load weight tile W: shape [BLOCK_K, BLOCK_N]
                # w[ci, co, kh, kw]
                w_ptrs = (
                    ((ci_b_col * C_out + co_b) * KH + kh) * KW + kw
                )  # [BLOCK_K, BLOCK_N]
                mask_w = mask_ci[:, None] & mask_n[None, :]  # [BLOCK_K, BLOCK_N]
                w_tile = tl.load(w_ptr + w_ptrs, mask=mask_w, other=0.0)

                # Accumulate
                acc += tl.dot(x_tile, w_tile)

    # ----- Add bias if present -----
    if HAS_BIAS:
        bias = tl.load(bias_ptr + co_idx, mask=mask_n, other=0.0)  # [BLOCK_N]
        acc = acc + bias[None, :]                                  # [BLOCK_M, BLOCK_N]

    # ----- Store results -----
    y_ptrs = (
        ((n_b * C_out + co_b) * H_out + ho_b) * W_out + wo_b
    )  # [BLOCK_M, BLOCK_N]
    mask_y = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + y_ptrs, acc, mask=mask_y)


def triton_conv_transpose2d_full(x: torch.Tensor,
                                 weight: torch.Tensor,
                                 bias: torch.Tensor | None) -> torch.Tensor:
    """
    Triton implementation of ConvTranspose2d for the restricted but
    common case used in the target model:

        stride = 1
        padding = 0
        output_padding = 0
        dilation = 1
        groups = 1

    which is equivalent to a 2D full convolution.

    Args:
        x:      [N, C_in, H_in, W_in], float32, CUDA
        weight: [C_in, C_out, KH, KW], float32, CUDA
        bias:   [C_out] or None, float32, CUDA

    Returns:
        y: [N, C_out, H_out, W_out]
           where H_out = H_in + KH - 1, W_out = W_in + KW - 1
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, "Only float32 is supported in Triton kernel"
    assert x.ndim == 4 and weight.ndim == 4, "x must be [N,C,H,W], weight must be [C_in,C_out,KH,KW]"
    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out, KH, KW = weight.shape
    assert C_in_w == C_in, "Input channels mismatch between x and weight"

    H_out = H_in + KH - 1
    W_out = W_in + KW - 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_M = 64  # power of 2
    BLOCK_N = 64  # power of 2
    BLOCK_K = 32  # power of 2

    grid = lambda meta: (
        triton.cdiv(N * H_out * W_out, meta["BLOCK_M"]),
        triton.cdiv(C_out, meta["BLOCK_N"]),
    )

    has_bias = bias is not None
    if bias is None:
        # Dummy pointer; will not be used when HAS_BIAS=False (compile-time)
        bias = y.new_empty(1)

    conv_transpose2d_full_kernel[grid](
        x, weight, y, bias,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW,
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated equivalent of the provided ConvTranspose2d model.

    For now, the Triton kernel is used when:
      - stride == 1
      - padding == 0
      - output_padding == 0
      - dilation == 1
      - groups == 1
      - input/weights are CUDA float32

    Otherwise, it falls back to the PyTorch ConvTranspose2d implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ct = self.conv_transpose2d

        # Conditions under which we use the Triton kernel
        use_triton = (
            x.is_cuda
            and x.dtype == torch.float32
            and ct.weight.is_cuda
            and ct.weight.dtype == torch.float32
            and ct.groups == 1
            and ct.dilation == (1, 1)
            and ct.stride == (1, 1)
            and ct.padding == (0, 0)
            and ct.output_padding == (0, 0)
        )

        if use_triton:
            return triton_conv_transpose2d_full(x, ct.weight, ct.bias)
        else:
            # Fallback for unsupported configurations
            return ct(x)
