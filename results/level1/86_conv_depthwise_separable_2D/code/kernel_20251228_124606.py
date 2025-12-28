import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_depthwise_pointwise_conv2d_kernel(
    x_ptr,        # float32[N, C_IN, H, W]
    w_dw_ptr,     # float32[C_IN, 1, K, K] treated as [C_IN, K, K]
    b_dw_ptr,     # float32[C_IN] or dummy
    w_pw_ptr,     # float32[C_OUT, C_IN, 1, 1] treated as [C_OUT, C_IN]
    b_pw_ptr,     # float32[C_OUT] or dummy
    y_ptr,        # float32[N, C_OUT, H_out, W_out]
    N, C_IN, H, W,
    C_OUT,
    H_out, W_out,
    stride,
    padding,
    dilation,
    KERNEL_SIZE: tl.constexpr,
    HAS_BIAS_DW: tl.constexpr,
    HAS_BIAS_PW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused depthwise (groups=in_channels) + pointwise (1x1) convolution.

    Computes:
        y = Conv1x1( ConvDW(x) ) + bias_pw
    where ConvDW has optional bias_dw, and both convolutions are fused so that
    the intermediate [N, C_IN, H_out, W_out] tensor is never materialized in GMEM.
    """
    # Flatten (N, H_out, W_out) -> M
    M = N * H_out * W_out

    pid_m = tl.program_id(axis=0)  # along M dimension
    pid_n = tl.program_id(axis=1)  # along C_OUT dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < C_OUT

    # Decode m -> (n_idx, h_out_idx, w_out_idx)
    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    h_out_idx = rem // W_out
    w_out_idx = rem % W_out

    n_idx_mat = n_idx[:, None]
    h_out_mat = h_out_idx[:, None]
    w_out_mat = w_out_idx[:, None]
    offs_n_mat = offs_n[None, :]

    # Accumulator for final outputs [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Accumulator for bias contributions (same for all M in tile)
    bias_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Precompute base input coordinates for each output position
    h_in0 = h_out_mat * stride - padding
    w_in0 = w_out_mat * stride - padding

    # Loop over input channels (reduction dimension for pointwise conv)
    for k0 in range(0, C_IN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < C_IN

        # Tile of depthwise outputs: [BLOCK_M, BLOCK_K]
        a = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        k_mat = offs_k[None, :]

        # Depthwise spatial kernel loop (per-channel)
        for kh in range(0, KERNEL_SIZE):
            for kw in range(0, KERNEL_SIZE):
                h_in = h_in0 + kh * dilation
                w_in = w_in0 + kw * dilation

                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

                mask_hw = m_mask[:, None] & in_bounds & k_mask[None, :]

                idx_x = ((n_idx_mat * C_IN + k_mat) * H + h_in) * W + w_in
                x_vals = tl.load(x_ptr + idx_x, mask=mask_hw, other=0.0)

                # depthwise weights: [C_IN, K, K] flattened as c*K*K + kh*K + kw
                w_dw_idx = offs_k * (KERNEL_SIZE * KERNEL_SIZE) + kh * KERNEL_SIZE + kw
                w_dw_vals = tl.load(w_dw_ptr + w_dw_idx, mask=k_mask, other=0.0)

                a += x_vals * w_dw_vals[None, :]

        # Pointwise weights tile: B [BLOCK_K, BLOCK_N]
        k_col = offs_k[:, None]
        n_col = offs_n[None, :]
        idx_b = k_col + n_col * C_IN
        mask_b = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_pw_ptr + idx_b, mask=mask_b, other=0.0)

        # Matmul: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, b, allow_tf32=True)

        # Accumulate depthwise-bias * pointwise-weight contributions:
        # sum_c w_pw[co, c] * b_dw[c]
        if HAS_BIAS_DW:
            b_dw_vec = tl.load(b_dw_ptr + offs_k, mask=k_mask, other=0.0)
            # Each row of B is scaled by b_dw_vec, then summed over K
            bias_acc += tl.sum(b * b_dw_vec[:, None], axis=0)

    # Add pointwise bias
    if HAS_BIAS_PW:
        b_pw = tl.load(b_pw_ptr + offs_n, mask=n_mask, other=0.0)
        bias_acc += b_pw

    # Broadcast bias contributions over M dimension
    acc += bias_acc[None, :]

    # Store result: y[n, co, h_out, w_out]
    idx_out = ((n_idx_mat * C_OUT + offs_n_mat) * H_out + h_out_mat) * W_out + w_out_mat
    mask_out = m_mask[:, None] & n_mask[None, :]
    tl.store(y_ptr + idx_out, acc, mask=mask_out)


def depthwise_separable_conv2d_fused_triton(
    x: torch.Tensor,
    weight_dw: torch.Tensor,
    bias_dw: torch.Tensor,
    weight_pw: torch.Tensor,
    bias_pw: torch.Tensor,
    stride: int,
    padding: int,
    dilation: int,
):
    """
    Fused depthwise-separable 2D convolution:
        depthwise (groups=in_channels, KxK) + pointwise (1x1).

    x:         [N, C_IN, H, W]
    weight_dw: [C_IN, 1, K, K]
    bias_dw:   [C_IN] or None
    weight_pw: [C_OUT, C_IN, 1, 1]
    bias_pw:   [C_OUT] or None
    """
    assert x.is_cuda and weight_dw.is_cuda and weight_pw.is_cuda
    N, C_IN, H, W = x.shape
    C_OUT = weight_pw.shape[0]
    K = weight_dw.shape[2]

    # Output spatial size (as in PyTorch Conv2d)
    H_out = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    W_out = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1

    y = torch.empty((N, C_OUT, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    HAS_BIAS_DW = bias_dw is not None
    HAS_BIAS_PW = bias_pw is not None

    M = N * H_out * W_out
    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(C_OUT, BLOCK_N),
    )

    fused_depthwise_pointwise_conv2d_kernel[grid](
        x,
        weight_dw,
        bias_dw if HAS_BIAS_DW else x,  # dummy pointer if no depthwise bias
        weight_pw,
        bias_pw if HAS_BIAS_PW else x,  # dummy pointer if no pointwise bias
        y,
        N,
        C_IN,
        H,
        W,
        C_OUT,
        H_out,
        W_out,
        stride,
        padding,
        dilation,
        KERNEL_SIZE=K,
        HAS_BIAS_DW=HAS_BIAS_DW,
        HAS_BIAS_PW=HAS_BIAS_PW,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return y


class ModelNew(nn.Module):
    """
    Depthwise-separable 2D convolution implemented with a single fused Triton kernel.
    Matches the behavior of the reference PyTorch Model:
        depthwise = Conv2d(in_channels, in_channels, K, groups=in_channels)
        pointwise = Conv2d(in_channels, out_channels, 1)
        forward(x) = pointwise(depthwise(x))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Keep Conv2d modules so state_dict is fully compatible with the reference model
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on the same device as the weights
        x = x.to(self.depthwise.weight.device)

        y = depthwise_separable_conv2d_fused_triton(
            x,
            self.depthwise.weight,
            self.depthwise.bias,
            self.pointwise.weight,
            self.pointwise.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        return y
