import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=["M", "N_OUT", "K"],
)
@triton.jit
def fused_relu_conv1x1_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N_OUT, K,
    N, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wk, stride_wn,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over (M = N*H*W, N_OUT)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N_OUT

    # Decode flattened M -> (n_idx, h_idx, w_idx)
    HW = H * W
    n_idx = offs_m // HW
    rem = offs_m - n_idx * HW
    h_idx = rem // W
    w_idx = rem - h_idx * W

    # Common base pointer for this (n, h, w) row (before channel offset)
    x_row_base = (
        x_ptr
        + n_idx * stride_xn
        + h_idx * stride_xh
        + w_idx * stride_xw
    )

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over input channels K
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A: input activations, apply ReLU in-place
        a_ptrs = x_row_base[:, None] + offs_k[None, :] * stride_xc
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        a = tl.maximum(a, 0.0)

        # B: weight matrix (K, N_OUT)
        b_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

    # Write back: map (m, n_out) -> (n_idx, h_idx, w_idx, c_out)
    y_row_base = (
        y_ptr
        + n_idx * stride_yn
        + h_idx * stride_yh
        + w_idx * stride_yw
    )
    y_ptrs = y_row_base[:, None] + offs_n[None, :] * stride_yc
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
    ],
    key=["OW"],
)
@triton.jit
def avg_pool2d_2x2_kernel(
    x_ptr, y_ptr,
    N, C, H, W, OH, OW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK: tl.constexpr,
):
    # 2D grid:
    #  - axis 0: fused (N, C, OH)
    #  - axis 1: OW tile
    pid_ncoh = tl.program_id(0)
    pid_ow = tl.program_id(1)

    offs_ow = pid_ow * BLOCK + tl.arange(0, BLOCK)
    mask_ow = offs_ow < OW

    # Decode pid_ncoh -> (n, c, oh)
    COH = C * OH
    n = pid_ncoh // COH
    rem = pid_ncoh - n * COH
    c = rem // OH
    oh = rem - c * OH

    # Top-left corner indices in input for each output (oh, ow)
    ih0 = oh * 2
    iw0 = offs_ow * 2

    in_base = (
        x_ptr
        + n * stride_xn
        + c * stride_xc
        + ih0 * stride_xh
        + iw0 * stride_xw
    )

    v00 = tl.load(in_base, mask=mask_ow, other=0.0)
    v01 = tl.load(in_base + stride_xw, mask=mask_ow, other=0.0)
    v10 = tl.load(in_base + stride_xh, mask=mask_ow, other=0.0)
    v11 = tl.load(in_base + stride_xh + stride_xw, mask=mask_ow, other=0.0)

    out = (v00 + v01 + v10 + v11) * 0.25

    out_base = (
        y_ptr
        + n * stride_yn
        + c * stride_yc
        + oh * stride_yh
        + offs_ow * stride_yw
    )
    tl.store(out_base, out, mask=mask_ow)


def fused_relu_conv1x1(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x: (N, C_in, H, W)
    weight: (C_out, C_in, 1, 1)
    Applies ReLU to x, then 1x1 conv with 'weight' using a fused Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors"

    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    M = N * H * W
    K = C_in
    N_out = C_out

    x = x.contiguous()
    weight = weight.contiguous()

    # Weight as (K, N_out)
    w_2d = weight.view(C_out, C_in)
    w_t = w_2d.t().contiguous()  # (K, N_out)

    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_wk, stride_wn = w_t.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N_out, meta["BLOCK_N"]),
        )

    fused_relu_conv1x1_kernel[grid](
        x, w_t, y,
        M, N_out, K,
        N, H, W,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wk, stride_wn,
        stride_yn, stride_yc, stride_yh, stride_yw,
    )
    return y


def avg_pool2d_2x2(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, C, H, W)
    returns: (N, C, H//2, W//2)
    """
    assert x.is_cuda, "Triton kernels require CUDA tensors"

    N, C, H, W = x.shape
    OH = H // 2
    OW = W // 2

    x = x.contiguous()
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    def grid(meta):
        return (
            N * C * OH,
            triton.cdiv(OW, meta["BLOCK"]),
        )

    avg_pool2d_2x2_kernel[grid](
        x, y,
        N, C, H, W, OH, OW,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_yn, stride_yc, stride_yh, stride_yw,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        Optimized version:
        BatchNorm2d -> fused ReLU + 1x1 Conv2d (Triton) -> AvgPool2d 2x2 stride 2 (Triton)
        """
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv_weight = nn.Parameter(
            torch.empty(num_output_features, num_input_features, 1, 1)
        )
        nn.init.kaiming_uniform_(self.conv_weight, a=(5.0 ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_in, H, W)
        x = self.bn(x)
        x = fused_relu_conv1x1(x, self.conv_weight)
        x = avg_pool2d_2x2(x)
        return x
