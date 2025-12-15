# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        # Higher-ILP, moderate tile, good for large channel counts
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        # Conservative fallback with lowest register pressure
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['P', 'C_out', 'Kd', 'Kh', 'Kw'],
)
@triton.jit
def conv3d_ncdhw_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out,
    D_in, H_in, W_in,
    Kd: tl.constexpr, Kh: tl.constexpr, Kw: tl.constexpr,
    D_out, H_out, W_out,
    P,  # N * D_out * H_out * W_out
    stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw,
    stride_y_n, stride_y_c, stride_y_d, stride_y_h, stride_y_w,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid over output positions (flattened N*D*H*W) and output channels
    pid_m = tl.program_id(0)  # over P = N * D_out * H_out * W_out
    pid_n = tl.program_id(1)  # over C_out

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = offs_m < P
    mask_n = offs_n < C_out
    out_mask = mask_m[:, None] & mask_n[None, :]

    # Decode flattened index offs_m -> (n, od, oh, ow)
    DHW = D_out * H_out * W_out
    HW = H_out * W_out

    n_idx = offs_m // DHW
    rem = offs_m % DHW
    od_idx = rem // HW
    rem = rem % HW
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    # Base input pointer for each output position
    x_base = (
        x_ptr
        + n_idx * stride_x_n
        + od_idx * stride_x_d
        + oh_idx * stride_x_h
        + ow_idx * stride_x_w
    )  # [BLOCK_M]

    # Precompute co offset for weights: [BLOCK_N]
    w_co_offs = offs_n * stride_w_co  # [BLOCK_N]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main convolution loop: reduction over input channels and kernel volume
    for ci in range(0, C_in):
        ci_x_off = ci * stride_x_c
        ci_w_off = ci * stride_w_ci

        for kd in tl.static_range(0, Kd):
            kd_x_off = kd * stride_x_d
            kd_w_off = kd * stride_w_kd

            for kh in tl.static_range(0, Kh):
                kh_x_off = kh * stride_x_h
                kh_w_off = kh * stride_w_kh

                for kw in tl.static_range(0, Kw):
                    kw_x_off = kw * stride_x_w
                    kw_w_off = kw * stride_w_kw

                    # Input pointers: [BLOCK_M]
                    x_ptrs = x_base + ci_x_off + kd_x_off + kh_x_off + kw_x_off
                    x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

                    # Weight pointers: [BLOCK_N]
                    w_ptrs = w_ptr + ci_w_off + kd_w_off + kh_w_off + kw_w_off + w_co_offs
                    w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0).to(tl.float32)

                    # Outer product and accumulate
                    acc += x_vals[:, None] * w_vals[None, :]

    # Bias add
    if HAS_BIAS:
        b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc += b_vals[None, :]

    # Store result
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_y_n
        + offs_n[None, :] * stride_y_c
        + od_idx[:, None] * stride_y_d
        + oh_idx[:, None] * stride_y_h
        + ow_idx[:, None] * stride_y_w
    )
    tl.store(y_ptrs, acc, mask=out_mask)


def conv3d_triton_ncdhw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    3D convolution (N, C, D, H, W) * (C_out, C_in, Kd, Kh, Kw) -> (N, C_out, D_out, H_out, W_out)
    Implemented in Triton for stride=1, padding=0, dilation=1.
    Accumulates in fp32 for numerical stability.
    """
    assert x.dim() == 5, "Input must be NCDHW"
    assert weight.dim() == 5, "Weight must be (C_out, C_in, Kd, Kh, Kw)"
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA device"

    N, C_in, D_in, H_in, W_in = x.shape
    C_out, C_in_w, Kd, Kh, Kw = weight.shape
    assert C_in == C_in_w, "In-channel mismatch between input and weight"

    # Output dimensions (valid convolution, no padding, stride=1)
    D_out = D_in - Kd + 1
    H_out = H_in - Kh + 1
    W_out = W_in - Kw + 1
    assert D_out > 0 and H_out > 0 and W_out > 0, "Kernel larger than input"

    # Use fp32 output/accumulator for numerical stability
    y = torch.empty(
        (N, C_out, D_out, H_out, W_out),
        device=x.device,
        dtype=torch.float32,
    )

    # Flattened number of output positions
    P = N * D_out * H_out * W_out

    # Strides
    stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w = x.stride()
    stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw = weight.stride()
    stride_y_n, stride_y_c, stride_y_d, stride_y_h, stride_y_w = y.stride()

    # 2D grid over spatial+batch (P) and output channels (C_out)
    def grid(meta):
        return (
            triton.cdiv(P, meta['BLOCK_M']),
            triton.cdiv(C_out, meta['BLOCK_N']),
        )

    conv3d_ncdhw_kernel[grid](
        x, weight, bias, y,
        N, C_in, C_out,
        D_in, H_in, W_in,
        Kd, Kh, Kw,
        D_out, H_out, W_out,
        P,
        stride_x_n, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kd, stride_w_kh, stride_w_kw,
        stride_y_n, stride_y_c, stride_y_d, stride_y_h, stride_y_w,
        True,  # HAS_BIAS
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-accelerated version of:
      Conv3d -> Softmax (dim=1) -> MaxPool3d -> MaxPool3d
    Conv3d is implemented in Triton; softmax and pooling use PyTorch.
    """

    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()

        # Normalize kernel_size to 3D (int -> (k,k,k))
        if isinstance(kernel_size, int):
            kd = kh = kw = kernel_size
        else:
            kd, kh, kw = kernel_size
        self.kernel_size = (kd, kh, kw)

        # Parameters for Conv3d: (C_out, C_in, Kd, Kh, Kw)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kd, kh, kw)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        # Initialization similar to nn.Conv3d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kd * kh * kw
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Max pooling layers
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

    def forward(self, x):
        # x: (N, C_in, D, H, W)
        # Conv3d via Triton
        y = conv3d_triton_ncdhw(x, self.weight, self.bias)

        # Softmax over channels (dim=1), then two max-pool layers
        y = torch.softmax(y, dim=1)
        y = self.pool1(y)
        y = self.pool2(y)
        return y
