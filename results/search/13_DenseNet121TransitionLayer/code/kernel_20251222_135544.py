# Optimized Triton code for fused ReLU + 1x1 Conv2d + 2x2 AvgPool2d (stride 2)
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Main high-performance config (balanced for 4090, good reuse vs registers)
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        # Alternative tiling for different C_IN/C_OUT aspect ratios
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=4,
            num_stages=3,
        ),
        # Conservative fallback with lower register pressure
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
            num_warps=2,
            num_stages=2,
        ),
    ],
    key=['M_OUT', 'C_OUT'],
)
@triton.jit
def relu_conv1x1_avgpool2d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C_IN, C_OUT,
    H, W, H_OUT, W_OUT,
    M_OUT,  # B * H_OUT * W_OUT
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_w_co, stride_w_ci,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tiles of pooled (B, H_OUT, W_OUT)
    BLOCK_N: tl.constexpr,  # tiles of output channels
    BLOCK_K: tl.constexpr,  # tiles of input channels
):
    # Program IDs for pooled positions and output channels
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets along M (pooled positions), N (output channels), K (input channels)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M_OUT  # [BLOCK_M]
    mask_n = offs_n < C_OUT  # [BLOCK_N]

    # Map linear pooled index -> (b, y_out, x_out)
    BHWO = H_OUT * W_OUT
    b = offs_m // BHWO
    rem = offs_m % BHWO
    y_out = rem // W_OUT
    x_out = rem % W_OUT

    # Corresponding top-left coordinates in the pre-pooled feature map
    y0_idx = y_out * 2
    x0_idx = x_out * 2

    # Prepare base pointers for the 4 positions in the 2x2 pooling window
    # Shapes: [BLOCK_M, BLOCK_K] for all pointer grids below
    b_bc = b[:, None]
    y0_bc = y0_idx[:, None]
    x0_bc = x0_idx[:, None]
    k_bc = offs_k[None, :]

    x_ptrs0 = (
        x_ptr
        + b_bc * stride_xb
        + k_bc * stride_xc
        + y0_bc * stride_xh
        + x0_bc * stride_xw
    )
    x_ptrs1 = (
        x_ptr
        + b_bc * stride_xb
        + k_bc * stride_xc
        + (y0_bc + 1) * stride_xh
        + x0_bc * stride_xw
    )
    x_ptrs2 = (
        x_ptr
        + b_bc * stride_xb
        + k_bc * stride_xc
        + y0_bc * stride_xh
        + (x0_bc + 1) * stride_xw
    )
    x_ptrs3 = (
        x_ptr
        + b_bc * stride_xb
        + k_bc * stride_xc
        + (y0_bc + 1) * stride_xh
        + (x0_bc + 1) * stride_xw
    )

    # Weight matrix W[K, N] where element [k, n] = w[n, k]
    w_ptrs = (
        w_ptr
        + offs_k[:, None] * stride_w_ci
        + offs_n[None, :] * stride_w_co
    )

    # Accumulator for output tile [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K chunks
    for k in range(0, C_IN, BLOCK_K):
        k_idx = k + offs_k  # [BLOCK_K]
        mask_k = k_idx < C_IN

        # Load weights tile [BLOCK_K, BLOCK_N]
        w_mask = mask_k[:, None] & mask_n[None, :]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Common mask for x loads [BLOCK_M, BLOCK_K]
        xm_mask = mask_m[:, None] & mask_k[None, :]

        # Load and ReLU + sum across 2x2 pooling window BEFORE matmul
        # This exploits linearity: AvgPool(Conv(ReLU(x))) == Conv(AvgPool(ReLU(x)))
        v0 = tl.load(x_ptrs0, mask=xm_mask, other=0.0)
        vsum = tl.maximum(v0, 0.0)

        v1 = tl.load(x_ptrs1, mask=xm_mask, other=0.0)
        vsum += tl.maximum(v1, 0.0)

        v2 = tl.load(x_ptrs2, mask=xm_mask, other=0.0)
        vsum += tl.maximum(v2, 0.0)

        v3 = tl.load(x_ptrs3, mask=xm_mask, other=0.0)
        vsum += tl.maximum(v3, 0.0)

        # Single matmul per K-tile with pooled activations
        acc += tl.dot(vsum, w, allow_tf32=True)

        # Advance pointers along K dimension
        x_ptrs0 += BLOCK_K * stride_xc
        x_ptrs1 += BLOCK_K * stride_xc
        x_ptrs2 += BLOCK_K * stride_xc
        x_ptrs3 += BLOCK_K * stride_xc
        w_ptrs += BLOCK_K * stride_w_ci

    # Apply 2x2 average pooling factor (linear, can be done at the end)
    acc = acc * 0.25

    # Store output: y[b, c_out, y_out, x_out]
    y_ptrs = (
        y_ptr
        + b_bc * stride_yb
        + offs_n[None, :] * stride_yc
        + y_out[:, None] * stride_yh
        + x_out[:, None] * stride_yw
    )
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def relu_conv1x1_avgpool2d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x:       [B, C_in, H, W]  (after BatchNorm)
    weight:  [C_out, C_in, 1, 1]
    Returns: [B, C_out, H//2, W//2] with fused ReLU + 1x1 conv + 2x2 avgpool (stride 2)
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    B, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    assert weight.shape[1] == C_IN and weight.shape[2] == 1 and weight.shape[3] == 1

    H_OUT = H // 2
    W_OUT = W // 2

    # Flatten conv kernel to [C_out, C_in]
    w_2d = weight.view(C_OUT, C_IN).contiguous()

    y = torch.empty((B, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    M_OUT = B * H_OUT * W_OUT

    def grid(meta):
        return (
            triton.cdiv(M_OUT, meta['BLOCK_M']),
            triton.cdiv(C_OUT, meta['BLOCK_N']),
        )

    relu_conv1x1_avgpool2d_kernel[grid](
        x, w_2d, y,
        B, C_IN, C_OUT,
        H, W, H_OUT, W_OUT,
        M_OUT,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_2d.stride(0), w_2d.stride(1),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(ModelNew, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        # Match original Conv2d: 1x1 kernel, no bias
        self.conv = nn.Conv2d(
            num_input_features,
            num_output_features,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        # Keep BatchNorm in PyTorch for correctness
        x = self.bn(x)
        # Fused ReLU + 1x1 Conv2d + AvgPool2d (kernel_size=2, stride=2) in Triton
        x = relu_conv1x1_avgpool2d(x, self.conv.weight)
        return x
