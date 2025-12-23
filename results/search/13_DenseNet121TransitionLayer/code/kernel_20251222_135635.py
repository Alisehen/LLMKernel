import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Larger tile for high arithmetic intensity, good when registers allow
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=3,
        ),
        # Asymmetric tiles for different C_IN/C_OUT aspect ratios
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["C_IN", "C_OUT", "H_OUT", "W_OUT"],
)
@triton.jit
def relu_conv1x1_avgpool2d_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C_IN, C_OUT,
    H, W, H_OUT, W_OUT,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_w_co, stride_w_ci,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_M: tl.constexpr,  # tiles along H_OUT * W_OUT
    BLOCK_N: tl.constexpr,  # tiles along C_OUT
    BLOCK_K: tl.constexpr,  # tiles along C_IN
):
    """
    Fused: ReLU + 2x2 avgpool(stride=2) + 1x1 conv

    We exploit linearity of the 1x1 conv:
        y = (1/4) * sum_p W @ ReLU(x_p)
          = W @ ((1/4) * sum_p ReLU(x_p))

    So we first do ReLU + 2x2 avg over input channels, then a single GEMM.
    Grid:
        pid_m over pooled spatial positions (H_OUT * W_OUT)
        pid_n over output channels
        pid_b over batch
    """

    pid_m = tl.program_id(0)  # pooled spatial tiles (H_OUT * W_OUT)
    pid_n = tl.program_id(1)  # output channel tiles
    pid_b = tl.program_id(2)  # batch index

    # Tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Global shapes
    M_HW = H_OUT * W_OUT

    # Boundary masks for M and N
    mask_m = offs_m < M_HW
    mask_n = offs_n < C_OUT

    # Map linear spatial index -> (y_out, x_out)
    # We will use y_out/x_out only for input pointer setup; recompute for store
    y_out = offs_m // W_OUT
    x_out = offs_m % W_OUT

    # Corresponding top-left coordinates in the pre-pooled feature map
    y0_idx = y_out * 2
    x0_idx = x_out * 2

    # Broadcasted indices for pointer grids
    y0_bc = y0_idx[:, None]
    x0_bc = x0_idx[:, None]
    k_bc = offs_k[None, :]

    # Batch index
    b = pid_b

    # Input pointers for the 4 positions in the 2x2 pooling window
    # Each data tile is [BLOCK_M, BLOCK_K]
    x_ptrs0 = (
        x_ptr
        + b * stride_xb
        + k_bc * stride_xc
        + y0_bc * stride_xh
        + x0_bc * stride_xw
    )
    x_ptrs1 = (
        x_ptr
        + b * stride_xb
        + k_bc * stride_xc
        + (y0_bc + 1) * stride_xh
        + x0_bc * stride_xw
    )
    x_ptrs2 = (
        x_ptr
        + b * stride_xb
        + k_bc * stride_xc
        + y0_bc * stride_xh
        + (x0_bc + 1) * stride_xw
    )
    x_ptrs3 = (
        x_ptr
        + b * stride_xb
        + k_bc * stride_xc
        + (y0_bc + 1) * stride_xh
        + (x0_bc + 1) * stride_xw
    )

    # Weight matrix [C_OUT, C_IN] => logical B[K, N] where [k, n] = w[n, k]
    # Tile shape: [BLOCK_K, BLOCK_N]
    w_ptrs = (
        w_ptr
        + offs_k[:, None] * stride_w_ci
        + offs_n[None, :] * stride_w_co
    )

    # Accumulator for output [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over input channels (K dimension)
    # NOTE: keep BLOCK_K small (32) to mitigate register pressure.
    for k in range(0, C_IN, BLOCK_K):
        k_idx = k + offs_k
        mask_k = k_idx < C_IN

        # Masks shared within this K-tile
        w_mask = mask_k[:, None] & mask_n[None, :]
        x_mask = mask_m[:, None] & mask_k[None, :]

        # Load weights tile
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Fused ReLU + 2x2 average pooling with minimal intermediates:
        # accumulate into a single 'v' tensor to reduce register pressure.
        v = tl.load(x_ptrs0, mask=x_mask, other=0.0)
        v = tl.maximum(v, 0.0)

        tmp1 = tl.load(x_ptrs1, mask=x_mask, other=0.0)
        v += tl.maximum(tmp1, 0.0)

        tmp2 = tl.load(x_ptrs2, mask=x_mask, other=0.0)
        v += tl.maximum(tmp2, 0.0)

        tmp3 = tl.load(x_ptrs3, mask=x_mask, other=0.0)
        v += tl.maximum(tmp3, 0.0)

        v = v * 0.25

        # GEMM: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(v, w, allow_tf32=True)

        # Advance pointers along K dimension
        x_ptrs0 += BLOCK_K * stride_xc
        x_ptrs1 += BLOCK_K * stride_xc
        x_ptrs2 += BLOCK_K * stride_xc
        x_ptrs3 += BLOCK_K * stride_xc
        w_ptrs += BLOCK_K * stride_w_ci

    # Recompute spatial coordinates for the store (cheap, avoids long-lived ints)
    y_out_store = offs_m // W_OUT
    x_out_store = offs_m % W_OUT

    # Store output: y[b, c_out, y_out, x_out]
    y_ptrs = (
        y_ptr
        + b * stride_yb
        + offs_n[None, :] * stride_yc
        + y_out_store[:, None] * stride_yh
        + x_out_store[:, None] * stride_yw
    )
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


def relu_conv1x1_avgpool2d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    x:       [B, C_in, H, W]  (after BatchNorm)
    weight:  [C_out, C_in, 1, 1]
    Returns: [B, C_out, H//2, W//2] with ReLU + 2x2 avgpool(stride=2) + 1x1 conv.
    Math matches: ReLU -> 1x1 Conv -> AvgPool(2x2, stride 2).
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    B, C_IN, H, W = x.shape
    C_OUT = weight.shape[0]
    assert weight.shape[1] == C_IN and weight.shape[2] == 1 and weight.shape[3] == 1
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even for 2x2 avgpool"

    H_OUT = H // 2
    W_OUT = W // 2

    # Flatten conv kernel to [C_out, C_in]
    # Contiguous to guarantee coalesced loads on weights
    w_2d = weight.view(C_OUT, C_IN).contiguous()

    # Output tensor
    y = torch.empty((B, C_OUT, H_OUT, W_OUT), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(H_OUT * W_OUT, meta["BLOCK_M"]),
            triton.cdiv(C_OUT, meta["BLOCK_N"]),
            B,
        )

    relu_conv1x1_avgpool2d_kernel[grid](
        x, w_2d, y,
        B, C_IN, C_OUT,
        H, W, H_OUT, W_OUT,
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
        # BatchNorm in PyTorch for numerical parity
        x = self.bn(x)
        # Fused ReLU + 2x2 AvgPool2d (stride=2) + 1x1 Conv2d in Triton
        x = relu_conv1x1_avgpool2d(x, self.conv.weight)
        return x
