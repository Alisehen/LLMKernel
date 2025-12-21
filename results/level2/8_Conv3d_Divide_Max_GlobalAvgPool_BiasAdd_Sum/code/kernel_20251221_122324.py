import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline config (matches typical Triton defaults, kept for safety)
        triton.Config(
            {"BLOCK_C": 128},
            num_warps=4,
            num_stages=2,
        ),
        # More aggressive pipelining for hiding memory latency
        triton.Config(
            {"BLOCK_C": 128},
            num_warps=4,
            num_stages=3,
        ),
        # Higher warp count for better occupancy on 4090 if register pressure allows
        triton.Config(
            {"BLOCK_C": 128},
            num_warps=8,
            num_stages=2,
        ),
    ],
    # Channel dimension is what matters for this kernel's performance
    key=["C"],
)
@triton.jit
def bias_add_sum_kernel(
    x_ptr,           # *f32, shape [B, C]
    bias_ptr,        # *f32, shape [C]
    out_ptr,         # *f32, shape [B]
    B,               # int32
    C,               # int32
    stride_x_batch,  # int32
    stride_x_channel,# int32
    stride_bias,     # int32
    stride_out_batch,# int32
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)   # batch index
    pid_blk = tl.program_id(axis=1) # channel block index

    block_start = pid_blk * BLOCK_C
    offs_c = block_start + tl.arange(0, BLOCK_C)

    # masks
    mask_valid_b = pid_b < B
    mask_c = offs_c < C
    mask_x = mask_c & mask_valid_b

    # base pointers
    x_batch_ptr = x_ptr + pid_b * stride_x_batch

    # load x and bias
    x_vals = tl.load(
        x_batch_ptr + offs_c * stride_x_channel,
        mask=mask_x,
        other=0.0,
    )
    bias_vals = tl.load(
        bias_ptr + offs_c * stride_bias,
        mask=mask_c,
        other=0.0,
    )

    vals = x_vals + bias_vals
    partial = tl.sum(vals, axis=0)

    # accumulate into output with atomic add
    tl.atomic_add(
        out_ptr + pid_b * stride_out_batch,
        partial,
        mask=mask_valid_b,
    )


def triton_bias_add_sum(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x:    [B, C, 1, 1, 1]  (CUDA)
    bias: [C, 1, 1, 1]     (CUDA)

    Returns:
        out: [B, 1, 1, 1] where
             out[b] = sum_c (x[b, c, 0, 0, 0] + bias[c, 0, 0, 0])
    """
    assert x.is_cuda and bias.is_cuda, "triton_bias_add_sum requires CUDA tensors"
    assert x.ndim == 5 and bias.ndim == 4, "Unexpected tensor ranks"
    B, C, D, H, W = x.shape
    assert D == 1 and H == 1 and W == 1, "triton_bias_add_sum expects D=H=W=1"
    assert bias.shape[0] == C, "Bias channels must match x channels"

    x2 = x.contiguous().view(B, C)
    bias2 = bias.contiguous().view(C)

    out = torch.zeros(B, device=x.device, dtype=x.dtype)

    stride_x_batch, stride_x_channel = x2.stride()
    stride_bias = bias2.stride()[0]
    stride_out_batch = out.stride()[0]

    BLOCK_C = 128  # fixed, power-of-2 block size (do not change)

    # Grid is fixed by earlier stages: 1D over batch, 1D over channel blocks
    grid = lambda meta: (
        max(1, B),
        triton.cdiv(C, meta["BLOCK_C"]),
    )

    bias_add_sum_kernel[grid](
        x2,
        bias2,
        out,
        B,
        C,
        stride_x_batch,
        stride_x_channel,
        stride_bias,
        stride_out_batch,
        BLOCK_C=BLOCK_C,
    )

    return out.view(B, 1, 1, 1)


class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.

    The bias-add + sum(dim=1) stage is implemented with a Triton kernel when possible.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim

    def forward(self, x):
        x = self.conv(x)
        x = x / self.divisor
        x = self.max_pool(x)
        x = self.global_avg_pool(x)

        # Use Triton kernel only in the specific case it is written for:
        #   - CUDA tensors
        #   - 5D tensor with D=H=W=1 after global average pooling
        #   - sum along channel dimension (dim=1)
        if (
            x.is_cuda
            and x.ndim == 5
            and x.shape[2:] == (1, 1, 1)
            and self.bias.is_cuda
            and self.sum_dim == 1
        ):
            x = triton_bias_add_sum(x, self.bias)
        else:
            x = x + self.bias
            x = torch.sum(x, dim=self.sum_dim)

        return x
