import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_C": 512}, num_warps=8, num_stages=2),
    ],
    key=["C"],
)
@triton.jit
def softmax_sigmoid_5d_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    # meta-parameter
    BLOCK_C: tl.constexpr,
):
    """
    Fused Softmax (dim=1 over channels) + Sigmoid for 5D tensor [N, C, D, H, W].

    Grid:
      - 1D grid over all (n, d, h, w) sites: pid in [0, N*D*H*W)
      - For each site, reduction over C is done inside the program

    Fusion rule (elementwise part: softmax -> sigmoid):
      - All tl.load/tl.store for the fused elementwise path share:
          - same `offsets` (channel-wise)
          - same `mask` (channel boundary)
    """
    pid = tl.program_id(0)

    # Decode linear pid -> (n, d, h, w)
    spatial = D * H * W
    hw = H * W

    n = pid // spatial
    tmp = pid - n * spatial
    d = tmp // hw
    tmp = tmp - d * hw
    h = tmp // W
    w = tmp - h * W

    # Base pointer offset for fixed (n, :, d, h, w)
    base_offset = (
        n * stride_n +
        d * stride_d +
        h * stride_h +
        w * stride_w
    )

    # Channel indices handled by this program
    offs_c = tl.arange(0, BLOCK_C)

    # -----------------------------
    # Pass 1: compute max over C
    # -----------------------------
    max_val = tl.full((), -float("inf"), dtype=tl.float32)

    c_start = 0
    while c_start < C:
        idx_c = c_start + offs_c
        mask = idx_c < C

        # Shared offsets & mask for all fused ops in this loop iteration
        offsets = base_offset + idx_c * stride_c

        x_vals = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
        x_vals = x_vals.to(tl.float32)

        curr_max = tl.max(x_vals, axis=0)
        max_val = tl.maximum(max_val, curr_max)

        c_start += BLOCK_C

    # -----------------------------
    # Pass 2: compute sum(exp(x - max))
    # -----------------------------
    sum_exp = tl.zeros((), dtype=tl.float32)

    c_start = 0
    while c_start < C:
        idx_c = c_start + offs_c
        mask = idx_c < C

        offsets = base_offset + idx_c * stride_c

        x_vals = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
        x_vals = x_vals.to(tl.float32)

        exp_vals = tl.exp(x_vals - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)

        c_start += BLOCK_C

    inv_sum_exp = 1.0 / sum_exp

    # -----------------------------
    # Pass 3: write sigmoid(softmax(x))
    # -----------------------------
    c_start = 0
    while c_start < C:
        idx_c = c_start + offs_c
        mask = idx_c < C

        offsets = base_offset + idx_c * stride_c

        x_vals = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
        x_vals = x_vals.to(tl.float32)

        # Softmax along C (already have max_val and inv_sum_exp)
        exp_vals = tl.exp(x_vals - max_val)
        softmax_vals = exp_vals * inv_sum_exp

        # Sigmoid(softmax(x)) (elementwise, same offsets/mask)
        sigmoid_vals = 1.0 / (1.0 + tl.exp(-softmax_vals))

        tl.store(out_ptr + offsets, sigmoid_vals, mask=mask)

        c_start += BLOCK_C


def fused_softmax_sigmoid_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Fused Softmax(dim=1) + Sigmoid for a 5D tensor [N, C, D, H, W] using Triton.

    Args:
        x: Input tensor, expected shape [N, C, D, H, W], CUDA tensor.

    Returns:
        Tensor of same shape as x.
    """
    assert x.is_cuda, "Input must be a CUDA tensor"
    assert x.dim() == 5, "Input must have shape [N, C, D, H, W]"

    N, C, D, H, W = x.shape
    y = torch.empty_like(x)

    total_sites = N * D * H * W
    grid = (total_sites,)

    softmax_sigmoid_5d_kernel[grid](
        x, y,
        N, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
    )
    return y


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch native) + fused Softmax(dim=1) + Sigmoid (Triton).
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        # Keep ConvTranspose3d as native PyTorch (indexing is complex)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [N, C_in, D, H, W]
        x = self.conv_transpose(x)
        # x shape: [N, C_out, D, H, W]
        x = fused_softmax_sigmoid_3d(x)
        return x
