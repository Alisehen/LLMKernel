import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_P": 64, "BLOCK_C": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_P": 128, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_P": 256, "BLOCK_C": 64}, num_warps=8, num_stages=2),
    ],
    key=["P", "C"],
)
@triton.jit
def channel_min_tanh2_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    P,
    BLOCK_P: tl.constexpr,  # number of (n, h, w) pixels per program
    BLOCK_C: tl.constexpr,  # number of channels processed per iteration
):
    """
    For input x of shape [N, C, H, W], compute:
        y[n, 0, h, w] = tanh(tanh(min_c x[n, c, h, w]))

    Grid & indexing:
      - 1D grid over the flattened (n, h, w) output positions
      - All fused ops (min + 2x tanh + store) share the SAME
        offsets (offs_p -> n,h,w) and mask (mask_p).
    """
    pid = tl.program_id(0)

    # Linear indices over the (n, h, w) positions
    offs_p = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = offs_p < P  # shared mask for all fused ops

    HW = H * W
    # Map linear index -> (n, h, w)
    n = offs_p // HW
    rem = offs_p % HW
    h = rem // W
    w = rem % W

    # Running minimum over channels for each (n, h, w)
    inf = 1e30
    min_val = tl.full((BLOCK_P,), inf, dtype=tl.float32)

    c_start = 0
    while c_start < C:
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        mask_c = c_offsets < C

        # Pointers to x[n, c, h, w] for this block of channels
        x_ptrs = (
            x_ptr
            + n[:, None] * stride_x_n
            + c_offsets[None, :] * stride_x_c
            + h[:, None] * stride_x_h
            + w[:, None] * stride_x_w
        )

        # Load a BLOCK_P x BLOCK_C tile; masked elements get +inf
        vals = tl.load(
            x_ptrs,
            mask=mask_p[:, None] & mask_c[None, :],
            other=inf,
        )
        # Accumulate reduction in fp32 for good numeric behavior
        vals_f32 = vals.to(tl.float32)

        # Per-(n,h,w) minimum over this channel block
        block_min = tl.min(vals_f32, axis=1)
        # Update running minimum
        min_val = tl.minimum(min_val, block_min)

        c_start += BLOCK_C

    # --- Fused elementwise: tanh(tanh(min_val)) ---
    # All ops use same offsets (offs_p -> n,h,w) and mask (mask_p)

    # First tanh
    x1 = min_val
    e2x1 = tl.exp(2.0 * x1)
    tanh1 = (e2x1 - 1.0) / (e2x1 + 1.0)

    # Second tanh
    e2x2 = tl.exp(2.0 * tanh1)
    tanh2 = (e2x2 - 1.0) / (e2x2 + 1.0)

    # Store result to output: shape [N, 1, H, W]
    out_ptrs = (
        out_ptr
        + n * stride_out_n
        + h * stride_out_h
        + w * stride_out_w
    )
    tl.store(out_ptrs, tanh2, mask=mask_p)


def conv_min_tanh2(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fast implementation of:
        y = conv2d(x, weight, bias, stride=1, padding=0)
        y = torch.min(y, dim=1, keepdim=True)[0]
        y = torch.tanh(y)
        y = torch.tanh(y)

    Convolution is delegated to PyTorch's backend (cuDNN / cuBLAS).
    Channel-wise min and double tanh are fused into a single Triton kernel.
    """
    assert x.dim() == 4, "Input must be NCHW"
    N, C_in, H, W = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w, "Input channels must match weight"

    # Stride=1, padding=0, dilation=1, groups=1
    y = torch.nn.functional.conv2d(
        x, weight, bias, stride=1, padding=0, dilation=1, groups=1
    )
    N_y, C_y, H_out, W_out = y.shape
    assert N_y == N and C_y == C_out, "Unexpected conv output shape"

    # Make conv output contiguous to get predictable strides
    y_ = y.contiguous()
    out = torch.empty((N, 1, H_out, W_out), device=x.device, dtype=x.dtype)

    stride_y_n, stride_y_c, stride_y_h, stride_y_w = y_.stride()
    stride_out_n, stride_out_c, stride_out_h, stride_out_w = out.stride()

    P = N * H_out * W_out  # number of (n, h, w) positions

    def grid(meta):
        # 1D grid over flattened (n,h,w); always > 0
        return (triton.cdiv(P, meta["BLOCK_P"]),)

    channel_min_tanh2_kernel[grid](
        y_, out,
        N, C_out, H_out, W_out,
        stride_y_n, stride_y_c, stride_y_h, stride_y_w,
        stride_out_n, stride_out_c, stride_out_h, stride_out_w,
        P,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated version of:
        x = conv2d(x)
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = torch.tanh(x)
        x = torch.tanh(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # Keep the same structure/state_dict layout as the original model
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return conv_min_tanh2(x, self.conv.weight, self.conv.bias)
