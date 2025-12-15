import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["W"],
)
@triton.jit
def ln_gelu_scale_kernel(
    x_ptr,          # *f16 / *bf16 / *f32, shape [M, W] contiguous
    gamma_ptr,      # *same as x, shape [W]
    beta_ptr,       # *same as x, shape [W]
    out_ptr,        # *same as x, shape [M, W] contiguous
    W,              # int32, last dimension size
    inv_W,          # f32, 1.0 / W
    eps,            # f32
    scaling_factor, # f32
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused LayerNorm (over last dim W) + GELU + scaling.

    - Operates on a 2D view [M, W] of the input/output, contiguous in the last dim.
    - Each program instance handles one row (size W).
    - Uses a single tl.store() for the final result (no intermediate stores).
    """
    pid = tl.program_id(axis=0)  # row index in [0, M)

    row_start = pid * W
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < W

    # Load input row (cast to f32 for numerically stable reductions)
    x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # LayerNorm statistics in a single pass:
    # mean = E[x], var = E[x^2] - mean^2
    x_sq = x_f32 * x_f32
    sum_x = tl.sum(x_f32, axis=0)
    sum_x_sq = tl.sum(x_sq, axis=0)
    mean = sum_x * inv_W
    var = sum_x_sq * inv_W - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize
    y = (x_f32 - mean) * inv_std

    # Affine transform: y = y * gamma + beta
    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = y * gamma + beta

    # GELU activation (tanh approximation via exp, all in f32)
    c0 = 0.7978845608028654  # sqrt(2/pi)
    y_sq = y * y
    y_cubed = y_sq * y
    t = c0 * (y + 0.044715 * y_cubed)
    exp_neg2t = tl.exp(-2.0 * t)
    tanh_t = (1.0 - exp_neg2t) / (1.0 + exp_neg2t)
    gelu = 0.5 * y * (1.0 + tanh_t)

    # Final scaling
    gelu = gelu * scaling_factor

    # Cast back to original dtype and store (single store, fully fused)
    out = gelu.to(x.dtype)
    tl.store(out_ptr + row_start + offs, out, mask=mask)


def fused_ln_gelu_scale(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    scaling_factor: float,
) -> torch.Tensor:
    """
    Fused LayerNorm (over last dimension) + GELU + scaling for 5D tensors.

    Input/Output layout: [N, C, D, H, W], contiguous.
    LayerNorm is applied over the last dimension W.
    """
    assert x.dim() == 5, "Input must be 5D [N, C, D, H, W]."
    N, C, D, H, W = x.shape

    assert gamma.numel() == W, f"gamma must have shape [{W}], got {gamma.numel()}."
    assert beta.numel() == W, f"beta must have shape [{W}], got {beta.numel()}."

    # Ensure contiguous layout; we specialize the kernel for this case.
    x_contig = x.contiguous()
    dtype = x_contig.dtype
    device = x_contig.device

    # Parameters are used as per-last-dimension affine terms; match x dtype for simplicity.
    gamma = gamma.to(device=device, dtype=dtype, non_blocking=True)
    beta = beta.to(device=device, dtype=dtype, non_blocking=True)

    # Flatten to [M, W] to simplify indexing and reduce integer arithmetic
    M = N * C * D * H
    x_2d = x_contig.view(M, W)
    out = torch.empty_like(x_contig)
    out_2d = out.view(M, W)

    # Choose BLOCK_SIZE as next power-of-two >= W (up to 1024)
    BLOCK_SIZE = 1 << (W - 1).bit_length()
    if BLOCK_SIZE > 1024:
        raise NotImplementedError(
            f"Last dimension W={W} is too large for this fused kernel (BLOCK_SIZE={BLOCK_SIZE} > 1024)."
        )

    # Grid: one program per row (N*C*D*H)
    grid = (M,)

    # Precompute 1/W in f32 for reductions
    inv_W = 1.0 / float(W)

    ln_gelu_scale_kernel[grid](
        x_2d,
        gamma,
        beta,
        out_2d,
        W,
        inv_W,
        eps,
        scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


class ModelNew(nn.Module):
    """
    ConvTranspose3d (PyTorch) + fused LayerNorm (over last dim W) + GELU + scaling in Triton.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias=True,
        eps=1e-5,
        scaling_factor=1.0,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        self.eps = eps
        self.scaling_factor = scaling_factor

        # LayerNorm-style affine parameters over the last dimension W'.
        # The harness is expected to choose shapes so that out_channels == W'
        # (last dimension of ConvTranspose3d output).
        self.ln_weight = nn.Parameter(torch.ones(out_channels))
        self.ln_bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, in_channels, D, H, W]
        x = self.conv_transpose(x)  # -> [N, out_channels, D', H', W']
        x = fused_ln_gelu_scale(
            x,
            self.ln_weight,
            self.ln_bias,
            self.eps,
            self.scaling_factor,
        )
        return x
