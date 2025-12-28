import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_transpose1d_fwd_gemm_kernel(
    x_ptr,            # *f32  [N, Cin, L_in]
    w_ptr,            # *f32  [Cin, Cout, K]
    b_ptr,            # *f32  [Cout] (ignored if HAS_BIAS=False)
    y_ptr,            # *f32  [N, Cout, L_out]
    N,                # int32
    L_in,             # int32
    L_out,            # int32
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    CIN: tl.constexpr,
    COUT: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile in Cout
    BLOCK_N: tl.constexpr,   # tile in L_out
    BLOCK_P: tl.constexpr,   # tile in Cin*K
):
    """
    GEMM-style implementation of ConvTranspose1d forward:

      y[n, co, l_out] = sum_{ci, k} x[n, ci, l_in] * w[ci, co, k]
      with relation:   l_out = l_in * stride - padding + k * dilation

    We reinterpret this as:
      For a block of output positions (l_out tile),
      build an implicit 'im2col' matrix of shape (Cin*K, BLOCK_N) from x,
      and multiply with W^T of shape (Cout, Cin*K).
    """

    # Program IDs:
    #  axis 0: over (batch, L_out tiles)
    #  axis 1: over Cout tiles
    pid_0 = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    # Number of tiles along L_out
    num_l_tiles = tl.cdiv(L_out, BLOCK_N)

    # Decode batch index and L_out tile index from pid_0
    n = pid_0 // num_l_tiles
    tile_l = pid_0 % num_l_tiles

    # Offsets in Cout (M dimension) and L_out (N dimension)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tile_l * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < COUT
    mask_n = n_offsets < L_out
    n_valid = n < N

    # Initialize accumulator for Y tile: [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Add bias if requested: broadcast bias across L_out tile
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)  # [BLOCK_M]
        acc += bias_vals[:, None]

    # Reduction dimension P = Cin * K
    # We iterate in chunks of BLOCK_P to build W-panel and X-panel and do a GEMM
    for p_start in range(0, CIN * K, BLOCK_P):
        p_offsets = p_start + tl.arange(0, BLOCK_P)        # [BLOCK_P]
        mask_p = p_offsets < (CIN * K)

        # Map flatten index p -> (ci, k) with:
        #   p = ci * K + k
        ci_idx = p_offsets // K                            # [BLOCK_P]
        k_idx = p_offsets % K                              # [BLOCK_P]

        # ----- Build X panel (implicit im2col) -----
        # For each row p (i.e., (ci, k)) and each l_out in the tile,
        # compute the contributing l_in and load x[n, ci, l_in] if valid.

        # shapes:
        #   l_out_mat: [1, BLOCK_N]
        #   k_mat:     [BLOCK_P, 1]
        l_out_mat = n_offsets[None, :]                     # [1, BLOCK_N]
        k_mat = k_idx[:, None]                             # [BLOCK_P, 1]

        # raw = l_out + padding - k * dilation
        raw = l_out_mat + padding - k_mat * dilation       # [BLOCK_P, BLOCK_N]

        # l_in = raw / stride (integer division)
        l_in = raw // stride

        # Validity checks: raw >= 0, raw < stride * L_in, and divisible by stride
        is_nonneg = raw >= 0
        is_lt_max = raw < stride * L_in
        is_div = raw == l_in * stride
        valid = is_nonneg & is_lt_max & is_div

        # Also apply masks for p, n, and batch index
        valid = valid & mask_p[:, None] & mask_n[None, :] & n_valid

        # Compute x indices:
        #   base for each (n, ci): (n * CIN + ci) * L_in
        base_ci = n * CIN + ci_idx                          # [BLOCK_P]
        base_x = base_ci * L_in                             # [BLOCK_P]
        x_idx = base_x[:, None] + l_in                      # [BLOCK_P, BLOCK_N]

        x_panel = tl.load(x_ptr + x_idx, mask=valid, other=0.0)  # [BLOCK_P, BLOCK_N]

        # ----- Build W panel -----
        # W has shape [Cin, Cout, K] in memory with layout:
        #   index = (ci * COUT + co) * K + k
        # Our A_panel is W^T chunk of shape [BLOCK_M, BLOCK_P]:
        #   A[m, p] = w[ci_idx[p], m_offsets[m], k_idx[p]]

        ci_bcast = ci_idx[None, :]                         # [1, BLOCK_P]
        k_bcast = k_idx[None, :]                           # [1, BLOCK_P]
        m_bcast = m_offsets[:, None]                       # [BLOCK_M, 1]

        w_index = (ci_bcast * COUT + m_bcast) * K + k_bcast   # [BLOCK_M, BLOCK_P]
        mask_wp = mask_m[:, None] & mask_p[None, :]

        w_panel = tl.load(w_ptr + w_index, mask=mask_wp, other=0.0)  # [BLOCK_M, BLOCK_P]

        # ----- GEMM update -----
        # acc += W_panel @ X_panel  ( (BLOCK_M, BLOCK_P) x (BLOCK_P, BLOCK_N) )
        acc += tl.dot(w_panel, x_panel, allow_tf32=True)

    # ----- Store Y tile -----
    y_index = (n * COUT + m_offsets[:, None]) * L_out + n_offsets[None, :]
    mask_y = mask_m[:, None] & mask_n[None, :] & n_valid
    tl.store(y_ptr + y_index, acc, mask=mask_y)


def triton_conv_transpose1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> torch.Tensor:
    """
    High-performance ConvTranspose1d forward using a GEMM-like Triton kernel.

    Args:
        x:       (N, Cin, L_in)  float32, CUDA
        weight:  (Cin, Cout, K)  float32, CUDA
        bias:    (Cout,) or None
        stride, padding, dilation: as in nn.ConvTranspose1d (no output_padding)

    Returns:
        y: (N, Cout, L_out) with
           L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == weight.dtype == torch.float32, "Only float32 is supported"

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, Cin, L_in = x.shape
    w_Cin, Cout, K = weight.shape
    assert w_Cin == Cin, "Weight Cin must match input Cin"

    # PyTorch ConvTranspose1d output length formula (no output_padding)
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    y = torch.empty((N, Cout, L_out), device=x.device, dtype=x.dtype)

    # Tiling parameters (all powers of 2)
    BLOCK_M = 64   # Cout tile
    BLOCK_N = 64   # L_out tile
    BLOCK_P = 32   # Cin*K tile

    grid = lambda meta: (
        N * triton.cdiv(L_out, meta["BLOCK_N"]),   # over (n, L_out tiles)
        triton.cdiv(Cout, meta["BLOCK_M"]),        # over Cout tiles
    )

    conv_transpose1d_fwd_gemm_kernel[grid](
        x,
        weight,
        bias if bias is not None else y,  # dummy if no bias
        y,
        N,
        L_in,
        L_out,
        stride=stride,
        padding=padding,
        dilation=dilation,
        CIN=Cin,
        COUT=Cout,
        K=K,
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_P=BLOCK_P,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Transposed 1D convolution implemented with a high-performance Triton kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to False.
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

        # Initialize parameters using PyTorch's ConvTranspose1d initialization
        ref = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.weight = nn.Parameter(ref.weight.detach())
        if bias:
            self.bias = nn.Parameter(ref.bias.detach())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_conv_transpose1d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
