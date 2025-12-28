import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv_transpose2d_gemm_kernel(
    x_ptr,          # float32[N, Cin, H_in, W_in]
    w_ptr,          # float32[Cout, K] (Cout x (Cin*kH*kW))
    b_ptr,          # float32[Cout] or dummy
    out_ptr,        # float32[N, Cout, H_out, W_out]
    N, Cin, Cout,
    H_in, W_in,
    H_out, W_out,
    kH, kW,
    stride_h, stride_w,
    pad_h, pad_w,
    K,              # Cin * kH * kW
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,   # tile size in Cout
    BLOCK_N: tl.constexpr,   # tile size in N*H_out*W_out
    BLOCK_K: tl.constexpr,   # tile size in Cin*kH*kW
):
    # Program IDs along M (Cout) and N (N * H_out * W_out)
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets in Cout dimension
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < Cout

    # Offsets in flattened (N * H_out * W_out) dimension
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    N_total = N * H_out * W_out
    mask_n = n_offsets < N_total

    # Decode n_offsets -> (n_batch, oh, ow)
    HW_out = H_out * W_out
    n_batch = n_offsets // HW_out
    hw = n_offsets % HW_out
    oh = hw // W_out
    ow = hw % W_out

    # Strides for input and output tensors
    in_batch_stride = Cin * H_in * W_in
    cin_stride = H_in * W_in
    out_batch_stride = Cout * H_out * W_out
    out_c_stride = H_out * W_out

    # Accumulator for [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    kHkW = kH * kW

    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Decode k_offsets -> (ci, kh, kw) to match w_ptr layout [Cout, Cin, kH, kW]
        ci = k_offsets // kHkW
        rem = k_offsets % kHkW
        kh = rem // kW
        kw = rem % kW

        # Compute input coordinates hi, wi for all (k, n) in tile (implicit im2col)
        # h_in_numer = oh + pad_h - kh
        h_in_numer = oh[None, :] + pad_h - kh[:, None]
        h_ge_zero = h_in_numer >= 0
        hi = h_in_numer // stride_h
        h_in_range = hi < H_in
        h_divisible = (h_in_numer % stride_h) == 0
        h_mask = h_ge_zero & h_in_range & h_divisible

        # w_in_numer = ow + pad_w - kw
        w_in_numer = ow[None, :] + pad_w - kw[:, None]
        w_ge_zero = w_in_numer >= 0
        wi = w_in_numer // stride_w
        w_in_range = wi < W_in
        w_divisible = (w_in_numer % stride_w) == 0

        # Valid input positions mask
        mask_x = (
            mask_k[:, None]
            & mask_n[None, :]
            & h_mask
            & w_ge_zero
            & w_in_range
            & w_divisible
        )

        # Compute input indices: x[n_batch, ci, hi, wi]
        x_indices = (
            n_batch[None, :] * in_batch_stride
            + ci[:, None] * cin_stride
            + hi * W_in
            + wi
        )

        # Load input tile [BLOCK_K, BLOCK_N]
        x_tile = tl.load(x_ptr + x_indices, mask=mask_x, other=0.0)
        x_tile = x_tile.to(tl.float32)

        # Load weight tile [BLOCK_M, BLOCK_K] from w_ptr [Cout, K]
        w_indices = m_offsets[:, None] * K + k_offsets[None, :]
        w_mask = mask_m[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptr + w_indices, mask=w_mask, other=0.0)
        w_tile = w_tile.to(tl.float32)

        # GEMM: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(w_tile, x_tile, allow_tf32=True)

        k_start += BLOCK_K

    # Add bias
    if HAS_BIAS:
        b_vals = tl.load(b_ptr + m_offsets, mask=mask_m, other=0.0)  # [BLOCK_M]
        acc += b_vals[:, None]

    # Store results to out[n_batch, m_offsets, oh, ow]
    out_indices = (
        n_batch[None, :] * out_batch_stride
        + m_offsets[:, None] * out_c_stride
        + oh[None, :] * W_out
        + ow[None, :]
    )
    out_mask = mask_m[:, None] & mask_n[None, :]

    tl.store(out_ptr + out_indices, acc, mask=out_mask)


def triton_conv_transpose2d(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride: tuple,
                            padding: tuple) -> torch.Tensor:
    """
    x:       [N, Cin, H_in, W_in]
    weight:  [Cin, Cout, kH, kW] (PyTorch ConvTranspose2d layout)
    bias:    [Cout] or None
    stride:  (stride_h, stride_w)
    padding: (pad_h, pad_w)
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton kernel."

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, Cin, H_in, W_in = x.shape
    Cin_w, Cout, kH, kW = weight.shape
    assert Cin_w == Cin, "Inconsistent in_channels between input and weight."

    stride_h, stride_w = int(stride[0]), int(stride[1])
    pad_h, pad_w = int(padding[0]), int(padding[1])

    # Standard ConvTranspose2d output shape (dilation=1, output_padding=0)
    H_out = (H_in - 1) * stride_h - 2 * pad_h + kH
    W_out = (W_in - 1) * stride_w - 2 * pad_w + kW

    # Flatten weights into GEMM layout: [Cout, Cin*kH*kW]
    # Original: [Cin, Cout, kH, kW]
    # Permute to [Cout, Cin, kH, kW] then flatten
    w_mat = weight.permute(1, 0, 2, 3).contiguous().view(Cout, Cin * kH * kW)

    out = torch.empty((N, Cout, H_out, W_out), device=x.device, dtype=x.dtype)

    K = Cin * kH * kW
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32

    has_bias = bias is not None
    b_ptr = bias if has_bias else out.new_empty(1)

    grid = (
        triton.cdiv(Cout, BLOCK_M),
        triton.cdiv(N * H_out * W_out, BLOCK_N),
    )

    conv_transpose2d_gemm_kernel[grid](
        x, w_mat, b_ptr, out,
        N, Cin, Cout,
        H_in, W_in,
        H_out, W_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w,
        K,
        HAS_BIAS=has_bias,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return out


class ModelNew(nn.Module):
    """
    Triton-accelerated 2D transposed convolution, API-compatible with the given Model.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple = (1, 1),
                 padding: tuple = (0, 0),
                 bias: bool = False):
        super().__init__()
        # Keep a standard ConvTranspose2d module to own parameters/state_dict
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            # Fallback to PyTorch implementation on CPU
            return self.conv_transpose2d(x)

        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        stride = self.conv_transpose2d.stride
        padding = self.conv_transpose2d.padding

        return triton_conv_transpose2d(x, w, b, stride, padding)
