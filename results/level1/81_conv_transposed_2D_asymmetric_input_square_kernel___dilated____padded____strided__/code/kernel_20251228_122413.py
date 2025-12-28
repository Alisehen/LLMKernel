# <complete ModelNew code with optimized Triton kernels>
import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def conv_transpose2d_im2col_gemm_kernel(
    x_ptr,   # [N, C_in, H_in, W_in]
    w_ptr,   # [C_in, C_out, K, K]
    b_ptr,   # [C_out] (ignored if has_bias=False)
    y_ptr,   # [N, C_out, H_out, W_out]
    N,
    H_in,
    W_in,
    C_out,
    H_out,
    W_out,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    has_bias: tl.constexpr,
    C_IN: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """
    Implicit-im2col + GEMM implementation of ConvTranspose2d forward.

    Matrix view:
      - A: [M_total, K_total]  (implicit, built from x)
      - B: [K_total, C_out]    (implicit, built from w)
      - Y: [M_total, C_out]    (flatten of [N, C_out, H_out, W_out])

    where:
      M_total = N * H_out * W_out
      K_total = C_in * K * K
    """
    pid_m = tl.program_id(axis=0)  # over flattened output positions (M dimension)
    pid_n = tl.program_id(axis=1)  # over output channels (N dimension of GEMM)

    # Total rows in the implicit A / Y matrices
    HW_out = H_out * W_out
    M_total = N * HW_out

    # Tile indices along M and output channels
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    co_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M_total
    mask_co = co_offsets < C_out

    # Decode flattened output index m -> (n, ho, wo)
    n_m = m_offsets // HW_out
    rem = m_offsets - n_m * HW_out
    ho_m = rem // W_out
    wo_m = rem - ho_m * W_out

    # Prepare for broadcasting: [BM, 1]
    n_b = n_m[:, None]
    ho_b = ho_m[:, None]
    wo_b = wo_m[:, None]

    # Accumulator for Y tile: [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Total K dimension (C_in * K * K) for GEMM
    K_total = C_IN * KERNEL_SIZE * KERNEL_SIZE

    # Loop over K dimension in tiles of size BLOCK_K
    for k0 in range(0, K_total, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K_total

        # Decode k index -> (cin, ky, kx)
        cin = k_offsets // (KERNEL_SIZE * KERNEL_SIZE)
        remk = k_offsets - cin * (KERNEL_SIZE * KERNEL_SIZE)
        ky = remk // KERNEL_SIZE
        kx = remk - ky * KERNEL_SIZE

        # Broadcast K components: [1, BK] or [BK, 1] as needed
        cin_b = cin[None, :]       # [1, BK]
        ky_b = ky[None, :]         # [1, BK]
        kx_b = kx[None, :]         # [1, BK]

        # ---------- Build A tile (from input x) ----------
        # For each (m, k): compute contributing input index if valid, else 0
        # tmp_h = ho + padding - ky * dilation
        # tmp_w = wo + padding - kx * dilation
        tmp_h = ho_b + padding - ky_b * dilation
        tmp_w = wo_b + padding - kx_b * dilation

        h_in = tmp_h // stride
        w_in = tmp_w // stride

        # Valid positions must satisfy:
        #   tmp_h >= 0, tmp_w >= 0
        #   tmp_h == h_in * stride, tmp_w == w_in * stride
        #   0 <= h_in < H_in, 0 <= w_in < W_in
        valid_h = (tmp_h >= 0) & (tmp_h == h_in * stride) & (h_in < H_in)
        valid_w = (tmp_w >= 0) & (tmp_w == w_in * stride) & (w_in < W_in)
        valid = valid_h & valid_w

        # Compute input offsets: x[n, cin, h_in, w_in]
        # Shape of all broadcasted tensors: [BM, BK]
        n_x = n_b
        cin_x = cin_b
        h_in_x = h_in
        w_in_x = w_in

        x_offsets = (((n_x * C_IN + cin_x) * H_in + h_in_x) * W_in + w_in_x)

        mask_x = valid & mask_m[:, None] & mask_k[None, :]
        a_tile = tl.load(x_ptr + x_offsets, mask=mask_x, other=0.0)
        a_tile = a_tile.to(tl.float32)  # accumulate in fp32

        # ---------- Build B tile (from weights w) ----------
        # w layout: [C_in, C_out, K, K] contiguous
        # index: (((cin * C_out + co) * K + ky) * K + kx)
        cin_w = cin[:, None]   # [BK, 1]
        ky_w = ky[:, None]     # [BK, 1]
        kx_w = kx[:, None]     # [BK, 1]
        co_b = co_offsets[None, :]  # [1, BN]

        w_offsets = (((cin_w * C_out + co_b) * KERNEL_SIZE + ky_w) * KERNEL_SIZE + kx_w)

        mask_w = mask_k[:, None] & mask_co[None, :]
        b_tile = tl.load(w_ptr + w_offsets, mask=mask_w, other=0.0)
        b_tile = b_tile.to(tl.float32)

        # ---------- Matmul accumulate ----------
        # [BM, BK] @ [BK, BN] -> [BM, BN]
        acc += tl.dot(a_tile, b_tile, allow_tf32=True)

    # ---------- Add bias (if any) ----------
    if has_bias:
        b_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0)
        b_vals = b_vals.to(tl.float32)
        acc += b_vals[None, :]

    # ---------- Store results ----------
    # y[n, co, ho, wo]
    co_y = co_offsets[None, :]  # [1, BN]
    n_y = n_b                   # [BM, 1]
    ho_y = ho_b                 # [BM, 1]
    wo_y = wo_b                 # [BM, 1]

    y_offsets = (((n_y * C_out + co_y) * H_out + ho_y) * W_out + wo_y)
    mask_y = mask_m[:, None] & mask_co[None, :]

    tl.store(y_ptr + y_offsets, acc.to(OUT_DTYPE), mask=mask_y)


def triton_conv_transpose2d(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride: int,
                            padding: int,
                            dilation: int) -> torch.Tensor:
    """
    x:       [N, C_in, H_in, W_in]
    weight:  [C_in, C_out, K, K]  (ConvTranspose2d layout)
    bias:    [C_out] or None

    Computes ConvTranspose2d forward using an implicit-im2col + GEMM Triton kernel.
    """
    assert x.is_cuda and weight.is_cuda, "Triton conv_transpose2d only supports CUDA tensors"
    x = x.contiguous()
    weight = weight.contiguous()

    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out, K, K2 = weight.shape
    assert C_in_w == C_in and K == K2, "Weight shape must be [C_in, C_out, K, K]"

    # Output size formula for ConvTranspose2d (output_padding=0)
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Bias handling
    if bias is not None:
        bias = bias.contiguous()
        has_bias = True
        b_ptr = bias
    else:
        has_bias = False
        # Dummy pointer; never accessed when has_bias=False
        b_ptr = weight

    # Dtype handling: support fp16 and fp32
    if x.dtype == torch.float16:
        out_dtype = tl.float16
    elif x.dtype == torch.float32:
        out_dtype = tl.float32
    else:
        raise TypeError(f"Unsupported dtype {x.dtype}; only float16/float32 are supported.")

    # Tiling parameters (all powers of two)
    BLOCK_M = 64  # rows (N * H_out * W_out)
    BLOCK_N = 64  # output channels
    BLOCK_K = 32  # C_in * K * K

    grid = lambda meta: (
        triton.cdiv(N * H_out * W_out, meta["BLOCK_M"]),
        triton.cdiv(C_out, meta["BLOCK_N"]),
    )

    conv_transpose2d_im2col_gemm_kernel[grid](
        x, weight, b_ptr, y,
        N, H_in, W_in,
        C_out, H_out, W_out,
        stride=stride,
        padding=padding,
        dilation=dilation,
        has_bias=has_bias,
        C_IN=C_in,
        KERNEL_SIZE=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        OUT_DTYPE=out_dtype,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for nn.ConvTranspose2d with square kernel,
    supporting stride, padding, dilation, and optional bias.

    Behavior matches the PyTorch nn.ConvTranspose2d module used in Model.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = False):
        super().__init__()
        # Keep a real ConvTranspose2d module so external code can copy weights/bias into it.
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        return triton_conv_transpose2d(
            x, w, b, self.stride, self.padding, self.dilation
        )
