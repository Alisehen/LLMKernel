# optimized Triton code
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline tile (current choice)
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # More rows in output-pixel dimension
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
        # More columns in Cout dimension
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["Cout", "H_out", "W_out", "KDIM"],
)
@triton.jit
def conv_transpose2d_implicit_gemm_kernel(
    x_ptr,         # float32 [N, Cin, H_in, W_in]
    w_flat_ptr,    # float32 [KDIM, Cout]  (row-major: KDIM x Cout)
    b_ptr,         # float32 [Cout] (unused if HAS_BIAS=False)
    y_ptr,         # float32 [N, Cout, H_out, W_out]
    N, Cin, H_in, W_in,
    H_out, W_out,
    Cout,
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    KDIM: tl.constexpr,          # KDIM = Cin * KERNEL_SIZE * KERNEL_SIZE
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,       # tile size along output-pixel (M) dimension
    BLOCK_N: tl.constexpr,       # tile size along Cout (N) dimension
    BLOCK_K: tl.constexpr,       # tile size along reduction (K) dimension
):
    # Flattened output pixels: M = N * H_out * W_out
    M = N * H_out * W_out

    # Program ids for 2D tiling of [M, Cout]
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < Cout

    # Map flattened M index -> (n, ho, wo)
    HW_out = H_out * W_out
    n = offs_m // HW_out
    rem = offs_m - n * HW_out
    ho = rem // W_out
    wo = rem - ho * W_out

    # Broadcasted versions used inside the K loop (avoid recomputing each iteration)
    n_b = n[:, None]
    ho_b = ho[:, None]
    wo_b = wo[:, None]

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Add bias if present: broadcast over rows
    if HAS_BIAS:
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias_vals[None, :]

    # Reduction over KDIM = Cin * KERNEL_SIZE * KERNEL_SIZE
    kk = tl.arange(0, BLOCK_K)
    KK = KERNEL_SIZE * KERNEL_SIZE  # constexpr

    for k0 in range(0, KDIM, BLOCK_K):
        k_indices = k0 + kk  # [BLOCK_K]
        mask_k = k_indices < KDIM

        # Decode flattened k_indices -> (ic, kh, kw)
        ic = k_indices // KK
        rem_k = k_indices - ic * KK
        kh = rem_k // KERNEL_SIZE
        kw = rem_k - kh * KERNEL_SIZE

        ic_b = ic[None, :]
        kh_b = kh[None, :]
        kw_b = kw[None, :]

        # Compute input h index
        h_in_numer = ho_b + PADDING - kh_b * DILATION
        h_in = h_in_numer // STRIDE
        h_in_times_stride = h_in * STRIDE
        valid_h = (h_in_numer == h_in_times_stride) & (h_in >= 0) & (h_in < H_in)

        # Compute input w index
        w_in_numer = wo_b + PADDING - kw_b * DILATION
        w_in = w_in_numer // STRIDE
        w_in_times_stride = w_in * STRIDE
        valid_w = (w_in_numer == w_in_times_stride) & (w_in >= 0) & (w_in < W_in)

        valid_mk = (
            mask_m[:, None]
            & mask_k[None, :]
            & valid_h
            & valid_w
        )

        # Safe indices (only used where valid_mk is True)
        h_in_safe = tl.where(valid_h, h_in, 0)
        w_in_safe = tl.where(valid_w, w_in, 0)

        # Compute input offsets: ((n * Cin + ic) * H_in + h_in) * W_in + w_in
        base_nc = n_b * Cin + ic_b
        base_nch = base_nc * H_in + h_in_safe
        x_offsets = base_nch * W_in + w_in_safe

        # Load A tile = X_col[M, K] implicitly
        a_tile = tl.load(x_ptr + x_offsets, mask=valid_mk, other=0.0)

        # Load B tile = W_flat[K, Cout]
        # Strides: row-major [KDIM, Cout] => stride_k = Cout, stride_n = 1
        b_tile = tl.load(
            w_flat_ptr + k_indices[:, None] * Cout + offs_n[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )

        # Multiply-accumulate: [M, K] @ [K, N] -> [M, N]
        acc += tl.dot(a_tile, b_tile, allow_tf32=True)

    # Store results back to y[N, Cout, H_out, W_out]
    co_b = offs_n[None, :]
    y_offsets = (((n_b * Cout + co_b) * H_out) + ho_b) * W_out + wo_b
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptr + y_offsets, acc, mask=mask_out)


def triton_conv_transpose2d(x: torch.Tensor,
                            weight: torch.Tensor,
                            bias: torch.Tensor,
                            stride: int,
                            padding: int,
                            dilation: int) -> torch.Tensor:
    # Ensure CUDA tensors and contiguous layout
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    x_f32 = x.contiguous().to(torch.float32)
    w_f32 = weight.contiguous().to(torch.float32)

    if bias is not None:
        b_f32 = bias.contiguous().to(torch.float32)
        has_bias = True
    else:
        # Dummy tensor (never read when HAS_BIAS=False)
        b_f32 = torch.empty(1, device=x.device, dtype=torch.float32)
        has_bias = False

    N, Cin, H_in, W_in = x_f32.shape
    Cin_w, Cout, K_h, K_w = w_f32.shape
    assert Cin_w == Cin, "Weight Cin must match input Cin"
    assert K_h == K_w, "Kernel must be square"
    K = K_h

    stride_int = int(stride)
    padding_int = int(padding)
    dilation_int = int(dilation)

    # Output size formula (no output_padding)
    H_out = (H_in - 1) * stride_int - 2 * padding_int + dilation_int * (K - 1) + 1
    W_out = (W_in - 1) * stride_int - 2 * padding_int + dilation_int * (K - 1) + 1

    y = torch.empty((N, Cout, H_out, W_out), device=x.device, dtype=torch.float32)

    # Flatten weights to [KDIM, Cout], row-major
    # w_f32: [Cin, Cout, K, K] -> [Cin, K, K, Cout] -> [Cin*K*K, Cout]
    w_flat = w_f32.permute(0, 2, 3, 1).reshape(Cin * K * K, Cout).contiguous()

    KDIM = Cin * K * K
    M = N * H_out * W_out

    # Grid over [M, Cout]; BLOCK_* are autotuned meta-parameters
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(Cout, META["BLOCK_N"]),
    )

    conv_transpose2d_implicit_gemm_kernel[grid](
        x_f32, w_flat, b_f32, y,
        N, Cin, H_in, W_in,
        H_out, W_out,
        Cout,
        STRIDE=stride_int,
        PADDING=padding_int,
        DILATION=dilation_int,
        KERNEL_SIZE=K,
        KDIM=KDIM,
        HAS_BIAS=has_bias,
    )

    return y.to(dtype=x.dtype)


class ModelNew(nn.Module):
    """
    Triton-optimized version of ConvTranspose2d forward using an implicit-GEMM formulation.
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
        # Keep an nn.ConvTranspose2d for parameter storage / state_dict compatibility
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose2d.weight
        b = self.conv_transpose2d.bias
        stride = self.conv_transpose2d.stride[0]
        padding = self.conv_transpose2d.padding[0]
        dilation = self.conv_transpose2d.dilation[0]
        return triton_conv_transpose2d(x, w, b, stride, padding, dilation)
