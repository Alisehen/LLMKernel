import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_valid_kernel(
    x_ptr,          # float32[N, C_in, H_pad, W_pad]
    w_ptr,          # float32[C_in, C_out, K_h, K_w] (spatially flipped)
    y_ptr,          # float32[N, C_out, H_out, W_out]
    N, C_in,
    C_out, K_h, K_w,
    x_sN, x_sC, x_sH, x_sW,
    w_sIC, w_sOC, w_sKH, w_sKW,
    y_sN, y_sC, y_sH, y_sW,
    H_out, W_out,
    M_total,        # kept for interface compatibility; not used in indexing
    K_total,        # = C_in * K_h * K_w
    BLOCK_M: tl.constexpr,  # tile in output spatial dimension (per batch)
    BLOCK_N: tl.constexpr,  # tile in C_out dimension
    BLOCK_K: tl.constexpr,  # reduction tile (C_in * K_h * K_w)
):
    # 3D launch grid:
    #   pid_b : batch dimension
    #   pid_m : tiles over H_out * W_out inside each batch
    #   pid_n : tiles over C_out
    pid_b = tl.program_id(axis=0)  # batch index n
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    # Per-batch output spatial size
    hw_out = H_out * W_out

    # Offsets in flattened spatial dimension (per batch) and C_out dimension
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Bounds masks for tail tiles
    mask_m = offs_m < hw_out
    mask_n = offs_n < C_out

    # Decode spatial index into (oh, ow)
    oh_idx = offs_m // W_out
    ow_idx = offs_m % W_out

    # Broadcasted versions for later pointer arithmetic
    oh_bc = oh_idx[:, None]        # (BM, 1)
    ow_bc = ow_idx[:, None]        # (BM, 1)
    oc_bc = offs_n[None, :]        # (1, BN)

    # Base pointers for this batch
    x_ptr_n = x_ptr + pid_b * x_sN
    y_ptr_n = y_ptr + pid_b * y_sN

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K = C_in * K_h * K_w
    k = 0
    khkw_total = K_h * K_w
    while k < K_total:
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K_total

        # Decode K index into (ic, kh, kw)
        ic_idx = k_offsets // khkw_total
        rem_k = k_offsets % khkw_total
        kh_idx = rem_k // K_w
        kw_idx = rem_k % K_w

        ic_bc = ic_idx[None, :]     # (1, BK)
        kh_bc = kh_idx[None, :]     # (1, BK)
        kw_bc = kw_idx[None, :]     # (1, BK)

        # For standard valid conv on padded input:
        # ih_pad = oh + kh, iw_pad = ow + kw
        ih_pad = oh_bc + kh_bc      # (BM, BK)
        iw_pad = ow_bc + kw_bc      # (BM, BK)

        # Masks for tail in M and K dimensions
        mask_x = mask_m[:, None] & mask_k[None, :]
        mask_w = mask_k[:, None] & mask_n[None, :]

        # Build pointers for A = implicit im2col(x_pad)
        x_ptrs = (
            x_ptr_n
            + ic_bc * x_sC
            + ih_pad * x_sH
            + iw_pad * x_sW
        )
        a = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Build pointers for B = reshaped, flipped weights w_flip[ic, oc, kh, kw]
        w_ptrs = (
            w_ptr
            + ic_bc.T * w_sIC
            + oc_bc * w_sOC
            + kh_bc.T * w_sKH
            + kw_bc.T * w_sKW
        )
        b = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # Matrix multiply-accumulate
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)

        k += BLOCK_K

    # Store results to y[n, oc, oh, ow]
    oh_out_bc = oh_idx[:, None]
    ow_out_bc = ow_idx[:, None]

    y_ptrs = (
        y_ptr_n
        + oc_bc * y_sC
        + oh_out_bc * y_sH
        + ow_out_bc * y_sW
    )
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_out)


def triton_conv_transpose2d_stride1_pad0(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    High-performance ConvTranspose2d for:
      - stride = (1, 1)
      - padding = (0, 0)
      - dilation = (1, 1)
      - groups = 1
      - bias handled outside (if needed)

    Implements ConvTranspose2d as a standard Conv2d on a padded input with
    spatially flipped weights:

      y = conv_transpose2d(x, W)
        == conv2d(pad(x, kH-1, kW-1), flip(W, H, W))

    Shapes:
      x:       [N, C_in, H_in, W_in]
      weight:  [C_in, C_out, K_h, K_w]
      output:  [N, C_out, H_out, W_out]
      with H_out = H_in + K_h - 1, W_out = W_in + K_w - 1
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == weight.dtype == torch.float32, "Kernel currently assumes float32"
    assert x.ndim == 4 and weight.ndim == 4

    N, C_in, H_in, W_in = x.shape
    C_in_w, C_out, K_h, K_w = weight.shape
    assert C_in == C_in_w, "Input channels mismatch between x and weight"

    # Pad input by (K_h-1, K_w-1) on all sides: x_pad[N, C_in, H_in+2*(K_h-1), W_in+2*(K_w-1)]
    pad_h = K_h - 1
    pad_w = K_w - 1
    x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
    x_pad = x_pad.contiguous()
    Np, Cp, H_pad, W_pad = x_pad.shape
    assert Np == N and Cp == C_in

    # Flip weights spatially (H, W) to turn transposed-conv into standard conv
    w_flip = weight.flip(dims=[2, 3]).contiguous()

    # Output spatial size for valid conv on padded input
    H_out = H_pad - K_h + 1  # == H_in + K_h - 1
    W_out = W_pad - K_w + 1  # == W_in + K_w - 1
    assert H_out == H_in + K_h - 1
    assert W_out == W_in + K_w - 1

    y = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides
    x_sN, x_sC, x_sH, x_sW = x_pad.stride()
    w_sIC, w_sOC, w_sKH, w_sKW = w_flip.stride()
    y_sN, y_sC, y_sH, y_sW = y.stride()

    # Global sizes
    M_total = N * H_out * W_out
    K_total = C_in * K_h * K_w

    # Tuned tile sizes (powers of 2)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    # 3D grid:
    #   dim0: batch (N)
    #   dim1: tiles over H_out * W_out inside each batch
    #   dim2: tiles over C_out
    grid = lambda META: (
        N,
        triton.cdiv(H_out * W_out, META["BLOCK_M"]),
        triton.cdiv(C_out, META["BLOCK_N"]),
    )

    conv2d_valid_kernel[grid](
        x_pad,
        w_flip,
        y,
        N,
        C_in,
        C_out,
        K_h,
        K_w,
        x_sN,
        x_sC,
        x_sH,
        x_sW,
        w_sIC,
        w_sOC,
        w_sKH,
        w_sKW,
        y_sN,
        y_sC,
        y_sH,
        y_sW,
        H_out,
        W_out,
        M_total,
        K_total,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized replacement for nn.ConvTranspose2d with:
      - stride = (1, 1)
      - padding = (0, 0)
      - output_padding = (0, 0)
      - dilation = (1, 1)
      - groups = 1

    Other configurations are not supported and will raise an assertion.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        output_padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        # Only the configuration used in the given example is supported
        assert groups == 1, "groups != 1 not supported in Triton kernel"
        assert stride == (1, 1), "Only stride = (1, 1) supported in Triton kernel"
        assert padding == (0, 0), "Only padding = (0, 0) supported in Triton kernel"
        assert output_padding == (0, 0), "Only output_padding = (0, 0) supported"
        assert dilation == (1, 1), "Only dilation = (1, 1) supported"

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        kH, kW = kernel_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kH, kW)
        self.bias_flag = bias

        # Use PyTorch ConvTranspose2d only to initialize weights & bias
        ref = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Parameters in the same layout as nn.ConvTranspose2d:
        # weight: [C_in, C_out, K_h, K_w]
        self.weight = nn.Parameter(ref.weight.detach().clone())
        if bias:
            self.bias = nn.Parameter(ref.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA for Triton
        assert x.is_cuda, "Input must be a CUDA tensor"

        y = triton_conv_transpose2d_stride1_pad0(x, self.weight)

        if self.bias is not None:
            y = y + self.bias[None, :, None, None]

        return y
