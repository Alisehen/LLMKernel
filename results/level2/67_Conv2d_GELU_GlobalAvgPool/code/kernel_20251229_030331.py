# <complete ModelNew code with optimized Triton kernels>
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_gelu_gap_matmul_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, C_in, H, W, C_out, K, H_out, W_out, P,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_outb, stride_outc,
    inv_P,
    BLOCK_M: tl.constexpr,  # block size along output channels (M)
    BLOCK_N: tl.constexpr,  # block size along spatial positions (N = H_out*W_out)
    BLOCK_K: tl.constexpr,  # block size along reduction dim (K = C_in*K*K)
):
    # Grid:
    #   pid_b : batch index
    #   pid_m : block of output channels
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_b
    # Offsets for output channels within this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < C_out

    # Base pointers
    x_batch_ptr = x_ptr + b * stride_xb
    out_batch_ptr = out_ptr + b * stride_outb

    # Load bias for this block of output channels
    bias_vals = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)

    # Total reduction length: R = C_in * K * K
    R = C_in * K * K

    # Accumulator over spatial positions (for global average pooling)
    acc_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over spatial positions in tiles of BLOCK_N (N dimension of GEMM)
    n_start = 0
    while n_start < P:
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < P

        # Map flattened spatial index -> (oh, ow)
        oh = offs_n // W_out
        ow = offs_n - oh * W_out  # (BLOCK_N,)

        # Accumulator for the current Y_tile = W @ X_col (shape BLOCK_M x BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop over reduction dimension (R) in tiles of BLOCK_K
        r_start = 0
        while r_start < R:
            offs_k = r_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < R

            # Decode reduction index -> (ic, kh, kw)
            kk = K * K
            ic = offs_k // kk
            rem = offs_k - ic * kk
            kh = rem // K
            kw = rem - kh * K

            # Compute input coordinates for each (k, n) pair
            # in_h[k, n] = oh[n] + kh[k]
            # in_w[k, n] = ow[n] + kw[k]
            in_h = oh[None, :] + kh[:, None]
            in_w = ow[None, :] + kw[:, None]

            # Pointers for input x[b, ic, in_h, in_w]
            x_ptrs = (
                x_batch_ptr
                + ic[:, None] * stride_xc
                + in_h * stride_xh
                + in_w * stride_xw
            )

            mask_x = mask_k[:, None] & mask_n[None, :]

            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

            # Pointers for weights w[oc, ic, kh, kw]
            w_ptrs = (
                w_ptr
                + offs_m[:, None] * stride_wco
                + ic[None, :] * stride_wci
                + kh[None, :] * stride_wkh
                + kw[None, :] * stride_wkw
            )

            mask_w = mask_m[:, None] & mask_k[None, :]

            w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

            # Matmul accumulation: (BLOCK_M x BLOCK_K) @ (BLOCK_K x BLOCK_N)
            acc += tl.dot(w_tile, x_tile, allow_tf32=True)

            r_start += BLOCK_K

        # Add bias and apply GELU to Y_tile
        acc = acc + bias_vals[:, None]

        # GELU (tanh-based approximation), same as in the original kernel
        c0 = 0.7978845608028654  # sqrt(2/pi)
        c1 = 0.044715
        x_val = acc
        x3 = x_val * x_val * x_val
        inner = c0 * (x_val + c1 * x3)
        t = tl.exp(2.0 * inner)
        tanh_inner = (t - 1.0) / (t + 1.0)
        gelu_val = 0.5 * x_val * (1.0 + tanh_inner)

        # Mask out-of-range channels/positions
        mask_mn = mask_m[:, None] & mask_n[None, :]
        gelu_val = tl.where(mask_mn, gelu_val, 0.0)

        # Accumulate over spatial positions for global average pooling
        tile_sum = tl.sum(gelu_val, axis=1)  # sum over N dimension
        acc_sum += tile_sum

        n_start += BLOCK_N

    # Finish global average pooling
    out_vals = acc_sum * inv_P

    # Store results: out[b, oc]
    out_ptrs = out_batch_ptr + offs_m * stride_outc
    tl.store(out_ptrs, out_vals, mask=mask_m)


def fused_conv_gelu_gap(x, weight, bias):
    """
    x:      (B, C_in, H, W)
    weight: (C_out, C_in, K, K)
    bias:   (C_out,)
    returns: (B, C_out) after Conv2d -> GELU -> global average pooling
    """
    assert x.is_cuda, "Input must be on CUDA device for Triton kernel"
    B, C_in, H, W = x.shape
    C_out, C_in_w, K_h, K_w = weight.shape
    assert C_in == C_in_w, "in_channels mismatch between input and weight"
    assert K_h == K_w, "Only square kernels supported"
    K = K_h

    # Valid convolution (no padding, stride 1)
    H_out = H - K + 1
    W_out = W - K + 1
    P = H_out * W_out
    inv_P = 1.0 / float(P)

    out = torch.empty((B, C_out), device=x.device, dtype=x.dtype)

    # Launch grid: one block per batch and per output-channel tile
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (B, triton.cdiv(C_out, BLOCK_M))

    conv_gelu_gap_matmul_kernel[grid](
        x, weight, bias, out,
        B, C_in, H, W, C_out, K, H_out, W_out, P,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out.stride(0), out.stride(1),
        inv_P,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    """
    Triton-optimized version of:
      Conv2d -> GELU -> global average pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        if isinstance(kernel_size, tuple):
            assert kernel_size[0] == kernel_size[1], "Only square kernels supported"
            k = kernel_size[0]
        else:
            k = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, k, k)
        )
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        return fused_conv_gelu_gap(x, self.weight, self.bias)
