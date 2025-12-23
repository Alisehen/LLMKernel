import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv_min_tanh2_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W,
    C_out, K_H, K_W,
    H_out, W_out,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_OC: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    P = N * H_out * W_out

    # Mask for valid output pixels (some pids may be out-of-range)
    mask_valid = pid < P

    # Map linear index -> (n, h_out, w_out)
    hw_size = H_out * W_out
    n = pid // hw_size
    rem = pid % hw_size
    h_out_idx = rem // W_out
    w_out_idx = rem % W_out

    # Precompute total K = C_in * K_H * K_W
    KH_KW = K_H * K_W
    K_total = C_in * KH_KW

    # Initialize running minimum over output channels for this pixel
    min_val = tl.zeros((), dtype=tl.float32) + 1e30

    oc_start = 0
    while oc_start < C_out:
        oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
        mask_oc = oc_offsets < C_out

        # Accumulator for this OC block (float32 for precision)
        acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

        k_start = 0
        while k_start < K_total:
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < K_total

            # Map K index -> (c_in, kh, kw)
            cin = k_offsets // KH_KW
            khkw = k_offsets % KH_KW
            kh = khkw // K_W
            kw = khkw % K_W

            # Load weights: shape [BLOCK_OC, BLOCK_K]
            w_ptrs = (
                w_ptr
                + oc_offsets[:, None] * stride_w_cout
                + cin[None, :] * stride_w_cin
                + kh[None, :] * stride_w_kh
                + kw[None, :] * stride_w_kw
            )
            w = tl.load(
                w_ptrs,
                mask=mask_oc[:, None] & mask_k[None, :],
                other=0.0,
            )

            # Load input patch for this output pixel: shape [BLOCK_K]
            x_ptrs = (
                x_ptr
                + n * stride_x_n
                + cin * stride_x_c
                + (h_out_idx + kh) * stride_x_h
                + (w_out_idx + kw) * stride_x_w
            )
            x_vals = tl.load(
                x_ptrs,
                mask=mask_valid & mask_k,
                other=0.0,
            )

            # FMA over K dimension using dot: [BLOCK_OC, BLOCK_K] x [BLOCK_K] -> [BLOCK_OC]
            acc += tl.dot(w, x_vals, allow_tf32=True)

            k_start += BLOCK_K

        # Add bias for this OC block
        b_ptrs = b_ptr + oc_offsets
        b_vals = tl.load(b_ptrs, mask=mask_oc, other=0.0)
        acc += b_vals

        # Compute block-wise minimum over valid OC entries
        inf_like = 1e30
        acc_masked = tl.where(mask_oc, acc, inf_like)
        block_min = -tl.max(-acc_masked, axis=0)

        # Update running minimum
        min_val = tl.minimum(min_val, block_min)

        oc_start += BLOCK_OC

    # Apply tanh twice: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    x1 = min_val
    e2x1 = tl.exp(2.0 * x1)
    tanh1 = (e2x1 - 1.0) / (e2x1 + 1.0)

    e2x2 = tl.exp(2.0 * tanh1)
    tanh2 = (e2x2 - 1.0) / (e2x2 + 1.0)

    # Store result to output: shape (N, 1, H_out, W_out)
    out_offset = (
        n * stride_out_n
        + 0 * stride_out_c
        + h_out_idx * stride_out_h
        + w_out_idx * stride_out_w
    )
    tl.store(out_ptr + out_offset, tanh2, mask=mask_valid)


def conv_min_tanh2(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Fused:
      y = conv2d(x, weight, bias, stride=1, padding=0)
      y = torch.min(y, dim=1, keepdim=True)[0]
      y = torch.tanh(y)
      y = torch.tanh(y)
    Assumes NCHW layout, stride=1, padding=0, dilation=1.
    """
    assert x.dim() == 4, "Input must be NCHW"
    N, C_in, H, W = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w, "Input channels must match weight"

    # Compute output spatial size for stride=1, padding=0
    H_out = H - K_H + 1
    W_out = W - K_W + 1
    assert H_out > 0 and W_out > 0, "Invalid kernel size for given input"

    # Ensure contiguous and on same device
    x_ = x.contiguous()
    w_ = weight.contiguous()
    b_ = bias.contiguous()

    # Allocate output: N x 1 x H_out x W_out
    out = torch.empty((N, 1, H_out, W_out), device=x.device, dtype=x.dtype)

    # Strides
    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x_.stride()
    stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw = w_.stride()
    stride_out_n, stride_out_c, stride_out_h, stride_out_w = out.stride()

    P = N * H_out * W_out

    def grid(meta):
        return (max(1, P),)

    conv_min_tanh2_kernel[grid](
        x_, w_, b_, out,
        N, C_in, H, W,
        C_out, K_H, K_W,
        H_out, W_out,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
        stride_out_n, stride_out_c, stride_out_h, stride_out_w,
        BLOCK_OC=64,
        BLOCK_K=32,
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
