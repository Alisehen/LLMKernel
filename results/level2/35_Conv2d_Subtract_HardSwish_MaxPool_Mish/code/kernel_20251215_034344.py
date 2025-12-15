import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=2, num_stages=3),
    ],
    key=["N", "H_out", "W_out", "C_out"],
)
@triton.jit
def conv2d_sub_hswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H_out, W_out,
    C_out,
    subtract_value,
    stride_x_n, stride_x_c, stride_x_h, stride_x_w,
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
    stride_out_n, stride_out_c, stride_out_h, stride_out_w,
    C_IN: tl.constexpr,
    K_H: tl.constexpr,
    K_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    P = N * H_out * W_out
    mask_m = offs_m < P
    mask_n = offs_n < C_out

    HW_out = H_out * W_out
    n_idx = offs_m // HW_out
    rem = offs_m % HW_out
    oh_idx = rem // W_out
    ow_idx = rem % W_out

    x_base = (
        x_ptr
        + n_idx * stride_x_n
        + oh_idx * stride_x_h
        + ow_idx * stride_x_w
    )
    w_base = w_ptr + offs_n * stride_w_co

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ic in tl.static_range(0, C_IN):
        x_ic_off = ic * stride_x_c
        w_ic_off = ic * stride_w_ci
        for kh in tl.static_range(0, K_H):
            x_kh_off = kh * stride_x_h
            w_kh_off = kh * stride_w_kh
            for kw in tl.static_range(0, K_W):
                x_kw_off = kw * stride_x_w
                w_kw_off = kw * stride_w_kw

                x_ptrs = x_base + x_ic_off + x_kh_off + x_kw_off
                x_vals = tl.load(x_ptrs, mask=mask_m, other=0.0).to(tl.float32)

                w_ptrs = w_base + w_ic_off + w_kh_off + w_kw_off
                w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0).to(tl.float32)

                acc += x_vals[:, None] * w_vals[None, :]

    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    acc -= subtract_value

    tmp = acc + 3.0
    tmp = tl.maximum(tmp, 0.0)
    tmp = tl.minimum(tmp, 6.0)
    acc = acc * tmp * (1.0 / 6.0)

    out_ptrs = (
        out_ptr
        + n_idx[:, None] * stride_out_n
        + offs_n[None, :] * stride_out_c
        + oh_idx[:, None] * stride_out_h
        + ow_idx[:, None] * stride_out_w
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def conv2d_sub_hswish(x, weight, bias, subtract_value):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    N, C_in, H_in, W_in = x.shape
    C_out, C_in_w, K_H, K_W = weight.shape
    assert C_in == C_in_w

    H_out = H_in - K_H + 1
    W_out = W_in - K_W + 1

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    out = torch.empty((N, C_out, H_out, W_out),
                      device=x.device,
                      dtype=x.dtype)

    stride_x_n, stride_x_c, stride_x_h, stride_x_w = x_contig.stride()
    stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw = w_contig.stride()
    stride_out_n, stride_out_c, stride_out_h, stride_out_w = out.stride()

    P = N * H_out * W_out

    def grid(meta):
        return (
            triton.cdiv(P, meta["BLOCK_M"]),
            triton.cdiv(C_out, meta["BLOCK_N"]),
        )

    conv2d_sub_hswish_kernel[grid](
        x_contig, w_contig, b_contig, out,
        N, H_out, W_out,
        C_out,
        float(subtract_value),
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_w_co, stride_w_ci, stride_w_kh, stride_w_kw,
        stride_out_n, stride_out_c, stride_out_h, stride_out_w,
        C_IN=C_in,
        K_H=K_H,
        K_W=K_W,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.subtract_value = float(subtract_value)
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = conv2d_sub_hswish(x, self.weight, self.bias, self.subtract_value)
        x = self.pool(x)
        x = torch.nn.functional.mish(x)
        return x
