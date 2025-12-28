import torch, torch.nn as nn, triton, triton.language as tl


@triton.jit
def avg_pool3d_kernel(
    x_ptr,            # *f32, input tensor
    y_ptr,            # *f32, output tensor
    N, C, D, H, W,    # input dimensions
    D_out, H_out, W_out,  # output dimensions
    STRIDE: tl.constexpr,
    PADDING: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_w = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_ncd = tl.program_id(axis=2)

    # Decode n, c, d_out from pid_ncd
    d_out = pid_ncd % D_out
    tmp = pid_ncd // D_out
    c = tmp % C
    n = tmp // C

    w_out_offsets = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_out_offsets < W_out

    # Input starting positions for this (d_out, h_out, w_out)
    h_out = pid_h
    d_in_start = d_out * STRIDE - PADDING
    h_in_start = h_out * STRIDE - PADDING
    w_in_start = w_out_offsets * STRIDE - PADDING

    # Accumulator
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Compute base offset for (n, c) in flattened NCDHW
    nc = n * C + c

    # Loop over kernel window
    for kd in range(0, KERNEL_SIZE):
        d_in = d_in_start + kd
        in_bounds_d = (d_in >= 0) & (d_in < D)
        for kh in range(0, KERNEL_SIZE):
            h_in = h_in_start + kh
            in_bounds_h = (h_in >= 0) & (h_in < H)

            # Scalar base index for (n, c, d_in, h_in, 0)
            # Only valid when in_bounds_d & in_bounds_h are True; otherwise we rely on masking.
            base_index_dh = ((nc * D + d_in) * H + h_in) * W

            for kw in range(0, KERNEL_SIZE):
                w_in = w_in_start + kw
                in_bounds_w = (w_in >= 0) & (w_in < W)

                mask = mask_w & in_bounds_w & in_bounds_d & in_bounds_h

                offsets = base_index_dh + w_in
                val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                acc += val

    # Divide by kernel volume (count_include_pad=True behavior)
    kernel_vol = KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE
    acc = acc / kernel_vol

    # Store result
    out_base = (((n * C + c) * D_out + d_out) * H_out + h_out) * W_out
    out_offsets = out_base + w_out_offsets
    tl.store(y_ptr + out_offsets, acc, mask=mask_w)


def triton_avg_pool3d(x: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA for Triton kernels"
    assert x.ndim == 5, "Expected input of shape (N, C, D, H, W)"

    N, C, D, H, W = x.shape

    D_out = (D + 2 * padding - kernel_size) // stride + 1
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1

    y = torch.empty((N, C, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    BLOCK_W = 128

    grid = (
        triton.cdiv(W_out, BLOCK_W),
        H_out,
        N * C * D_out,
    )

    avg_pool3d_kernel[grid](
        x,
        y,
        N,
        C,
        D,
        H,
        W,
        D_out,
        H_out,
        W_out,
        STRIDE=stride,
        PADDING=padding,
        KERNEL_SIZE=kernel_size,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )

    return y


class ModelNew(nn.Module):
    """
    Triton-optimized 3D Average Pooling module replacing nn.AvgPool3d.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return triton_avg_pool3d(x, self.kernel_size, self.stride, self.padding)
