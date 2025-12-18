import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv2d_3x3_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H, W, C_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_NHW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # Program IDs
    pid_nhw = tl.program_id(axis=0)  # over N*H*W
    pid_oc = tl.program_id(axis=1)   # over C_out

    # Offsets for flattened (N,H,W) and output channels
    offs_nhw = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    NHW = N * H * W
    mask_nhw = offs_nhw < NHW
    mask_oc = offs_oc < C_out

    # Compute (n, h, w) from flattened index
    hw = H * W
    n_idx = offs_nhw // hw
    rem = offs_nhw - n_idx * hw
    h_idx = rem // W
    w_idx = rem - h_idx * W

    # Accumulator for output
    acc = tl.zeros((BLOCK_NHW, BLOCK_OC), dtype=tl.float32)

    # 3x3 convolution: iterate over input channels and kernel positions
    for c in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                h_in = h_idx + kh - 1
                w_in = w_idx + kw - 1

                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                mask_x = mask_nhw & in_bounds

                # Input pointers and load
                x_ptrs = (
                    x_ptr
                    + n_idx * stride_xn
                    + c * stride_xc
                    + h_in * stride_xh
                    + w_in * stride_xw
                )
                x_vals = tl.load(x_ptrs, mask=mask_x, other=0.0)  # [BLOCK_NHW]

                # Weight pointers and load
                w_ptrs = (
                    w_ptr
                    + offs_oc * stride_wn
                    + c * stride_wc
                    + kh * stride_wh
                    + kw * stride_ww
                )
                w_vals = tl.load(w_ptrs, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                # FMA into accumulator
                acc += x_vals[:, None] * w_vals[None, :]

    # Store result
    y_ptrs = (
        y_ptr
        + n_idx[:, None] * stride_yn
        + offs_oc[None, :] * stride_yc
        + h_idx[:, None] * stride_yh
        + w_idx[:, None] * stride_yw
    )
    mask_store = mask_nhw[:, None] & mask_oc[None, :]
    tl.store(y_ptrs, acc, mask=mask_store)


def conv2d_3x3_triton(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D convolution with:
      - kernel_size = 3x3
      - stride = 1
      - padding = 1 (same spatial size)
      - dilation = 1
      - groups = 1
      - bias = False (handled outside if needed)

    Args:
        x: (N, C_in, H, W) float32 CUDA tensor
        weight: (C_out, C_in, 3, 3) float32 CUDA tensor

    Returns:
        y: (N, C_out, H, W) float32 CUDA tensor
    """
    assert x.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
    assert x.dtype == torch.float32 and weight.dtype == torch.float32, "Only float32 supported"
    assert x.dim() == 4 and weight.dim() == 4, "Expected 4D tensors"
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w, "Channel mismatch between input and weight"
    assert KH == 3 and KW == 3, "Kernel size must be 3x3"

    # Ensure contiguity (simplifies stride handling, safe for DenseNet pattern)
    x_c = x.contiguous()
    w_c = weight.contiguous()

    y = torch.empty((N, C_out, H, W), device=x.device, dtype=x.dtype)

    stride_xn, stride_xc, stride_xh, stride_xw = x_c.stride()
    stride_wn, stride_wc, stride_wh, stride_ww = w_c.stride()
    stride_yn, stride_yc, stride_yh, stride_yw = y.stride()

    BLOCK_NHW = 32  # power-of-2
    BLOCK_OC = 32   # power-of-2

    grid = lambda META: (
        triton.cdiv(N * H * W, META["BLOCK_NHW"]),
        triton.cdiv(C_out, META["BLOCK_OC"]),
    )

    conv2d_3x3_kernel[grid](
        x_c, w_c, y,
        N, C_in, H, W, C_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wn, stride_wc, stride_wh, stride_ww,
        stride_yn, stride_yc, stride_yh, stride_yw,
        BLOCK_NHW=BLOCK_NHW,
        BLOCK_OC=BLOCK_OC,
    )

    return y


class ModelNew(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        Triton-accelerated version of the given Dense block model.
        Conv2d(3x3) layers are computed via a Triton kernel.
        """
        super(ModelNew, self).__init__()
        layers = []
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            layers.append(self._make_layer(in_features, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        """
        Same structure as original:
        BatchNorm2d -> ReLU -> Conv2d(3x3, padding=1, bias=False) -> Dropout
        Conv2d weights are later used by the Triton kernel in forward().
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        """
        :param x: (batch_size, num_input_features, height, width)
        :return: (batch_size, num_output_features, height, width)
        """
        features = [x]
        for layer in self.layers:
            bn = layer[0]
            relu = layer[1]
            conv = layer[2]
            dropout = layer[3]

            out = bn(x)
            out = relu(out)
            # Triton-accelerated 3x3 convolution (bias=False)
            out = conv2d_3x3_triton(out, conv.weight)
            if dropout.p > 0.0:
                out = F.dropout(out, p=dropout.p, training=self.training)

            features.append(out)
            x = torch.cat(features, dim=1)
        return x
