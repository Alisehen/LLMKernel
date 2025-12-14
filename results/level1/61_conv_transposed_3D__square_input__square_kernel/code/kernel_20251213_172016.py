# Optimized Triton ConvTranspose3d with autotuned BLOCK_P / BLOCK_OC / BLOCK_K

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Baseline config (kept for safety / regression-free)
        triton.Config(
            {"BLOCK_P": 16, "BLOCK_OC": 32, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Larger tiles in P and OC, same K
        triton.Config(
            {"BLOCK_P": 32, "BLOCK_OC": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=2,
        ),
        # Larger K tile to reduce loop trips over K
        triton.Config(
            {"BLOCK_P": 32, "BLOCK_OC": 64, "BLOCK_K": 64},
            num_warps=8,
            num_stages=2,
        ),
        # Even larger P tile; good when spatial dims are big
        triton.Config(
            {"BLOCK_P": 64, "BLOCK_OC": 64, "BLOCK_K": 32},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=[
        "N",
        "ID",
        "IH",
        "IW",
        "OD",
        "OH",
        "OW",
        "IC",
        "OC",
        "KD",
        "KH",
        "KW",
        "HAS_BIAS",
    ],
)
@triton.jit
def conv_transpose3d_kernel(
    x_ptr,           # *float32,  shape [N, IC, ID, IH, IW]
    w_ptr,           # *float32,  shape [IC, OC, KD, KH, KW]
    b_ptr,           # *float32,  shape [OC] (or dummy if no bias)
    out_ptr,         # *float32,  shape [N, OC, OD, OH, OW]
    N,               # int32
    ID, IH, IW,      # int32
    OD, OH, OW,      # int32
    IC: tl.constexpr,
    OC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_p = tl.program_id(axis=0)   # over output positions (N * OD * OH * OW)
    pid_oc = tl.program_id(axis=1)  # over output channels

    # Offsets along flattened output position and channel dimensions
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    P_total = N * OD * OH * OW
    mask_p = offs_p < P_total
    mask_oc = offs_oc < OC

    # Decode flattened position -> (n, od, oh, ow)
    tmp = offs_p
    ow = tmp % OW
    tmp = tmp // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    n = tmp  # batch index

    # Initialize accumulator
    acc = tl.zeros((BLOCK_P, BLOCK_OC), dtype=tl.float32)

    # Total K dimension: IC * KD * KH * KW
    K_TOTAL = IC * KD * KH * KW

    # Broadcasted position tensors for later use
    n_bc = n[:, None]   # [P, 1]
    od_bc = od[:, None]
    oh_bc = oh[:, None]
    ow_bc = ow[:, None]

    KD_KH_KW = KD * KH * KW
    KH_KW = KH * KW

    # Loop over K dimension in tiles of BLOCK_K
    for k_start in range(0, K_TOTAL, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [K]
        k_mask = k_offsets < K_TOTAL

        # Decode k_offsets -> (ic, kd, kh, kw)
        tmpk = k_offsets
        ic = tmpk // KD_KH_KW
        rem = tmpk % KD_KH_KW
        kd = rem // KH_KW
        rem2 = rem % KH_KW
        kh = rem2 // KW
        kw = rem2 % KW

        # Broadcasted kernel indices
        ic_bc = ic[None, :]   # [1, K]
        kd_bc = kd[None, :]
        kh_bc = kh[None, :]
        kw_bc = kw[None, :]

        # Compute input coordinates for each (position, k)
        id_ = od_bc - kd_bc
        ih_ = oh_bc - kh_bc
        iw_ = ow_bc - kw_bc

        # Mask for valid input coordinates
        mask_x = (
            mask_p[:, None] & k_mask[None, :] &
            (id_ >= 0) & (id_ < ID) &
            (ih_ >= 0) & (ih_ < IH) &
            (iw_ >= 0) & (iw_ < IW)
        )

        # Flat input index: (((n * IC + ic) * ID + id_) * IH + ih_) * IW + iw_
        base_nc = n_bc * IC + ic_bc               # [P, K]
        idx_x = ((base_nc * ID + id_) * IH + ih_) * IW + iw_

        x_vals = tl.load(x_ptr + idx_x, mask=mask_x, other=0.0)

        # Load weight tile [K, OC_tile]
        oc_bc = offs_oc[None, :]   # [1, OC_tile]
        icw_bc = ic[:, None]       # [K, 1]
        kdw_bc = kd[:, None]
        khw_bc = kh[:, None]
        kww_bc = kw[:, None]

        # Flat weight index: ((((ic * OC + oc) * KD + kd) * KH + kh) * KW + kw)
        idx_w = (
            (((icw_bc * OC + oc_bc) * KD + kdw_bc) * KH + khw_bc) * KW + kww_bc
        )

        mask_w = k_mask[:, None] & mask_oc[None, :]
        w_vals = tl.load(w_ptr + idx_w, mask=mask_w, other=0.0)

        # Accumulate using matmul-style dot
        # (x_vals: [P, K], w_vals: [K, OC_tile]) -> [P, OC_tile]
        acc += tl.dot(x_vals.to(tl.float32), w_vals.to(tl.float32))

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)  # [OC_tile]
        acc += bias[None, :]

    # Compute output indices and store
    oc_bc = offs_oc[None, :]  # [1, OC_tile]

    out_idx = (
        ((((n_bc * OC + oc_bc) * OD + od_bc) * OH + oh_bc) * OW + ow_bc)
    )

    out_mask = mask_p[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def _triple(x):
    if isinstance(x, int):
        return (x, x, x)
    return x


def triton_conv_transpose3d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride,
    padding,
    output_padding,
    groups: int,
) -> torch.Tensor:
    """
    Triton implementation of ConvTranspose3d for the restricted but common case:
    - stride = (1, 1, 1)
    - padding = (0, 0, 0)
    - output_padding = (0, 0, 0)
    - groups = 1

    Falls back to torch.nn.functional.conv_transpose3d for unsupported configs or non-CUDA tensors.
    """
    # Fallback for non-CUDA tensors
    if not x.is_cuda:
        return torch.nn.functional.conv_transpose3d(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )

    stride = _triple(stride)
    padding = _triple(padding)
    output_padding = _triple(output_padding)

    # Only handle the simple common case with the Triton kernel
    if (
        stride != (1, 1, 1)
        or padding != (0, 0, 0)
        or output_padding != (0, 0, 0)
        or groups != 1
    ):
        return torch.nn.functional.conv_transpose3d(
            x,
            weight,
            bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
        )

    # Ensure contiguous layout
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    N, IC, ID, IH, IW = x.shape
    ICw, OC_per_group, KD, KH, KW = weight.shape
    assert ICw == IC, "Input channels mismatch between x and weight"
    assert groups == 1, "Groups > 1 not supported in Triton path"
    OC = OC_per_group * groups

    # Output spatial dimensions for stride=1, padding=0, output_padding=0
    OD = ID + KD - 1
    OH = IH + KH - 1
    OW = IW + KW - 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    P_total = N * OD * OH * OW

    def grid(meta):
        return (
            triton.cdiv(P_total, meta["BLOCK_P"]),
            triton.cdiv(OC, meta["BLOCK_OC"]),
        )

    conv_transpose3d_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,  # dummy ptr if no bias
        out,
        N,
        ID,
        IH,
        IW,
        OD,
        OH,
        OW,
        IC=IC,
        OC=OC,
        KD=KD,
        KH=KH,
        KW=KW,
        HAS_BIAS=bias is not None,
    )

    return out


class ModelNew(nn.Module):
    """
    Performs a transposed 3D convolution with square input and square kernel,
    using a Triton kernel for the common case (stride=1, padding=0, output_padding=0, groups=1),
    and falling back to PyTorch otherwise.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.conv_transpose3d.weight
        b = self.conv_transpose3d.bias
        stride = self.conv_transpose3d.stride
        padding = self.conv_transpose3d.padding
        output_padding = self.conv_transpose3d.output_padding
        groups = self.conv_transpose3d.groups

        return triton_conv_transpose3d(x, w, b, stride, padding, output_padding, groups)
