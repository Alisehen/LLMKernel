# <complete ModelNew code with optimized Triton kernels>

import torch, torch.nn as nn, triton, triton.language as tl
import math


# ----------------------------
# Helper functions
# ----------------------------

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def window_partition(x, window_size):
    # x: (B, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    # windows: (num_windows*B, window_size, window_size, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ----------------------------
# Triton Kernels
# ----------------------------

@triton.jit
def linear_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    APPLY_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    if APPLY_GELU:
        x = acc
        sqrt_2_over_pi = 0.7978845608028654
        k = 0.044715
        x_cube = x * x * x
        inner = sqrt_2_over_pi * (x + k * x_cube)
        e2 = tl.exp(2.0 * inner)
        tanh_inner = (e2 - 1.0) / (e2 + 1.0)
        acc = 0.5 * x * (1.0 + tanh_inner)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def layernorm_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    B, C,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    in_bounds_row = row < B

    # compute mean and variance
    mean = tl.zeros((), dtype=tl.float32)
    m2 = tl.zeros((), dtype=tl.float32)

    for col_start in range(0, C, BLOCK_SIZE):
        idx = col_start + offs
        mask = (idx < C) & in_bounds_row

        x = tl.load(
            x_ptr + row * stride_xm + idx * stride_xn,
            mask=mask,
            other=0.0,
        )
        mean += tl.sum(x, axis=0)
        m2 += tl.sum(x * x, axis=0)

    mean = mean / C
    var = m2 / C - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    for col_start in range(0, C, BLOCK_SIZE):
        idx = col_start + offs
        mask = (idx < C) & in_bounds_row

        x = tl.load(
            x_ptr + row * stride_xm + idx * stride_xn,
            mask=mask,
            other=0.0,
        )
        gamma = tl.load(
            gamma_ptr + idx,
            mask=idx < C,
            other=1.0,
        )
        beta = tl.load(
            beta_ptr + idx,
            mask=idx < C,
            other=0.0,
        )

        y = (x - mean) * inv_std * gamma + beta
        tl.store(
            y_ptr + row * stride_ym + idx * stride_yn,
            y,
            mask=mask,
        )


@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    N_ROWS, N_COLS,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    in_bounds_row = row < N_ROWS
    mask = (offs < N_COLS) & in_bounds_row

    row_x_ptr = x_ptr + row * stride_x_row + offs
    x = tl.load(row_x_ptr, mask=mask, other=-float("inf"))

    x_max = tl.max(x, axis=0)
    x = x - x_max

    num = tl.exp(x)
    den = tl.sum(num, axis=0)

    out = num / den

    row_y_ptr = y_ptr + row * stride_y_row + offs
    tl.store(row_y_ptr, out, mask=mask)


# ----------------------------
# Python wrappers for kernels
# ----------------------------

def triton_linear(x, weight, bias=None, apply_gelu=False):
    """
    x: (..., in_features)
    weight: (out_features, in_features)
    bias: (out_features,) or None
    """
    assert x.is_cuda and weight.is_cuda
    in_features = weight.shape[1]
    out_features = weight.shape[0]
    assert x.shape[-1] == in_features

    x_flat = x.reshape(-1, in_features).contiguous()
    M, K = x_flat.shape

    Wt = weight.t().contiguous()  # (K, N)
    N = out_features

    y_flat = torch.empty((M, N), device=x.device, dtype=torch.float32)

    has_bias = bias is not None
    if not has_bias:
        # dummy tensor, won't be used when HAS_BIAS=False
        bias = torch.empty(1, device=x.device, dtype=torch.float32)
    else:
        bias = bias.to(dtype=torch.float32, device=x.device).contiguous()

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    linear_kernel[grid](
        x_flat, Wt, bias, y_flat,
        M, N, K,
        x_flat.stride(0), x_flat.stride(1),
        Wt.stride(0), Wt.stride(1),
        y_flat.stride(0), y_flat.stride(1),
        has_bias,
        apply_gelu,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )

    # cast back to original dtype
    y_flat = y_flat.to(x.dtype)
    y = y_flat.view(*x.shape[:-1], out_features)
    return y


def triton_layernorm(x, weight, bias, eps=1e-5):
    """
    x: (..., C)
    weight, bias: (C,)
    """
    assert x.is_cuda and weight is not None and bias is not None
    C = x.shape[-1]
    x_2d = x.reshape(-1, C).contiguous()
    B = x_2d.shape[0]

    y_2d = torch.empty_like(x_2d, dtype=torch.float32)

    w = weight.to(dtype=torch.float32, device=x.device).contiguous()
    b = bias.to(dtype=torch.float32, device=x.device).contiguous()

    BLOCK_SIZE = 256  # must be power of 2 and >= any C chunk size

    grid = lambda META: (triton.cdiv(B, 1),)
    layernorm_kernel[grid](
        x_2d, w, b, y_2d,
        B, C,
        x_2d.stride(0), x_2d.stride(1),
        y_2d.stride(0), y_2d.stride(1),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    y_2d = y_2d.to(x.dtype)
    y = y_2d.view_as(x)
    return y


def triton_softmax_lastdim(x):
    """
    Softmax over last dim using Triton.
    x: (..., N)
    """
    assert x.is_cuda
    N = x.shape[-1]
    assert N <= 128, "softmax kernel assumes last dimension <= 128"

    x_2d = x.reshape(-1, N).contiguous()
    rows, cols = x_2d.shape
    y_2d = torch.empty_like(x_2d, dtype=torch.float32)

    BLOCK_SIZE = 128

    grid = lambda META: (triton.cdiv(rows, 1),)
    softmax_kernel[grid](
        x_2d, y_2d,
        rows, cols,
        x_2d.stride(0),
        y_2d.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    y_2d = y_2d.to(x.dtype)
    y = y_2d.view_as(x)
    return y


# ----------------------------
# Triton-based modules
# ----------------------------

class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, apply_gelu=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.apply_gelu = apply_gelu
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if not x.is_cuda:
            # fall back to torch for CPU
            y = torch.nn.functional.linear(x, self.weight, self.bias)
            if self.apply_gelu:
                y = torch.nn.functional.gelu(y)
            return y
        return triton_linear(x, self.weight, self.bias, apply_gelu=self.apply_gelu)


class TritonLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if (not x.is_cuda) or (not self.elementwise_affine):
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return triton_layernorm(x, self.weight, self.bias, self.eps)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # fuse Linear + GELU in first FC
        self.fc1 = TritonLinear(in_features, hidden_features, bias=True, apply_gelu=True)
        # act_layer is ignored since we fuse GELU; kept for API compatibility
        self.drop = nn.Dropout(drop)
        self.fc2 = TritonLinear(hidden_features, out_features, bias=True, apply_gelu=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) with continuous relative position bias. """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0., pretrained_window_size=(0, 0)):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias (small; keep as torch)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        coords = torch.meshgrid(relative_coords_h, relative_coords_w, indexing='ij')
        relative_coords_table = torch.stack(coords, dim=-1).unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8.0
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0
        ) / torch.log2(torch.tensor(8.0))
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = TritonLinear(dim, dim * 3, bias=False, apply_gelu=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = TritonLinear(dim, dim, bias=True, apply_gelu=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        x: (num_windows*B, N, C)
        mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x)
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias, requires_grad=False),
                 self.v_bias),
                dim=0,
            )
            qkv = qkv + qkv_bias

        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, N, head_dim)

        # cosine attention
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1))  # (B_, num_heads, N, N)

        logit_scale = torch.clamp(
            self.logit_scale.to(x.device),
            max=torch.log(torch.tensor(1.0 / 0.01, device=x.device)),
        ).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # Triton softmax over last dim
        attn = triton_softmax_lastdim(attn)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block (simplified, matches given reference). """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=TritonLayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
        )

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, ws, ws, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, ws, ws, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, N, C

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, N, C

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer. """

    def __init__(self, input_resolution, dim, norm_layer=TritonLayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = TritonLinear(4 * dim, 2 * dim, bias=False, apply_gelu=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B H/2 W/2 4C
        x = x.view(B, -1, 4 * C)

        x = self.reduction(x)
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage. """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=TritonLayerNorm, downsample=None,
                 use_checkpoint=False, pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
            )
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding (keep Conv2d; patch sizes are small). """

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, Ph*Pw, C
        if self.norm is not None:
            x = self.norm(x)
        return x


# ----------------------------
# Top-level Model (Swin Transformer) with Triton kernels
# ----------------------------

class ModelNew(nn.Module):
    r""" Swin Transformer with Triton-optimized Linear/LayerNorm/Softmax. """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=TritonLayerNorm, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=(0, 0, 0, 0),
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule (not used functionally since drop_path is Identity in blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer],
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if num_classes > 0:
            self.head = TritonLinear(self.num_features, num_classes, bias=True, apply_gelu=False)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B, L, C
        x = self.avgpool(x.transpose(1, 2))  # B, C, 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
