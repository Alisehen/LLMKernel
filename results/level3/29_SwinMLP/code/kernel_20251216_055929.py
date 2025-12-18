import torch
import torch.nn as nn
import triton
import triton.language as tl


# ------------------------------------------------------------
# Fused GEMM(+bias) Triton kernel: C[M, N] = A[M, K] @ W[N, K]^T + bias[N]
#   A: [M, K], row-major
#   W: [N, K], row-major (nn.Linear.weight layout)
#   C: [M, N], row-major
#
# Optimized for Ada (RTX 4090) with a small set of autotuned configs
# focusing on num_warps / num_stages as requested.
# ------------------------------------------------------------

@triton.autotune(
    configs=[
        # Conservative baseline: good occupancy, low register pressure.
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
        # High-throughput config for large, square-ish matrices.
        # More warps and pipeline stages to better hide latency
        # when register pressure allows.
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 32,
                "GROUP_M": 4,
            },
            num_warps=8,
            num_stages=3,
        ),
        # Rectangular fallback for tall/skinny or short/wide shapes.
        triton.Config(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "GROUP_M": 8,
            },
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _linear_kernel_fwd(
    a_ptr,            # *A
    w_ptr,            # *W  (N, K)
    bias_ptr,         # *bias (N,) or dummy
    c_ptr,            # *C
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Compute C = A @ W^T + bias

    A: [M, K], strides (stride_am, stride_ak)
    W: [N, K], strides (stride_wn, stride_wk)
       We treat W^T logically as [K, N] without materializing it.
    bias: [N]
    C: [M, N], strides (stride_cm, stride_cn)
    """

    # 1D program id with grouped swizzle over M to improve L2 locality on N
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    group_size = GROUP_M

    group_id = pid // (group_size * num_pid_n)
    first_pid_m = group_id * group_size
    group_size_m = tl.minimum(num_pid_m - first_pid_m, group_size)
    pid_in_group = pid % (group_size * num_pid_n)

    pid_m = first_pid_m + (pid_in_group // num_pid_n)
    pid_n = pid_in_group % num_pid_n

    # Compute offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    # Pointer to the first A and W tiles for this block
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    # Treat W[n, k] as B^T[k, n] without explicit transpose
    w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    # Accumulator in FP32 (to leverage TF32 / tensor cores on Ada)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    k_iter = 0
    while k_iter < K:
        k_mask = offs_k + k_iter < K

        a = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )

        # Matrix multiply accumulation (uses tensor cores when available)
        acc += tl.dot(a, w, allow_tf32=True)

        # Advance along K
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk
        k_iter += BLOCK_K

    # Add bias in-register if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias[None, :]

    # Write back the final result (single global store, no intermediates)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc,
        mask=m_mask[:, None] & n_mask[None, :],
    )


# ------------------------------------------------------------
# Wrapper: fused linear using Triton (A @ W^T + bias)
# ------------------------------------------------------------

def triton_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """
    Fused linear using Triton: y = x @ weight.T + bias

    Args:
        x:      [M, K], row-major
        weight: [N, K], row-major (nn.Linear.weight layout)
        bias:   [N] or None
    """
    assert x.is_cuda and weight.is_cuda, "triton_linear only supports CUDA tensors"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), "Unsupported dtype for Triton linear"
    assert weight.dtype == x.dtype, "x and weight must have the same dtype"

    # Ensure contiguous for predictable strides & alignment
    x = x.contiguous()
    weight = weight.contiguous()

    M, K = x.shape
    N, Kw = weight.shape
    assert Kw == K, f"Incompatible shapes: x: [{M}, {K}], weight: [{N}, {Kw}]"

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    has_bias = bias is not None
    if has_bias:
        assert bias.is_cuda and bias.dtype == x.dtype
        assert bias.numel() == N
        bias_ptr = bias
    else:
        # dummy tensor; never accessed when HAS_BIAS=False
        bias_ptr = weight.new_empty(1)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )

    _linear_kernel_fwd[grid](
        x,
        weight,
        bias_ptr,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        HAS_BIAS=has_bias,
    )

    return y


# ------------------------------------------------------------
# Full Swin-MLP-style model using Triton-fused linears
# ------------------------------------------------------------

class ModelNew(nn.Module):
    r"""Swin MLP (with Triton-fused linear layers in MLPs, patch merging, and head)

    This is a refactoring of the provided architecture so that all dense
    projections use the optimized Triton kernel above.
    """

    # -----------------------------
    # Static helpers
    # -----------------------------

    @staticmethod
    def to_2tuple(x):
        if isinstance(x, tuple):
            return x
        return (x, x)

    @staticmethod
    def window_partition(x, window_size):
        """
        x: (B, H, W, C)
        returns windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(
            B,
            H // window_size,
            window_size,
            W // window_size,
            window_size,
            C,
        )
        windows = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, window_size, window_size, C)
        )
        return windows

    @staticmethod
    def window_reverse(windows, window_size, H, W):
        """
        windows: (num_windows*B, window_size, window_size, C)
        returns x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(
            B,
            H // window_size,
            W // window_size,
            window_size,
            window_size,
            -1,
        )
        x = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, H, W, -1)
        )
        return x

    # -----------------------------
    # Nested modules
    # -----------------------------

    class Mlp(nn.Module):
        """
        Two-layer MLP:
        x -> Linear(in, hidden) -> GELU -> Dropout
          -> Linear(hidden, out) -> Dropout

        Linear projections are computed with Triton GEMM (+bias).
        """

        def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
        ):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features

            # Parameters only; forward uses triton_linear
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            # x: [B, L, C]
            B, L, C = x.shape

            # First linear with Triton
            x_2d = x.contiguous().view(B * L, C)
            x_2d = triton_linear(x_2d, self.fc1.weight, self.fc1.bias)
            x = x_2d.view(B, L, self.fc1.out_features)

            x = self.act(x)
            x = self.drop(x)

            # Second linear with Triton
            B, L, C_hidden = x.shape
            x_2d = x.contiguous().view(B * L, C_hidden)
            x_2d = triton_linear(x_2d, self.fc2.weight, self.fc2.bias)
            x = x_2d.view(B, L, self.fc2.out_features)

            x = self.drop(x)
            return x

    class SwinMLPBlock(nn.Module):
        r""" Swin MLP Block.
        """

        def __init__(
            self,
            dim,
            input_resolution,
            num_heads,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
        ):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.num_heads = num_heads
            self.window_size = window_size
            self.shift_size = shift_size
            self.mlp_ratio = mlp_ratio

            if min(self.input_resolution) <= self.window_size:
                # if window size is larger than input resolution, we don't partition windows
                self.shift_size = 0
                self.window_size = min(self.input_resolution)
            assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

            self.padding = [
                self.window_size - self.shift_size,
                self.shift_size,
                self.window_size - self.shift_size,
                self.shift_size,
            ]  # P_l,P_r,P_t,P_b

            self.norm1 = norm_layer(dim)
            # use group convolution to implement multi-head MLP
            self.spatial_mlp = nn.Conv1d(
                self.num_heads * self.window_size**2,
                self.num_heads * self.window_size**2,
                kernel_size=1,
                groups=self.num_heads,
            )

            # drop_path kept as Identity here (no stochastic depth)
            self.drop_path = nn.Identity()
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = ModelNew.Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )

        def forward(self, x):
            H, W = self.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"

            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # shift
            if self.shift_size > 0:
                P_l, P_r, P_t, P_b = self.padding
                x = torch.nn.functional.pad(
                    x, [0, 0, P_l, P_r, P_t, P_b], "constant", 0
                )
            shifted_x = x
            _, _H, _W, _ = shifted_x.shape

            # partition windows
            x_windows = ModelNew.window_partition(
                shifted_x, self.window_size
            )  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(
                -1, self.window_size * self.window_size, C
            )  # nW*B, window_size*window_size, C

            # Window/Shifted-Window Spatial MLP
            x_windows_heads = x_windows.view(
                -1,
                self.window_size * self.window_size,
                self.num_heads,
                C // self.num_heads,
            )
            x_windows_heads = x_windows_heads.transpose(
                1, 2
            )  # nW*B, nH, window_size*window_size, C//nH
            x_windows_heads = x_windows_heads.reshape(
                -1,
                self.num_heads * self.window_size * self.window_size,
                C // self.num_heads,
            )
            spatial_mlp_windows = self.spatial_mlp(
                x_windows_heads
            )  # nW*B, nH*window_size*window_size, C//nH
            spatial_mlp_windows = spatial_mlp_windows.view(
                -1,
                self.num_heads,
                self.window_size * self.window_size,
                C // self.num_heads,
            ).transpose(1, 2)
            spatial_mlp_windows = spatial_mlp_windows.reshape(
                -1,
                self.window_size * self.window_size,
                C,
            )

            # merge windows
            spatial_mlp_windows = spatial_mlp_windows.reshape(
                -1,
                self.window_size,
                self.window_size,
                C,
            )
            shifted_x = ModelNew.window_reverse(
                spatial_mlp_windows,
                self.window_size,
                _H,
                _W,
            )  # B H' W' C

            # reverse shift
            if self.shift_size > 0:
                P_l, P_r, P_t, P_b = self.padding
                x = shifted_x[:, P_t:-P_b, P_l:-P_r, :].contiguous()
            else:
                x = shifted_x
            x = x.view(B, H * W, C)

            # FFN
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            return x

    class PatchMerging(nn.Module):
        r""" Patch Merging Layer.
        """

        def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
            super().__init__()
            self.input_resolution = input_resolution
            self.dim = dim
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

        def forward(self, x):
            """
            x: B, H*W, C
            """
            H, W = self.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"
            assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

            x = x.view(B, H, W, C)

            x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
            x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
            x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
            x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
            x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

            x = self.norm(x)
            B2, L2, C4 = x.shape
            x_2d = x.contiguous().view(B2 * L2, C4)
            # reduction has no bias
            x_2d = triton_linear(x_2d, self.reduction.weight, None)
            x = x_2d.view(B2, L2, 2 * self.dim)

            return x

    class BasicLayer(nn.Module):
        """A basic Swin MLP layer for one stage.
        """

        def __init__(
            self,
            dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            mlp_ratio=4.0,
            drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            downsample=None,
        ):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.depth = depth

            # build blocks
            self.blocks = nn.ModuleList(
                [
                    ModelNew.SwinMLPBlock(
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        drop=drop,
                        drop_path=(
                            drop_path[i]
                            if isinstance(drop_path, (list, tuple))
                            else drop_path
                        ),
                        norm_layer=norm_layer,
                    )
                    for i in range(depth)
                ]
            )

            # patch merging layer
            if downsample is not None:
                self.downsample = downsample(
                    input_resolution,
                    dim=dim,
                    norm_layer=norm_layer,
                )
            else:
                self.downsample = None

        def forward(self, x):
            for blk in self.blocks:
                x = blk(x)
            if self.downsample is not None:
                x = self.downsample(x)
            return x

    class PatchEmbed(nn.Module):
        r"""Image to Patch Embedding
        """

        def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            norm_layer=None,
        ):
            super().__init__()
            img_size = ModelNew.to_2tuple(img_size)
            patch_size = ModelNew.to_2tuple(patch_size)
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
            assert H == self.img_size[0] and W == self.img_size[1], (
                f"Input image size ({H}*{W}) doesn't match model "
                f"({self.img_size[0]}*{self.img_size[1]})."
            )
            x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
            if self.norm is not None:
                x = self.norm(x)
            return x

        def flops(self):
            Ho, Wo = self.patches_resolution
            flops = (
                Ho
                * Wo
                * self.embed_dim
                * self.in_chans
                * (self.patch_size[0] * self.patch_size[1])
            )
            if self.norm is not None:
                flops += Ho * Wo * self.embed_dim
            return flops

    # -----------------------------
    # ModelNew main class
    # -----------------------------

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = ModelNew.PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth (kept but no actual DropPath; values unused)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ModelNew.BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                downsample=ModelNew.PatchMerging
                if (i_layer < self.num_layers - 1)
                else None,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # keep nn.Linear for parameters; forward will use Triton
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # Head projection via Triton linear if applicable
        if isinstance(self.head, nn.Linear):
            x = triton_linear(x, self.head.weight, self.head.bias)
        else:
            x = self.head(x)
        return x
