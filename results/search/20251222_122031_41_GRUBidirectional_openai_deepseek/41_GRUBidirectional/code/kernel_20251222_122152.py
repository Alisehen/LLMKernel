import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    ADD_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C[M, N] = A[M, K] @ B[K, N] (+ bias[N]) using block GEMM.
    A, B, C are row-major.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        k_remaining = K - k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k += BLOCK_K

    if ADD_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


def triton_matmul_bias(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    Wrapper for matmul_bias_kernel:
    Computes a @ b (+ bias) where:
      a: (M, K), row-major
      b: (K, N), row-major
      bias: (N,)
    Returns:
      c: (M, N)
    """
    assert a.is_cuda and b.is_cuda, "Inputs must be CUDA tensors"
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "Only float32 supported in this kernel"

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, f"Incompatible shapes: {a.shape} x {b.shape}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    add_bias_flag = 1 if bias is not None else 0
    bias_ptr = bias if bias is not None else b  # dummy when not used

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    matmul_bias_kernel[grid](
        a, b, bias_ptr, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ADD_BIAS=add_bias_flag,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
    )
    return c


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        """
        Custom multi-layer bidirectional GRU implemented with Triton matmul kernels.
        Interface matches the original Model (nn.GRU-based).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias_flag = bias
        self.batch_first = batch_first
        self.num_directions = 2  # bidirectional=True

        # Parameters:
        # We store weights as:
        #   weight_ih_l[k][dir]: (input_dim, 3*hidden_size)
        #   weight_hh_l[k][dir]: (hidden_size, 3*hidden_size)
        # which is convenient for A(M,K) @ B(K,N) layout.
        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        if self.bias_flag:
            self.bias_ih = nn.ParameterList()
            self.bias_hh = nn.ParameterList()
        else:
            self.bias_ih = None
            self.bias_hh = None

        for layer in range(num_layers):
            if layer == 0:
                in_dim = input_size
            else:
                in_dim = hidden_size * self.num_directions

            for direction in range(self.num_directions):
                w_ih = nn.Parameter(torch.empty(in_dim, 3 * hidden_size))
                w_hh = nn.Parameter(torch.empty(hidden_size, 3 * hidden_size))
                nn.init.xavier_uniform_(w_ih)
                nn.init.orthogonal_(w_hh)

                self.weight_ih.append(w_ih)
                self.weight_hh.append(w_hh)

                if self.bias_flag:
                    b_ih = nn.Parameter(torch.zeros(3 * hidden_size))
                    b_hh = nn.Parameter(torch.zeros(3 * hidden_size))
                    self.bias_ih.append(b_ih)
                    self.bias_hh.append(b_hh)

    def _idx(self, layer, direction):
        return layer * self.num_directions + direction

    def forward(self, x, h0):
        """
        x:  (seq_len, batch, input_size) if batch_first=False,
            (batch, seq_len, input_size) if batch_first=True
        h0: (num_layers * num_directions, batch, hidden_size)
        Returns:
            output: same as nn.GRU(..., bidirectional=True)'s output, but we only return output
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # (T, B, C)

        seq_len, batch_size, _ = x.shape
        assert h0.shape[0] == self.num_layers * self.num_directions
        assert h0.shape[1] == batch_size
        assert h0.shape[2] == self.hidden_size

        layer_input = x  # (T, B, in_dim)

        for layer in range(self.num_layers):
            # Prepare outputs for this layer
            outputs_f = []
            outputs_b_rev = []

            # Forward direction
            idx_f = self._idx(layer, 0)
            w_ih_f = self.weight_ih[idx_f]
            w_hh_f = self.weight_hh[idx_f]
            b_ih_f = self.bias_ih[idx_f] if self.bias_flag else None
            b_hh_f = self.bias_hh[idx_f] if self.bias_flag else None
            h_t_f = h0[idx_f]  # (B, H)

            for t in range(seq_len):
                x_t = layer_input[t]  # (B, in_dim)
                gi = triton_matmul_bias(x_t, w_ih_f, b_ih_f)  # (B, 3H)
                gh = triton_matmul_bias(h_t_f, w_hh_f, b_hh_f)  # (B, 3H)

                i_r, i_z, i_n = gi.chunk(3, dim=1)
                h_r, h_z, h_n = gh.chunk(3, dim=1)

                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)
                h_t_f = (1.0 - z) * n + z * h_t_f

                outputs_f.append(h_t_f)

            # Backward direction
            idx_b = self._idx(layer, 1)
            w_ih_b = self.weight_ih[idx_b]
            w_hh_b = self.weight_hh[idx_b]
            b_ih_b = self.bias_ih[idx_b] if self.bias_flag else None
            b_hh_b = self.bias_hh[idx_b] if self.bias_flag else None
            h_t_b = h0[idx_b]  # (B, H)

            for t in range(seq_len - 1, -1, -1):
                x_t = layer_input[t]  # (B, in_dim)
                gi = triton_matmul_bias(x_t, w_ih_b, b_ih_b)  # (B, 3H)
                gh = triton_matmul_bias(h_t_b, w_hh_b, b_hh_b)  # (B, 3H)

                i_r, i_z, i_n = gi.chunk(3, dim=1)
                h_r, h_z, h_n = gh.chunk(3, dim=1)

                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)
                h_t_b = (1.0 - z) * n + z * h_t_b

                outputs_b_rev.append(h_t_b)

            outputs_b = list(reversed(outputs_b_rev))

            # Concatenate forward and backward outputs for this layer
            layer_outputs = []
            for t in range(seq_len):
                out_t = torch.cat([outputs_f[t], outputs_b[t]], dim=1)  # (B, 2H)
                layer_outputs.append(out_t)

            layer_input = torch.stack(layer_outputs, dim=0)  # (T, B, 2H)

        output = layer_input  # final layer output: (T, B, 2H)

        if self.batch_first:
            output = output.transpose(0, 1)  # (B, T, 2H)

        return output
