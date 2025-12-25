import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):
    """
    Flash Attention - Memory-efficient attention mechanism
    Computes scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Performs scaled dot-product attention.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch, n_heads, seq_len, head_dim)
            K (torch.Tensor): Key tensor of shape (batch, n_heads, seq_len, head_dim)
            V (torch.Tensor): Value tensor of shape (batch, n_heads, seq_len, head_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch, n_heads, seq_len, head_dim)
        """
        batch, n_heads, seq_len, head_dim = Q.shape
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        # Multiply by V
        output = torch.matmul(attn_weights, V)

        return output

BATCH = 2
N_HEADS = 2
SEQ_LEN = 128
HEAD_DIM = 64

def get_inputs():
    Q = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)
    K = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)
    V = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM)
    return [Q, K, V]

def get_init_inputs():
    return []
