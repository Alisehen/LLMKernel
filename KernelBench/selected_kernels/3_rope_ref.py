import torch
import torch.nn as nn

class Model(nn.Module):
    """
    RoPE (Rotary Position Embedding) - PyTorch Reference Implementation

    Rotary Position Embedding applies a rotation to the query and key vectors
    based on their position in the sequence. This allows the model to naturally
    encode relative positions.

    Formula:
        For each position, split the embedding into two halves [x1, x2]
        Apply rotation: [x1*cos - x2*sin, x2*cos + x1*sin]

    Used in: LLaMA, GPT-J, GPT-NeoX, PaLM, and many modern LLMs
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, q, k, cos, sin):
        """
        Apply RoPE to query and key tensors.

        Args:
            q (torch.Tensor): Query tensor of shape (batch, n_heads, seq_len, head_dim)
            k (torch.Tensor): Key tensor of shape (batch, n_heads, seq_len, head_dim)
            cos (torch.Tensor): Cosine values of shape (seq_len, head_dim//2)
            sin (torch.Tensor): Sine values of shape (seq_len, head_dim//2)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - q_rotated: Rotated query (batch, n_heads, seq_len, head_dim)
                - k_rotated: Rotated key (batch, n_heads, seq_len, head_dim)
                - cos: Cosine values (unchanged)
                - sin: Sine values (unchanged)
        """
        # Transpose to (batch, seq_len, n_heads, head_dim) for easier position-wise operation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        batch_size, seq_len, n_heads, head_dim = q.shape
        half_dim = head_dim // 2

        # Split into two halves along head_dim
        q1 = q[..., :half_dim]  # First half
        q2 = q[..., half_dim:]  # Second half
        k1 = k[..., :half_dim]
        k2 = k[..., half_dim:]

        # Reshape cos/sin for broadcasting: (seq_len, head_dim//2) -> (1, seq_len, 1, head_dim//2)
        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(2)

        # Apply rotation transformation
        # RoPE formula: rotate_half([x1, x2]) = [x1*cos - x2*sin, x2*cos + x1*sin]
        q_rotated = torch.cat([
            q1 * cos - q2 * sin,  # New first half
            q2 * cos + q1 * sin   # New second half
        ], dim=-1)

        k_rotated = torch.cat([
            k1 * cos - k2 * sin,
            k2 * cos + k1 * sin
        ], dim=-1)

        # Transpose back to (batch, n_heads, seq_len, head_dim)
        q_rotated = q_rotated.transpose(1, 2)
        k_rotated = k_rotated.transpose(1, 2)

        # Return cos/sin as well to match Triton interface
        return q_rotated, k_rotated, cos.squeeze(0).squeeze(1), sin.squeeze(0).squeeze(1)


BATCH_SIZE = 2
N_HEADS = 8
SEQ_LEN = 4
HEAD_DIM = 16

def get_inputs():
    """
    Generate test inputs for RoPE.

    Returns:
        List containing [q, k, cos, sin]:
            - q: Query tensor (batch, n_heads, seq_len, head_dim)
            - k: Key tensor (batch, n_heads, seq_len, head_dim)
            - cos: Cosine values (seq_len, head_dim//2)
            - sin: Sine values (seq_len, head_dim//2)
    """
    q = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
    k = torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
    cos = torch.randn(SEQ_LEN, HEAD_DIM // 2, dtype=torch.float32)
    sin = torch.randn(SEQ_LEN, HEAD_DIM // 2, dtype=torch.float32)
    return [q, k, cos, sin]

def get_init_inputs():
    """
    Get initialization parameters for Model.

    Returns:
        Empty list (no initialization parameters needed)
    """
    return []
