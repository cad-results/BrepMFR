"""Multi-head Attention with Rotary Position Embeddings (RoPE).

Implementation follows the official BrepFormer attention.py with:
- Rotary Position Embeddings (RoPE)
- Grouped Query Attention (GQA) support
- Attention bias injection for edge/topology features
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    Args:
        dim: Dimension of the embedding (head_dim).
        max_seq_len: Maximum sequence length.
        theta: Base for the exponential.

    Returns:
        Tensor of shape (max_seq_len, dim // 2, 2) containing cos and sin values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.stack([freqs.cos(), freqs.sin()], dim=-1)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting with x.

    Args:
        freqs_cis: Frequency tensor of shape (seq_len, head_dim // 2, 2).
        x: Input tensor of shape (batch, num_heads, seq_len, head_dim).

    Returns:
        Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert ndim >= 2
    seq_len = x.shape[2]
    freqs_cis = freqs_cis[:seq_len]
    shape = [1, 1, seq_len, x.shape[-1] // 2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors using the given frequency tensor.

    Args:
        xq: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        xk: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        freqs_cis: Precomputed frequency tensor.

    Returns:
        Tuple of modified query and key tensors.
    """
    # Reshape x to (batch, num_heads, seq_len, head_dim // 2, 2)
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Reshape freqs_cis for broadcasting
    freqs_cis_q = reshape_for_broadcast(freqs_cis, xq)
    freqs_cis_k = reshape_for_broadcast(freqs_cis, xk)

    # Apply rotation using complex multiplication
    # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    xq_out_r = xq_r[..., 0] * freqs_cis_q[..., 0] - xq_r[..., 1] * freqs_cis_q[..., 1]
    xq_out_i = xq_r[..., 0] * freqs_cis_q[..., 1] + xq_r[..., 1] * freqs_cis_q[..., 0]
    xk_out_r = xk_r[..., 0] * freqs_cis_k[..., 0] - xk_r[..., 1] * freqs_cis_k[..., 1]
    xk_out_i = xk_r[..., 0] * freqs_cis_k[..., 1] + xk_r[..., 1] * freqs_cis_k[..., 0]

    # Stack and reshape back
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).reshape(*xq.shape)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).reshape(*xk.shape)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads.

    Args:
        x: Tensor of shape (batch, num_kv_heads, seq_len, head_dim).
        n_rep: Number of times to repeat each head.

    Returns:
        Tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return x
    batch, num_kv_heads, seq_len, head_dim = x.shape
    x = x.unsqueeze(2).expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class MultiheadAttention(nn.Module):
    """Multi-head Attention with RoPE and Grouped Query Attention.

    Implements attention with:
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (num_kv_heads < num_heads)
    - Attention bias injection from edge/topology features
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        max_seq_len: int = 512,
    ):
        """Initialize MultiheadAttention.

        Args:
            hidden_dim: Hidden dimension of the model.
            num_heads: Number of attention heads for queries.
            num_kv_heads: Number of attention heads for keys/values (for GQA).
            dropout: Dropout probability for attention weights.
            use_rope: Whether to apply Rotary Position Embeddings.
            max_seq_len: Maximum sequence length for RoPE.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.use_rope = use_rope

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        # Projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # Precompute RoPE frequencies (only used when use_rope=True)
        if self.use_rope:
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(self.head_dim, max_seq_len),
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            attn_bias: Attention bias from edge/topology features.
                       Shape: (batch, num_heads, seq_len, seq_len).
            attn_mask: Attention mask for padding.
                       Shape: (batch, seq_len) or (batch, 1, 1, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE (optional; disabled by default for graph nodes with no sequential order)
        if self.use_rope:
            q, k = apply_rotary_emb(q, k, self.freqs_cis.to(x.device))

        # Repeat KV for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add attention bias (KEY FEATURE: edge/topology features)
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias

        # Apply attention mask (for padding)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))

        # Softmax with proper handling of fully masked rows
        # Find rows where all values are -inf (query positions that are padded)
        all_masked = torch.isinf(attn_weights).all(dim=-1, keepdim=True)  # (batch, heads, seq, 1)

        # Replace -inf with 0 for fully masked rows before softmax to avoid NaN
        # Use in-place to avoid allocating a full copy of attn_weights
        attn_weights.masked_fill_(all_masked.expand_as(attn_weights), 0.0)

        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Zero out the attention weights for fully masked query positions
        attn_weights.masked_fill_(all_masked, 0.0)
        del all_masked

        attn_weights = self.dropout(attn_weights)

        # Compute output
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        return output
