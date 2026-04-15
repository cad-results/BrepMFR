"""Graph Encoder Layer for BrepFormer.

Implementation follows the official BrepFormer encoder.py with:
- Pre-LayerNorm (RMSNorm)
- MultiheadAttention with RoPE and attention bias injection
- SwiGLU FFN with double invocation (per paper)
"""

from typing import Optional

import torch
import torch.nn as nn

from brepformer.models.layers.attention import MultiheadAttention
from brepformer.models.layers.blocks import RMSNorm, SwiGLU


class GraphEncoderLayer(nn.Module):
    """Single Transformer encoder layer for graph encoding.

    Architecture (pre-LayerNorm):
        x = x + Attention(RMSNorm(x), attn_bias)
        x = x + FFN(RMSNorm(x))

    Note: Paper uses double FFN invocation which we implement here.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        ffn_dim: int = 512,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        dropout: float = 0.3,
        attention_dropout: float = 0.3,
        activation_dropout: float = 0.3,
        use_rope: bool = False,
        max_seq_len: int = 512,
    ):
        """Initialize GraphEncoderLayer.

        Args:
            hidden_dim: Hidden dimension.
            ffn_dim: FFN intermediate dimension.
            num_heads: Number of attention heads.
            num_kv_heads: Number of KV heads for GQA.
            dropout: Dropout probability.
            attention_dropout: Attention dropout probability.
            activation_dropout: Activation dropout probability.
            use_rope: Whether to use Rotary Position Embeddings.
            max_seq_len: Maximum sequence length for RoPE.
        """
        super().__init__()

        # Pre-norm layers (RMSNorm per paper)
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.norm3 = RMSNorm(hidden_dim)

        # Attention
        self.attention = MultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=attention_dropout,
            use_rope=use_rope,
            max_seq_len=max_seq_len,
        )

        # Two independent FFN blocks (SwiGLU)
        self.ffn1 = SwiGLU(
            in_dim=hidden_dim,
            hidden_dim=ffn_dim,
            out_dim=hidden_dim,
            dropout=activation_dropout,
        )
        self.ffn2 = SwiGLU(
            in_dim=hidden_dim,
            hidden_dim=ffn_dim,
            out_dim=hidden_dim,
            dropout=activation_dropout,
        )

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim).
            attn_bias: Attention bias from GraphAttnBias.
                       Shape: (batch, num_heads, seq_len, seq_len).
            attn_mask: Attention mask for padding.
                       Shape: (batch, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        # Attention block with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attn_bias=attn_bias, attn_mask=attn_mask)
        x = self.dropout(x)
        x = residual + x

        # First FFN block with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn1(x)
        x = residual + x

        # Second FFN block with independent weights
        residual = x
        x = self.norm3(x)
        x = self.ffn2(x)
        x = residual + x

        return x
