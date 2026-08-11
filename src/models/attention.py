import math
from typing import Optional, Tuple

import torch
from torch import nn


def split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """[B, N, D] -> [B, H, N, d_h]."""
    if x.ndim != 3:
        raise ValueError(f"Expected [B, N, D], got shape {tuple(x.shape)}")
    b, n, d = x.shape
    if d % heads != 0:
        raise ValueError(f"Embedding dim {d} must be divisible by heads {heads}")
    return x.view(b, n, heads, d // heads).transpose(1, 2).contiguous()


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, H, N, d_h] -> [B, N, D]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B, H, N, d_h], got shape {tuple(x.shape)}")
    b, h, n, dh = x.shape
    return x.transpose(1, 2).contiguous().view(b, n, h * dh)


def attention_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hand-written scaled dot-product attention.

    q/k/v use shape [B, H, N, d_h]. The optional additive mask is broadcastable
    to [B, H, N, N] and should contain 0 for allowed positions and a large
    negative value such as -inf for blocked positions.
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"q, k, v must have identical shapes, got {q.shape}, {k.shape}, {v.shape}"
        )
    dh = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
    if attn_mask is not None:
        scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)
    attn = torch.softmax(scores, dim=-1)
    attn = torch.dropout(attn, p=dropout_p, train=training)
    out = torch.matmul(attn, v)
    return out, attn


class MultiHeadSelfAttention(nn.Module):
    """Custom MHA module; does not use nn.MultiheadAttention."""

    def __init__(
        self,
        embed_dim: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % heads != 0:
            raise ValueError("embed_dim must be divisible by heads")
        self.embed_dim = embed_dim
        self.heads = heads
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rope_coords: Optional[torch.Tensor] = None,
        rope_fn=None,
        need_weights: bool = False,
    ):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (split_heads(t, self.heads) for t in qkv)
        if rope_fn is not None:
            if rope_coords is None:
                raise ValueError("rope_coords is required when rope_fn is provided")
            q, k = rope_fn(q, k, rope_coords)
        out, attn = attention_core(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        out = self.proj(merge_heads(out))
        if need_weights:
            return out, attn
        return out
