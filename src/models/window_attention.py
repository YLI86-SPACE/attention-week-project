from typing import Tuple

import torch
from torch import nn

from .attention import MultiHeadSelfAttention


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """[B, H, W, C] -> [B*num_windows, window_size*window_size, C]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B, H, W, C], got {tuple(x.shape)}")
    b, h, w, c = x.shape
    if h % window_size != 0 or w % window_size != 0:
        raise ValueError("H and W must be divisible by window_size")
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, window_size * window_size, c)


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    grid_size: Tuple[int, int],
    batch_size: int,
) -> torch.Tensor:
    """[B*num_windows, M, C] -> [B, H, W, C]."""
    h, w = grid_size
    c = windows.shape[-1]
    expected = batch_size * (h // window_size) * (w // window_size)
    if windows.shape[0] != expected:
        raise ValueError(f"Expected {expected} windows, got {windows.shape[0]}")
    x = windows.view(
        batch_size,
        h // window_size,
        w // window_size,
        window_size,
        window_size,
        c,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, h, w, c)


class WindowMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        heads: int = 4,
        window_size: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.mha = MultiHeadSelfAttention(embed_dim, heads, dropout)

    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int], need_weights: bool = False):
        b, n, c = x.shape
        h, w = grid_size
        if n != h * w:
            raise ValueError(f"Token count {n} does not match grid {grid_size}")
        x_grid = x.view(b, h, w, c)
        windows = window_partition(x_grid, self.window_size)
        if need_weights:
            out_windows, attn = self.mha(windows, need_weights=True)
        else:
            out_windows = self.mha(windows)
            attn = None
        out_grid = window_reverse(out_windows, self.window_size, grid_size, b)
        out = out_grid.view(b, n, c)
        if need_weights:
            return out, attn
        return out
