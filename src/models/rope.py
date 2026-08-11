from typing import Tuple

import torch


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


def _rope_cos_sin(
    positions: torch.Tensor,
    dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    pos = positions.to(device=device, dtype=torch.float32)
    inv_freq = base ** (
        -torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    )
    angles = pos.unsqueeze(-1) * inv_freq
    return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def _expand_pos_for_qk(positions: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if positions.ndim == 1:
        return positions.view(1, 1, -1)
    if positions.ndim == 2:
        return positions.unsqueeze(1)
    raise ValueError("positions must have shape [N] or [B, N]")


def apply_rope_1d(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 1D RoPE to Q/K with shape [B, H, N, d_h]."""
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")
    if q.ndim != 4:
        raise ValueError(f"Expected [B, H, N, d_h], got {tuple(q.shape)}")
    dh = q.shape[-1]
    if dh % 2 != 0:
        raise ValueError(f"Head dimension must be even for 1D RoPE, got {dh}")
    pos = _expand_pos_for_qk(positions, q)
    cos, sin = _rope_cos_sin(pos, dh, base, q.device, q.dtype)
    return _rotate_pairs(q, cos, sin), _rotate_pairs(k, cos, sin)


def apply_rope_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    coords: torch.Tensor,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D axial RoPE for q/k with shape [B, H, N, d_h].

    The first half of each head uses row coordinates and the second half uses
    column coordinates. Frequencies are computed with the axis dimension, not
    the full head dimension.
    """
    if q.shape != k.shape:
        raise ValueError("q and k must have the same shape")
    if q.ndim != 4:
        raise ValueError(f"Expected [B, H, N, d_h], got {tuple(q.shape)}")
    dh = q.shape[-1]
    if dh % 4 != 0:
        raise ValueError(f"Head dimension must be divisible by 4 for 2D RoPE, got {dh}")
    if coords.shape[-1] != 2:
        raise ValueError("coords must end with [row, col]")

    axis_dim = dh // 2
    q_row, q_col = q[..., :axis_dim], q[..., axis_dim:]
    k_row, k_col = k[..., :axis_dim], k[..., axis_dim:]

    rows = coords[..., 0]
    cols = coords[..., 1]
    q_row, k_row = apply_rope_1d(q_row, k_row, rows, base=base)
    q_col, k_col = apply_rope_1d(q_col, k_col, cols, base=base)
    return torch.cat((q_row, q_col), dim=-1), torch.cat((k_row, k_col), dim=-1)
