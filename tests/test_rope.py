import math

import torch

from src.models.rope import apply_rope_1d, apply_rope_2d


def explicit_rotate(x: torch.Tensor, position: int, base: float = 10000.0) -> torch.Tensor:
    d = x.shape[-1]
    out = x.clone()
    for i in range(d // 2):
        theta = base ** (-2 * i / d)
        angle = position * theta
        c = math.cos(angle)
        s = math.sin(angle)
        a = x[..., 2 * i]
        b = x[..., 2 * i + 1]
        out[..., 2 * i] = a * c - b * s
        out[..., 2 * i + 1] = a * s + b * c
    return out


def test_rope_1d_matches_explicit_rotation_and_preserves_norm():
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        q = torch.randn(1, 1, 5, 8)
        k = torch.randn(1, 1, 5, 8)
        pos = torch.arange(5)
        q_rot, k_rot = apply_rope_1d(q, k, pos)
        for n in range(5):
            torch.testing.assert_close(q_rot[0, 0, n], explicit_rotate(q[0, 0, n], n))
            torch.testing.assert_close(k_rot[0, 0, n], explicit_rotate(k[0, 0, n], n))
        torch.testing.assert_close(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-6, rtol=1e-6)


def test_rope_relative_dot_product_identity_with_common_shift():
    torch.manual_seed(3)
    q = torch.randn(8)
    k = torch.randn(8)
    for m, n, shift in [(1, 4, 0), (1, 4, 7), (3, 2, 5)]:
        lhs = (explicit_rotate(q, m + shift) * explicit_rotate(k, n + shift)).sum()
        rhs = (q * explicit_rotate(k, n - m)).sum()
        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)


def test_rope_2d_same_delta_same_rotated_dot():
    torch.manual_seed(4)
    q = torch.randn(1, 1, 2, 32)
    k = torch.randn(1, 1, 2, 32)
    q[:, :, 1] = q[:, :, 0]
    k[:, :, 1] = k[:, :, 0]
    coords_a = torch.tensor([[2, 3], [5, 8]])
    coords_b = torch.tensor([[0, 1], [3, 6]])
    qa, ka = apply_rope_2d(q, k, coords_a)
    qb, kb = apply_rope_2d(q, k, coords_b)
    dot_a = (qa[:, :, 0] * ka[:, :, 1]).sum()
    dot_b = (qb[:, :, 0] * kb[:, :, 1]).sum()
    torch.testing.assert_close(dot_a, dot_b, atol=1e-5, rtol=1e-5)


def test_rope_invalid_dimensions_raise():
    q = torch.randn(1, 1, 2, 10)
    k = torch.randn(1, 1, 2, 10)
    coords = torch.zeros(2, 2, dtype=torch.long)
    try:
        apply_rope_2d(q, k, coords)
    except ValueError as exc:
        assert "divisible by 4" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
