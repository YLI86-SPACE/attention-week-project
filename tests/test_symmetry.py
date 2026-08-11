import torch

from src.models.attention import MultiHeadSelfAttention
from src.models.rope import apply_rope_2d
from src.models.window_attention import WindowMultiHeadSelfAttention


def rel_frob(a, b):
    return (a - b).norm() / (b.norm() + 1e-12)


def test_nopos_full_attention_permutation_equivariance():
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        layer = MultiHeadSelfAttention(embed_dim=16, heads=4, dropout=0.0).eval()
        x = torch.randn(2, 9, 16)
        perm = torch.randperm(9)
        y = layer(x)
        yp = layer(x[:, perm])
        torch.testing.assert_close(yp, y[:, perm], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(yp.mean(dim=1), y.mean(dim=1), atol=1e-5, rtol=1e-4)


def test_rope_token_coordinate_pair_permutation_equivariance():
    torch.manual_seed(3)
    layer = MultiHeadSelfAttention(embed_dim=32, heads=4, dropout=0.0).eval()
    x = torch.randn(1, 16, 32)
    coords = torch.tensor([(r, c) for r in range(4) for c in range(4)])
    perm = torch.randperm(16)
    y = layer(x, rope_coords=coords, rope_fn=apply_rope_2d)
    yp = layer(x[:, perm], rope_coords=coords[perm], rope_fn=apply_rope_2d)
    torch.testing.assert_close(yp, y[:, perm], atol=1e-5, rtol=1e-4)


def test_window_attention_window_unit_swap_equivariance():
    torch.manual_seed(4)
    layer = WindowMultiHeadSelfAttention(embed_dim=16, heads=4, window_size=2, dropout=0.0).eval()
    x = torch.randn(1, 4, 4, 16)
    y = layer(x.view(1, 16, 16), (4, 4)).view(1, 4, 4, 16)
    xp = x.clone()
    xp[:, :2, :2], xp[:, 2:, 2:] = x[:, 2:, 2:].clone(), x[:, :2, :2].clone()
    yp = layer(xp.view(1, 16, 16), (4, 4)).view(1, 4, 4, 16)
    expected = y.clone()
    expected[:, :2, :2], expected[:, 2:, 2:] = y[:, 2:, 2:].clone(), y[:, :2, :2].clone()
    torch.testing.assert_close(yp, expected, atol=1e-5, rtol=1e-4)
