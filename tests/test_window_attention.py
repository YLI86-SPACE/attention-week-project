import torch

from src.models.attention import MultiHeadSelfAttention
from src.models.window_attention import WindowMultiHeadSelfAttention, window_partition, window_reverse


def test_window_partition_reverse_round_trip():
    x = torch.randn(2, 16, 16, 8)
    windows = window_partition(x, window_size=4)
    y = window_reverse(windows, window_size=4, grid_size=(16, 16), batch_size=2)
    torch.testing.assert_close(y, x)


def test_window_partition_shapes():
    x = torch.randn(3, 8, 8, 5)
    windows = window_partition(x, window_size=4)
    assert windows.shape == (12, 16, 5)


def test_window_equals_full_when_window_covers_grid():
    torch.manual_seed(0)
    full = MultiHeadSelfAttention(embed_dim=16, heads=4, dropout=0.0).eval()
    win = WindowMultiHeadSelfAttention(embed_dim=16, heads=4, window_size=4, dropout=0.0).eval()
    win.mha.load_state_dict(full.state_dict())
    x = torch.randn(2, 16, 16)
    y_full = full(x)
    y_win = win(x, (4, 4))
    torch.testing.assert_close(y_win, y_full, atol=1e-5, rtol=1e-4)
