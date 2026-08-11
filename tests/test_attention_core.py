import torch
import torch.nn.functional as F

from src.models.attention import attention_core, merge_heads, split_heads


def test_split_merge_round_trip():
    x = torch.randn(2, 5, 12)
    y = merge_heads(split_heads(x, heads=3))
    torch.testing.assert_close(y, x)


def test_attention_core_matches_sdpa_forward_and_backward():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 7, 4, requires_grad=True)
    k = torch.randn(2, 3, 7, 4, requires_grad=True)
    v = torch.randn(2, 3, 7, 4, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out, attn = attention_core(q, k, v, dropout_p=0.0, training=False)
    ref = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        dropout_p=0.0,
        is_causal=False,
    )
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))
    assert torch.isfinite(out).all()

    g = torch.randn_like(out)
    (out * g).sum().backward()
    (ref * g).sum().backward()
    torch.testing.assert_close(q.grad, q_ref.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(k.grad, k_ref.grad, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(v.grad, v_ref.grad, atol=1e-5, rtol=1e-4)


def test_attention_additive_mask():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., -1] = float("-inf")
    _, attn = attention_core(q, k, v, attn_mask=mask)
    assert torch.all(attn[..., -1] < 1e-6)
    torch.testing.assert_close(attn[..., :3].sum(dim=-1), torch.ones(1, 2, 4))
