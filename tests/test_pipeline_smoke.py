import torch

from src.train import build_model, forward_loss


def test_pipeline_smoke_checkpoint_round_trip():
    torch.manual_seed(0)
    cfg = {"model": "minivit", "position": "abspos", "attention": "full", "lambda_rgb": 10, "seed": 42}
    model = build_model(cfg, num_classes=10)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(2, 3, 64, 64)
    y = torch.tensor([1, 3])
    rgb = torch.rand(2, 3)
    model.train()
    loss, _, logits, rgb_pred = forward_loss(model, x, y, rgb, lambda_rgb=10.0)
    assert logits.shape == (2, 10)
    assert rgb_pred.shape == (2, 3)
    loss.backward()
    opt.step()
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    reloaded = build_model(cfg, num_classes=10)
    reloaded.load_state_dict(snapshot)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        a = model(x)
        b = reloaded(x)
    torch.testing.assert_close(a[0], b[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a[1], b[1], atol=1e-6, rtol=1e-6)
