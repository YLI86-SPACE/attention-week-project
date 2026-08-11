import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn.functional as F

from src.data import TinyImageNetDataset, load_manifest
from src.train import build_model, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/tiny-imagenet-200"))
    parser.add_argument("--split-json", type=Path, default=Path("splits/split_seed42.json"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.split_json)
    records = sorted(
        manifest["train"],
        key=lambda r: hashlib.sha256(("overfit-v1|42|" + r["path"]).encode("utf-8")).hexdigest(),
    )[:128]
    ds = TinyImageNetDataset(args.data_root, records, "train", deterministic=True)
    xs, ys, _, _ = zip(*(ds[i] for i in range(128)))
    x = torch.stack(xs).to(device)
    y = torch.tensor(ys, device=device)
    cfg = {"model": "minivit", "position": "abspos", "attention": "full", "lambda_rgb": 0, "seed": 42}
    model = build_model(cfg, len(manifest["classes"])).to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    acc = 0.0
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                pred = model(x)[0].argmax(dim=1)
                acc = float((pred == y).float().mean())
            print(json.dumps({"step": step, "loss": float(loss.detach()), "train128_acc": acc}))
            if acc >= 0.95:
                break
    out = {"steps_run": step, "train128_acc": acc, "device": str(device)}
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)
    Path("outputs/experiments/overfit_sanity.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
