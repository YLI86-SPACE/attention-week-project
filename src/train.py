import argparse
import csv
import gzip
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from .data import make_loaders, validate_manifest
from .models.cnn import SmallCNN
from .models.patch_lstm import PatchLSTM
from .models.vit import MiniViT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model(cfg: dict, num_classes: int):
    name = cfg["model"]
    if name in ("cnn", "small_cnn"):
        return SmallCNN(num_classes)
    if name in ("lstm", "patch_lstm"):
        return PatchLSTM(num_classes)
    if name == "minivit":
        return MiniViT(
            num_classes=num_classes,
            position=cfg.get("position", "abspos"),
            attention=cfg.get("attention", "full"),
        )
    raise ValueError(f"unknown model {name}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def lr_at_step(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def unpack(batch, device):
    x, y, rgb, paths = batch
    return x.to(device), y.to(device), rgb.to(device), paths


def forward_loss(model, x, y, rgb, lambda_rgb: float):
    out = model(x)
    if isinstance(out, tuple):
        logits, rgb_pred = out
    else:
        logits, rgb_pred = out, None
    ce = F.cross_entropy(logits, y)
    if lambda_rgb == 0 or rgb_pred is None:
        return ce, {"ce": ce.detach(), "mse": None, "weighted_mse": None, "total": ce.detach()}, logits, rgb_pred
    mse = F.mse_loss(rgb_pred, rgb)
    total = ce + lambda_rgb * mse
    return total, {"ce": ce.detach(), "mse": mse.detach(), "weighted_mse": (lambda_rgb * mse).detach(), "total": total.detach()}, logits, rgb_pred


@torch.no_grad()
def evaluate(model, loader, device, lambda_rgb: float = 0.0):
    model.eval()
    total_loss = total_ce = total_mse = 0.0
    correct = n = 0
    for batch in loader:
        x, y, rgb, _ = unpack(batch, device)
        loss, parts, logits, rgb_pred = forward_loss(model, x, y, rgb, lambda_rgb)
        bs = y.numel()
        total_loss += float(loss) * bs
        total_ce += float(parts["ce"]) * bs
        if parts["mse"] is not None:
            total_mse += float(parts["mse"]) * bs
        correct += int((logits.argmax(dim=1) == y).sum())
        n += bs
    return {"loss": total_loss / n, "ce": total_ce / n, "accuracy": correct / n, "rgb_mse": (total_mse / n if lambda_rgb else None), "n": n}


@torch.no_grad()
def save_predictions(model, loader, device, path: Path, lambda_rgb: float = 0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        fieldnames = ["relative_path", "target_id", "predicted_id", "correct"]
        if lambda_rgb:
            fieldnames += ["rgb_r", "rgb_g", "rgb_b"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for batch in loader:
            x, y, _, paths = unpack(batch, device)
            out = model(x)
            logits, rgb_pred = out if isinstance(out, tuple) else (out, None)
            pred = logits.argmax(dim=1)
            for i, rel in enumerate(paths):
                row = {"relative_path": rel, "target_id": int(y[i]), "predicted_id": int(pred[i]), "correct": int(pred[i] == y[i])}
                if lambda_rgb and rgb_pred is not None:
                    row.update({"rgb_r": float(rgb_pred[i, 0]), "rgb_g": float(rgb_pred[i, 1]), "rgb_b": float(rgb_pred[i, 2])})
                writer.writerow(row)


def train(cfg: dict, args):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effective_bs = int(cfg.get("effective_batch_size", 64))
    micro_bs = int(cfg.get("micro_batch_size", 16))
    accum = effective_bs // micro_bs
    train_loader, val_loader, test_loader, manifest = make_loaders(Path(args.data_root), Path(args.split_json), micro_bs, cfg.get("num_workers", 2), cfg.get("seed", 42), args.overfit_128)
    if args.overfit_128:
        subset_batches = []
        seen = 0
        for batch in train_loader:
            subset_batches.append(batch)
            seen += len(batch[1])
            if seen >= 128:
                break
        train_iterable = subset_batches
        steps_per_epoch = max(1, len(subset_batches) // accum)
        epochs = int(args.overfit_epochs)
    else:
        train_iterable = train_loader
        steps_per_epoch = len(train_loader) // accum
        epochs = int(cfg.get("epochs", 20))
    model = build_model(cfg, len(manifest["classes"])).to(device)
    if args.dropout_zero:
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", cfg.get("lr", 3e-4)), weight_decay=cfg.get("weight_decay", 0.05))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and bool(cfg.get("amp", True))))
    lambda_rgb = float(cfg.get("lambda_rgb", 0.0))
    out_dir = Path(args.output_dir) / cfg.get("run_id", "run")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (out_dir / "data_checks.json").write_text(json.dumps(validate_manifest(manifest), indent=2) + "\n", encoding="utf-8")
    history = []
    best = None
    global_step = 0
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_ce = train_mse = train_total = count = 0.0
        used_micro = 0
        for batch_idx, batch in enumerate(tqdm(train_iterable, desc=f"epoch {epoch}", leave=False)):
            if used_micro >= steps_per_epoch * accum:
                break
            x, y, rgb, _ = unpack(batch, device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and bool(cfg.get("amp", True)))):
                loss, parts, _, _ = forward_loss(model, x, y, rgb, lambda_rgb)
                scaled_loss = loss / accum
            scaler.scale(scaled_loss).backward()
            bs = y.numel()
            train_ce += float(parts["ce"]) * bs
            train_total += float(parts["total"]) * bs
            if parts["mse"] is not None:
                train_mse += float(parts["mse"]) * bs
            count += bs
            used_micro += 1
            if used_micro % accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip_norm", cfg.get("grad_clip", 1.0)))
                set_lr(optimizer, lr_at_step(global_step, warmup_steps, total_steps, cfg.get("learning_rate", cfg.get("lr", 3e-4))))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        val = evaluate(model, val_loader, device, lambda_rgb)
        row = {"epoch": epoch, "train_ce": train_ce / count, "train_total": train_total / count, "train_rgb_mse": (train_mse / count if lambda_rgb else None), "val_loss": val["loss"], "val_ce": val["ce"], "val_acc": val["accuracy"], "val_rgb_mse": val["rgb_mse"], "lr": optimizer.param_groups[0]["lr"], "epoch_time_sec": time.perf_counter() - t0}
        history.append(row)
        better = best is None or row["val_acc"] > best["val_acc"] or (row["val_acc"] == best["val_acc"] and row["val_ce"] < best["val_ce"])
        if better:
            best = row.copy()
            torch.save({"model": model.state_dict(), "config": cfg, "best": best}, out_dir / "best.pt")
        print(json.dumps(row, ensure_ascii=False))
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader(); writer.writerows(history)
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device, lambda_rgb)
    final = {"run_id": cfg.get("run_id"), "params": count_params(model), "best_epoch": best["epoch"], "best_val_acc": best["val_acc"], "test": test, "device": str(device), "micro_batch_size": micro_bs, "accumulation_steps": accum, "optimizer_steps_per_epoch": steps_per_epoch}
    (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    save_predictions(model, test_loader, device, Path("outputs/predictions") / f"{cfg.get('run_id', 'run')}.csv.gz", lambda_rgb)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--split-json", default="splits/split_seed42.json")
    parser.add_argument("--output-dir", default="outputs/runs")
    parser.add_argument("--overfit-128", action="store_true")
    parser.add_argument("--overfit-epochs", type=int, default=30)
    parser.add_argument("--dropout-zero", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train(cfg, args)


if __name__ == "__main__":
    main()
