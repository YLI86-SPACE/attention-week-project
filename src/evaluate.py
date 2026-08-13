import argparse
import csv
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import yaml

from .data import make_loaders
from .train import build_model, evaluate, save_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/tiny-imagenet-200")
    parser.add_argument("--split-json", default="splits/split_seed42.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, manifest = make_loaders(
        Path(args.data_root), Path(args.split_json), args.batch_size, cfg.get("num_workers", 2), cfg.get("seed", 42), True
    )
    model = build_model(cfg, len(manifest["classes"])).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    lambda_rgb = float(cfg.get("lambda_rgb", 0.0))
    metrics = evaluate(model, test_loader, device, lambda_rgb=lambda_rgb)
    result = {"config": args.config, "checkpoint": args.checkpoint, "test": metrics, "device": str(device)}
    out_json = Path(args.output_json) if args.output_json else Path("outputs/experiments") / f"evaluate_{cfg.get('run_id', 'run')}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pred_path = Path(args.predictions) if args.predictions else Path("outputs/predictions") / f"{cfg.get('run_id', 'run')}_eval.csv.gz"
    save_predictions(model, test_loader, device, pred_path, lambda_rgb=lambda_rgb)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
