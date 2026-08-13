import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.data import load_manifest


def write_samples(manifest: dict, out_path: Path, seed: int = 42):
    records = sorted(
        manifest["validation"],
        key=lambda r: (hashlib.sha256((f"perturb-v1|{seed}|" + r["path"]).encode("utf-8")).hexdigest(), r["path"]),
    )[:64]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(r["path"] for r in records) + "\n", encoding="utf-8")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", type=Path, default=Path("splits/split_seed42.json"))
    parser.add_argument("--samples", type=Path, default=Path("splits/perturbation_samples_seed42.txt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/perturbation.json"))
    args = parser.parse_args()
    manifest = load_manifest(args.split_json)
    records = write_samples(manifest, args.samples)
    result = {
        "status": "sample_manifest_created",
        "sample_count": len(records),
        "sample_file": args.samples.as_posix(),
        "patch_coord": [6, 6],
        "replacement": "train split RGB channel mean, computed before normalization",
        "note": "Full perturbation metrics require trained R1-R4 best-validation checkpoints. This file records the deterministic sample set now."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
