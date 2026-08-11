import argparse
import csv
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from PIL import Image
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.data import load_manifest, manifest_digest


def read_words(data_root: Path):
    words = {}
    for line in (data_root / "words.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        wnid, names = line.split("\t", 1)
        words[wnid] = names.split(",")[0].strip()
    return words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/tiny-imagenet-200"))
    parser.add_argument("--split-json", type=Path, default=Path("splits/split_seed42.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = load_manifest(args.split_json)
    words = read_words(args.data_root)
    prompts = [f"a photo of a {words.get(wnid, wnid)}" for wnid in manifest["classes"]]
    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    print({k: list(v.shape) for k, v in text_inputs.items()})
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    correct = 0; rows = []; sim_shape = None
    records = manifest["test"]
    for start in tqdm(range(0, len(records), args.batch_size)):
        batch = records[start:start + args.batch_size]
        images = [Image.open(args.data_root / r["path"]).convert("RGB") for r in batch]
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        if start == 0:
            print({k: list(v.shape) for k, v in image_inputs.items()})
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = image_features @ text_features.t()
        sim_shape = list(sims.shape)
        pred = sims.argmax(dim=1).cpu().tolist()
        for r, p in zip(batch, pred):
            ok = int(p == r["class_id"]); correct += ok
            rows.append({"relative_path": r["path"], "target_id": r["class_id"], "predicted_id": p, "correct": ok})
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    result = {"model": args.model, "prompt": "a photo of a <class_name>", "split_sha256": manifest_digest(manifest), "class_count": len(manifest["classes"]), "sample_count": len(records), "correct_count": correct, "accuracy": correct / len(records), "similarity_matrix_shape_last_batch": sim_shape}
    Path("outputs/experiments/clip_zero_shot.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with gzip.open("outputs/predictions/clip_zero_shot.csv.gz", "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "target_id", "predicted_id", "correct"])
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
