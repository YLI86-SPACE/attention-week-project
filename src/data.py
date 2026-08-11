import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _posix(path: Path) -> str:
    return path.as_posix()


def choose_classes(data_root: Path, config: str, seed: int = 42) -> List[str]:
    wnids = sorted(_read_lines(data_root / "wnids.txt"))
    if config == "standard":
        return wnids
    if config == "lite":
        return sorted(random.Random(seed).sample(wnids, 100))
    raise ValueError("config must be standard or lite")


def _split_train_class(data_root: Path, wnid: str, seed: int):
    image_dir = data_root / "train" / wnid / "images"
    records = []
    for path in image_dir.glob("*.JPEG"):
        rel = _posix(path.relative_to(data_root))
        digest = hashlib.sha256(f"split-v1|{seed}|{rel}".encode("utf-8")).hexdigest()
        records.append((digest, rel))
    records.sort(key=lambda x: (x[0], x[1]))
    if len(records) != 500:
        raise ValueError(f"Expected 500 train images for {wnid}, got {len(records)}")
    return records[:450], records[450:]


def _read_val_annotations(data_root: Path) -> Dict[str, str]:
    mapping = {}
    for line in _read_lines(data_root / "val" / "val_annotations.txt"):
        parts = line.split("\t")
        mapping[parts[0]] = parts[1]
    return mapping


def build_manifest(data_root: Path, config: str, seed: int = 42) -> dict:
    classes = choose_classes(data_root, config=config, seed=seed)
    class_to_idx = {wnid: i for i, wnid in enumerate(classes)}
    train_records = []
    validation_records = []
    for wnid in classes:
        train_split, val_split = _split_train_class(data_root, wnid, seed)
        class_id = class_to_idx[wnid]
        for _, rel in train_split:
            train_records.append({"path": rel, "wnid": wnid, "class_id": class_id})
        for _, rel in val_split:
            validation_records.append({"path": rel, "wnid": wnid, "class_id": class_id})
    test_records = []
    for image_name, wnid in _read_val_annotations(data_root).items():
        if wnid in class_to_idx:
            test_records.append({"path": f"val/images/{image_name}", "wnid": wnid, "class_id": class_to_idx[wnid]})
    key = lambda r: (r["class_id"], r["path"])
    train_records.sort(key=key)
    validation_records.sort(key=key)
    test_records.sort(key=key)
    return {
        "seed": seed,
        "data_configuration": config,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }


def manifest_digest(manifest: dict) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_split_files(manifest: dict, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = manifest_digest(manifest)
    seed = manifest["seed"]
    (out_dir / f"classes_seed{seed}.txt").write_text("\n".join(manifest["classes"]) + "\n", encoding="utf-8")
    (out_dir / f"class_to_idx_seed{seed}.json").write_text(json.dumps(manifest["class_to_idx"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / f"split_seed{seed}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / f"split_seed{seed}.sha256").write_text(digest, encoding="utf-8")
    return digest


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def imagenet_normalize():
    return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def make_transform(split: str, deterministic: bool = False):
    ops = []
    if split == "train" and not deterministic:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend([transforms.ToTensor(), imagenet_normalize()])
    return transforms.Compose(ops)


class TinyImageNetDataset(Dataset):
    def __init__(self, data_root: Path, records: List[dict], split: str, deterministic: bool = False) -> None:
        self.data_root = Path(data_root)
        self.records = records
        self.transform = make_transform(split, deterministic=deterministic)
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(self.data_root / record["path"]).convert("RGB")
        base_tensor = self.to_tensor(image)
        rgb_mean = base_tensor.mean(dim=(1, 2))
        return self.transform(image), int(record["class_id"]), rgb_mean, record["path"]


def validate_manifest(manifest: dict) -> dict:
    class_ids = set(manifest["class_to_idx"].values())
    out = {"classes": len(manifest["classes"])}
    for split in ("train", "validation", "test"):
        paths = [r["path"] for r in manifest[split]]
        if len(paths) != len(set(paths)):
            raise ValueError(f"{split} contains duplicate paths")
        counts = {}
        for r in manifest[split]:
            if r["class_id"] not in class_ids:
                raise ValueError(f"{split} has invalid class_id")
            counts[r["class_id"]] = counts.get(r["class_id"], 0) + 1
        out[split] = len(paths)
        out[f"{split}_min_per_class"] = min(counts.values())
        out[f"{split}_max_per_class"] = max(counts.values())
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(r["path"] for r in manifest[a]) & set(r["path"] for r in manifest[b]):
            raise ValueError(f"{a} and {b} overlap")
    out["sha256"] = manifest_digest(manifest)
    return out


def make_loaders(data_root: Path, split_json: Path, batch_size: int, num_workers: int = 2, seed: int = 42, deterministic_train: bool = False):
    manifest = load_manifest(split_json)
    generator = torch.Generator().manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": torch.cuda.is_available()}
    train = TinyImageNetDataset(data_root, manifest["train"], "train", deterministic_train)
    val = TinyImageNetDataset(data_root, manifest["validation"], "validation", True)
    test = TinyImageNetDataset(data_root, manifest["test"], "test", True)
    return (
        DataLoader(train, shuffle=True, drop_last=True, generator=generator, **common),
        DataLoader(val, shuffle=False, drop_last=False, **common),
        DataLoader(test, shuffle=False, drop_last=False, **common),
        manifest,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", choices=["standard", "lite"], default="lite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("splits"))
    args = parser.parse_args()
    manifest = build_manifest(args.data_root, args.config, args.seed)
    digest = write_split_files(manifest, args.out_dir)
    print(json.dumps({**validate_manifest(manifest), "data_configuration": args.config, "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
