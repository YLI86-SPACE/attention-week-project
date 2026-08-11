import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

from src.models.cnn import SmallCNN
from src.models.patch_lstm import PatchLSTM
from src.models.vit import MiniViT


def shape(x):
    return list(x.shape) if torch.is_tensor(x) else str(type(x))


def summarize(name, model, x):
    model.eval()
    with torch.no_grad():
        feat, inter = model.forward_features(x, return_intermediates=True)
        out = model(x)
    row = {"model": name, "model_input": shape(x), "pooled_feature": shape(feat)}
    if name == "SmallCNN":
        row.update({"first_representation": shape(inter["block1"]), "intermediate_representation": shape(inter["block3"]), "output": shape(out)})
    elif name == "Patch-LSTM":
        row.update({"first_representation": shape(inter["patches"]), "intermediate_representation": shape(inter["lstm_outputs"]), "output": shape(out), "patch_index_to_coord_first_last": [PatchLSTM.patch_index_to_coord()[0], PatchLSTM.patch_index_to_coord()[-1]]})
    else:
        logits, rgb = out
        block0 = inter["block0"]["block_output"]
        row.update({"first_representation": shape(inter["patch_tokens"]), "intermediate_representation": shape(block0), "output": {"class": shape(logits), "rgb": shape(rgb)}})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(42)
    x = torch.randn(2, 3, 64, 64)
    rows = [
        summarize("SmallCNN", SmallCNN(args.classes), x),
        summarize("Patch-LSTM", PatchLSTM(args.classes), x),
        summarize("MiniViT-Full", MiniViT(args.classes, position="abspos", attention="full"), x),
        summarize("MiniViT-Window", MiniViT(args.classes, position="abspos", attention="window"), x),
    ]
    out = Path("outputs/experiments"); out.mkdir(parents=True, exist_ok=True)
    (out / "implementation_checks.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
