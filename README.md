# Attention Week Project

Tiny ImageNet review project implementing SmallCNN, Patch-LSTM, hand-written Multi-Head Self-Attention, 2D Axial RoPE, and Window Attention.

## Current Status

Implemented and locally verified:

- deterministic Tiny ImageNet lite split generation
- `SmallCNN`, `PatchLSTM`, `MiniViT` Full/Window/NoPos/RoPE2D
- hand-written scaled dot-product attention and MHA
- hand-written 1D RoPE and 2D Axial RoPE
- window partition/reverse and W-MSA
- attention, RoPE, window, symmetry, and pipeline smoke tests
- attention scaling numerical experiment and figure
- Full/Window attention CPU benchmark and figure
- implementation shape check JSON
- corrected 128-sample overfit sanity script

This machine currently has CPU-only PyTorch (`2.13.0+cpu`). The full 9-run training matrix should be run on a CUDA machine or Colab.

## Setup

```powershell
cd D:\attention-week-project
python -m pip install -r requirements.txt
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

The `KMP_DUPLICATE_LIB_OK` line is needed on this Windows/Anaconda install because importing PyTorch otherwise hits an OpenMP runtime conflict.

## Data

```powershell
New-Item -ItemType Directory -Force data
curl.exe -L https://cs231n.stanford.edu/tiny-imagenet-200.zip -o data\tiny-imagenet-200.zip
(Get-Item data\tiny-imagenet-200.zip).Length   # expected 248100043
tar.exe -xf data\tiny-imagenet-200.zip -C data
python -m src.data --data-root data\tiny-imagenet-200 --config lite --seed 42 --out-dir splits
```

Generated lite split checksum on this machine:

```text
1cb97ea048322c7e7625cd43b895f9425ae60b6999b19eb3ab063f206ae76bce
```

## Test Command

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
pytest tests -q
```

Local result: `14 passed`.

## Quick Experiments

```powershell
python -m experiments.scaling_analysis
python -m experiments.benchmark_attention
python -m experiments.compare_backbones --classes 100
python -m experiments.overfit_sanity --steps 300
```

The corrected overfit sanity check reached `train128_acc = 1.0` at 50 steps on CPU.

## Required Training Commands

Run these on CUDA/Colab for the lite configuration. Use the same `splits/split_seed42.json`.

```powershell
python -m src.train --config configs\cnn.yaml --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json
python -m src.train --config configs\lstm.yaml --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json
python -m src.train --config configs\minivit_full.yaml --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json
python -m src.train --config configs\minivit_window.yaml --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json
python -m src.train --config configs\minivit_full_seed43.yaml --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json
python -m src.train --config configs\minivit_window_seed43.yaml --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json
python -m src.train --config configs\minivit_full_nopos.yaml --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json
python -m src.train --config configs\minivit_full_rope2d.yaml --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json
python -m src.train --config configs\minivit_full_multitask.yaml --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json
```

## CLIP Zero-Shot

```powershell
python -m experiments.clip_zero_shot --data-root data\tiny-imagenet-200 --split-json splits\split_seed42.json --batch-size 128
```

Local HuggingFace download failed with a remote connection reset. Run this command in Colab or another network that can access `openai/clip-vit-base-patch32`.

## Final Submission Template

```text
Name:
Repository URL:
Collaborator invitation sent: yes / no
Data configuration: lite
Best reproduction command: pytest tests -q; python -m experiments.scaling_analysis; python -m src.train --config configs/minivit_full.yaml --data-root data/tiny-imagenet-200 --split-json splits/split_seed42.json
Test command: pytest tests -q
Known incomplete item or limitation: full 9-run training matrix and CLIP zero-shot require CUDA/HuggingFace access if not completed locally.
```
