import json
import os
import random
import statistics
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib.pyplot as plt
import torch

from src.models.attention import MultiHeadSelfAttention
from src.models.window_attention import WindowMultiHeadSelfAttention


def bench_once(module, x, backward=False):
    module.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    if backward:
        out = module(x) if not isinstance(module, WindowMultiHeadSelfAttention) else module(x, (int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5)))
        loss = out.square().mean(); loss.backward()
    else:
        with torch.inference_mode():
            _ = module(x) if not isinstance(module, WindowMultiHeadSelfAttention) else module(x, (int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5)))


def time_module(module, x, backward=False, warmup=10, iters=30):
    device = x.device
    for _ in range(warmup):
        bench_once(module, x, backward)
    if device.type == "cuda":
        torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        bench_once(module, x, backward)
        if device.type == "cuda":
            torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": statistics.median(vals), "p25_ms": statistics.quantiles(vals, n=4)[0], "p75_ms": statistics.quantiles(vals, n=4)[2]}


def main():
    torch.manual_seed(42); random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    rows = []
    for n, grid in [(64, 8), (256, 16), (1024, 32)]:
        full = MultiHeadSelfAttention(128, 4, 0.0).to(device).eval()
        win = WindowMultiHeadSelfAttention(128, 4, 4, 0.0).to(device).eval()
        win.mha.load_state_dict(full.state_dict())
        x = torch.randn(4, n, 128, device=device, dtype=dtype, requires_grad=True)
        order = [("full", full), ("window", win)]
        random.shuffle(order)
        for name, module in order:
            for backward in [False, True]:
                stat = time_module(module, x, backward=backward, warmup=20 if device.type == "cuda" else 10, iters=50 if device.type == "cuda" else 30)
                score_elems = 4 * 4 * n * n if name == "full" else 4 * 4 * n * 16
                rows.append({"module": name, "N": n, "grid": grid, "backward": backward, "score_elements": score_elems, "score_bytes_fp32": score_elems * 4, **stat})
    out = Path("outputs/experiments"); figs = Path("outputs/figures")
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    result = {"device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": "float32", "threads": torch.get_num_threads(), "rows": rows}
    (out / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name in ["full", "window"]:
        xs = [r["N"] for r in rows if r["module"] == name and not r["backward"]]
        ys = [r["median_ms"] for r in rows if r["module"] == name and not r["backward"]]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("N"); ax.set_ylabel("forward median ms"); ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(figs / "attention_benchmark_latency.png", dpi=180)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
