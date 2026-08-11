import json
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib.pyplot as plt
import torch


def softmax_jacobian_frobenius(p):
    diag_sq = (p * (1 - p)).square().sum(dim=-1)
    off_sq = ((p.unsqueeze(-1) * p.unsqueeze(-2)).square().sum(dim=(-1, -2)) - (p.square().square().sum(dim=-1)))
    return torch.sqrt(diag_sq + off_sq)


def run(seed=42, trials=100, length=64, dks=(8, 32, 128, 512)):
    torch.manual_seed(seed)
    rows = []
    for dk in dks:
        for scaled in [False, True]:
            scores_all, ent_all, sens_all = [], [], []
            for _ in range(trials):
                q = torch.randn(length, dk)
                k = torch.randn(length, dk)
                s = q @ k.t()
                if scaled:
                    s = s / math.sqrt(dk)
                p = torch.softmax(s, dim=-1)
                entropy = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1) / math.log(length)
                sens = softmax_jacobian_frobenius(p)
                scores_all.append(s.reshape(-1))
                ent_all.append(entropy)
                sens_all.append(sens)
            score = torch.cat(scores_all); ent = torch.cat(ent_all); sens = torch.cat(sens_all)
            rows.append({
                "d_k": dk,
                "scaled": scaled,
                "score_mean": float(score.mean()),
                "score_std_mean": float(score.std(unbiased=True)),
                "normalized_entropy_mean": float(ent.mean()),
                "normalized_entropy_std": float(ent.std(unbiased=True)),
                "jacobian_sensitivity_mean": float(sens.mean()),
                "jacobian_sensitivity_std": float(sens.std(unbiased=True)),
            })
    return rows


def finite_difference(seed=123, length=64, eps=1e-5):
    torch.manual_seed(seed)
    z = torch.randn(length, dtype=torch.float64)
    u = torch.randn(length, dtype=torch.float64); u = u / u.norm()
    p = torch.softmax(z, dim=-1)
    j = torch.diag(p) - p[:, None] * p[None, :]
    fd = (torch.softmax(z + eps * u, dim=-1) - p).norm() / eps
    exact = (j @ u).norm()
    return {"epsilon": eps, "finite_difference": float(fd), "jacobian_times_u": float(exact), "absolute_error": float((fd - exact).abs())}


def plot(rows, out_path):
    dks = sorted({r["d_k"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    metrics = [("score_std_mean", "Score std"), ("normalized_entropy_mean", "Normalized entropy"), ("jacobian_sensitivity_mean", "Jacobian sensitivity")]
    for ax, (key, title) in zip(axes, metrics):
        for scaled, label in [(False, "unscaled"), (True, "scaled")]:
            ys = [next(r for r in rows if r["d_k"] == dk and r["scaled"] == scaled)[key] for dk in dks]
            ax.plot(dks, ys, marker="o", label=label)
        ax.set_xscale("log", base=2); ax.set_title(title); ax.set_xlabel("d_k"); ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=180)


def main():
    out = Path("outputs/experiments"); figs = Path("outputs/figures")
    out.mkdir(parents=True, exist_ok=True); figs.mkdir(parents=True, exist_ok=True)
    rows = run()
    result = {"seed": 42, "length": 64, "trials": 100, "rows": rows, "finite_difference": finite_difference()}
    (out / "scaling.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    plot(rows, figs / "attention_scaling.png")
    print(json.dumps(result["finite_difference"], indent=2))


if __name__ == "__main__":
    main()
