import argparse
import json
from pathlib import Path


def load_metric(run_dir: Path):
    path = run_dir / "final_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/positional_ablation.json"))
    args = parser.parse_args()
    mapping = {"NoPos": "R7", "AbsPos": "R3", "RoPE2D": "R8"}
    rows = []
    for variant, run_id in mapping.items():
        m = load_metric(args.runs_dir / run_id)
        if m is None:
            rows.append({"variant": variant, "run_id": run_id, "status": "missing final_metrics.json"})
            continue
        rows.append({
            "variant": variant,
            "run_id": run_id,
            "params": m.get("params"),
            "best_val_acc": m.get("best_val_acc"),
            "test_acc": (m.get("test") or {}).get("accuracy"),
            "train_time_per_epoch": None,
            "status": "ok",
        })
    abs_row = next((r for r in rows if r["variant"] == "AbsPos" and r.get("status") == "ok"), None)
    if abs_row:
        for r in rows:
            if r.get("status") == "ok":
                r["test_acc_diff_vs_abspos"] = r["test_acc"] - abs_row["test_acc"]
    result = {"rows": rows, "note": "AbsPos reuses R3 by project protocol."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
