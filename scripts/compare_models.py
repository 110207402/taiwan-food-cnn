"""Combine results from multiple trained models into a single comparison table.

Usage:
    python -m scripts.compare_models --runs outputs/convnext_tiny outputs/resnet50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def collect(run_dirs):
    rows = []
    for d in run_dirs:
        d = Path(d)
        metrics_path = d / "test_metrics.json"
        if not metrics_path.exists():
            print(f"[skip] {d}: no test_metrics.json", file=sys.stderr)
            continue
        with open(metrics_path) as f:
            m = json.load(f)

        train_summary = {}
        ts_path = d / "train_summary.json"
        if ts_path.exists():
            with open(ts_path) as f:
                train_summary = json.load(f)

        rows.append({
            "model": d.name,
            "test_top1": round(m["top1"], 2),
            "test_top5": round(m["top5"], 2),
            "macro_f1": round(m["macro_f1"], 2),
            "weighted_f1": round(m["weighted_f1"], 2),
            "best_val_top1": round(train_summary.get("best_val_top1", float("nan")), 2),
            "best_epoch": train_summary.get("best_epoch", "-"),
            "params": train_summary.get("params_total", "-"),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="outputs/comparison.csv")
    args = parser.parse_args()

    df = collect(args.runs)
    if df.empty:
        print("No runs with metrics found.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
