"""Re-run test evaluation and error analysis only (no training).

Useful if you want to regenerate plots with a different TTA setting, etc.

Usage:
    python -m scripts.run_analysis --config configs/convnext_tiny.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_train import run_test_and_analysis
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    run_test_and_analysis(cfg)


if __name__ == "__main__":
    main()
