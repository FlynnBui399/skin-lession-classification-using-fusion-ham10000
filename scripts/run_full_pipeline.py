"""End-to-end pipeline: EDA -> features -> baselines -> tuning -> melanoma.

Usage
-----
    python -m scripts.run_full_pipeline [--gpu] [--skip-eda] [--skip-tune]
"""
from __future__ import annotations

import argparse
import subprocess
import sys


STEPS: list[tuple[str, list[str]]] = [
    ("EDA + stats",        [sys.executable, "-m", "scripts.run_eda"]),
    ("Image features",     [sys.executable, "-m", "scripts.extract_features"]),
    ("Baseline models",    [sys.executable, "-m", "scripts.train_baselines"]),
    ("LightGBM tuning",    [sys.executable, "-m", "scripts.tune_lightgbm"]),
    ("Melanoma pipeline",  [sys.executable, "-m", "scripts.run_melanoma_pipeline"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-tune", action="store_true")
    args = parser.parse_args()

    for title, cmd in STEPS:
        if args.skip_eda and title.startswith("EDA"):
            continue
        if args.skip_tune and ("tuning" in title or "Melanoma" in title):
            continue
        if args.gpu and title in ("Baseline models", "LightGBM tuning",
                                  "Melanoma pipeline"):
            cmd = cmd + ["--gpu"]
        print("\n" + "=" * 72)
        print(f">>> {title}")
        print("=" * 72)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
