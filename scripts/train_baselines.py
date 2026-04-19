"""Train the six baseline models and save the comparison table / plots.

Usage
-----
    python -m scripts.train_baselines [--gpu]
"""
from __future__ import annotations

import argparse
import json
import os

from src.config import resolve_paths
from src.data import build_image_index, load_metadata
from src.features import load_or_build_features
from src.models import train_all_baselines
from src.plotting import (
    plot_confusion_matrices,
    plot_model_comparison,
    plot_roc_curves,
    results_to_dataframe,
)
from src.preprocessing import build_training_matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU for XGBoost / LightGBM.")
    args = parser.parse_args()

    paths = resolve_paths()
    df = load_metadata(paths["metadata_path"])
    image_path_dict = build_image_index(paths["base_dir"])
    features_df = load_or_build_features(
        df, image_path_dict,
        os.path.join(paths["output_dir"], "image_features.csv"),
    )

    td = build_training_matrices(df, features_df)
    print(f"Train: {td.X_train_resampled.shape} | Test: {td.X_test_scaled.shape}")

    results = train_all_baselines(td, gpu=args.gpu)

    comp_df = results_to_dataframe(results)
    comp_df.to_csv(os.path.join(paths["output_dir"], "model_comparison.csv"),
                   index=False)
    plot_model_comparison(comp_df, paths["output_dir"])
    plot_confusion_matrices(results, td.y_test, td.class_names, paths["output_dir"])
    plot_roc_curves(results, td.y_test, td.class_names, paths["output_dir"])

    summary = {k: {kk: v for kk, v in r.items()
                   if kk not in ("model", "y_pred", "y_proba")}
               for k, r in results.items()}
    with open(os.path.join(paths["output_dir"], "baseline_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nComparison table:")
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
