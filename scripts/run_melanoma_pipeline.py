"""Melanoma-focused improvements (§11 of the Report).

Runs the tuned 7-class LightGBM to get ``y_proba``, then evaluates:
* threshold sweep on mel-probability,
* a dedicated binary mel-vs-rest LightGBM,
* the two-tier pipeline that combines both.

Usage
-----
    python -m scripts.run_melanoma_pipeline [--gpu]
"""
from __future__ import annotations

import argparse
import json
import os

from src.config import resolve_paths
from src.data import build_image_index, load_metadata
from src.features import load_or_build_features
from src.melanoma import (
    melanoma_headline_metrics,
    sweep_threshold_7class,
    train_binary_mel_vs_rest,
    two_tier_predict,
)
from src.preprocessing import build_training_matrices
from src.tuning import fit_final_lightgbm, randomized_search_lightgbm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--n-iter", type=int, default=15)
    args = parser.parse_args()

    paths = resolve_paths()
    df = load_metadata(paths["metadata_path"])
    image_path_dict = build_image_index(paths["base_dir"])
    features_df = load_or_build_features(
        df, image_path_dict,
        os.path.join(paths["output_dir"], "image_features.csv"),
    )
    td = build_training_matrices(df, features_df)

    search = randomized_search_lightgbm(
        td.X_train_scaled, td.y_train,
        n_iter=args.n_iter, gpu=args.gpu,
        checkpoint_path=os.path.join(paths["output_dir"], "hp_checkpoint.pkl"),
    )
    multi_model = fit_final_lightgbm(
        search["best_params"],
        td.X_train_resampled, td.y_train_resampled,
        gpu=args.gpu,
    )
    y_proba = multi_model.predict_proba(td.X_test_scaled)

    print("\n--- Threshold sweep on 7-class model ---")
    sweep_df = sweep_threshold_7class(y_proba, td.y_test, td.class_names)
    print(sweep_df.round(4).to_string(index=False))
    sweep_df.to_csv(os.path.join(paths["output_dir"], "mel_threshold_sweep.csv"),
                    index=False)

    print("\n--- Binary mel-vs-rest ---")
    bin_model = train_binary_mel_vs_rest(
        td.X_train_scaled, td.y_train, td.class_names, gpu=args.gpu,
    )

    print("\n--- Two-tier (binary gate + 7-class) ---")
    y_pred_2t = two_tier_predict(bin_model, multi_model,
                                 td.X_test_scaled, td.class_names)
    headline = melanoma_headline_metrics(td.y_test, y_pred_2t, td.class_names)
    print(f"Mel recall / precision / F1 : "
          f"{headline['recall']:.3f} / {headline['precision']:.3f} / {headline['f1']:.3f}")

    with open(os.path.join(paths["output_dir"], "melanoma_pipeline.json"), "w") as f:
        json.dump({
            "threshold_sweep": sweep_df.to_dict(orient="records"),
            "two_tier": headline,
        }, f, indent=2)


if __name__ == "__main__":
    main()
