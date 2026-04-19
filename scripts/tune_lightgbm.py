"""Hyperparameter tuning for LightGBM (5-fold CV with SMOTE-in-fold).

Usage
-----
    python -m scripts.tune_lightgbm --n-iter 15
    python -m scripts.tune_lightgbm --n-iter 30 --gpu
"""
from __future__ import annotations

import argparse
import json
import os

from src.config import resolve_paths
from src.data import build_image_index, load_metadata
from src.features import load_or_build_features
from src.models import evaluate_model
from src.plotting import plot_feature_importance
from src.preprocessing import build_training_matrices
from src.tuning import fit_final_lightgbm, randomized_search_lightgbm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    paths = resolve_paths()
    df = load_metadata(paths["metadata_path"])
    image_path_dict = build_image_index(paths["base_dir"])
    features_df = load_or_build_features(
        df, image_path_dict,
        os.path.join(paths["output_dir"], "image_features.csv"),
    )

    td = build_training_matrices(df, features_df)

    ckpt = os.path.join(paths["output_dir"], "hp_checkpoint.pkl")
    search = randomized_search_lightgbm(
        td.X_train_scaled, td.y_train,
        n_iter=args.n_iter, cv_folds=args.cv_folds,
        gpu=args.gpu, checkpoint_path=ckpt,
    )

    print("\nBest CV Macro F1:", search["best_score"])
    print("Best params:")
    for k, v in search["best_params"].items():
        print(f"  {k}: {v}")

    final_model = fit_final_lightgbm(
        search["best_params"],
        td.X_train_resampled, td.y_train_resampled,
        gpu=args.gpu,
    )
    res = evaluate_model(
        final_model, td.X_train_resampled, td.y_train_resampled,
        td.X_test_scaled, td.y_test, "Tuned LightGBM", td.class_names,
    )

    plot_feature_importance(
        final_model, list(td.X_train_scaled.columns),
        td.image_feat_cols, paths["output_dir"],
    )

    with open(os.path.join(paths["output_dir"], "tuned_lightgbm.json"), "w") as f:
        json.dump({
            "best_params":    search["best_params"],
            "cv_best_score":  search["best_score"],
            "test_accuracy":  res["accuracy"],
            "test_macro_f1":  res["macro_f1"],
            "test_weight_f1": res["weighted_f1"],
            "test_roc_auc":   res["roc_auc"],
        }, f, indent=2)


if __name__ == "__main__":
    main()
