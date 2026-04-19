"""Melanoma-focused improvements (§11 of the Report).

The goal here is to boost melanoma recall without leaving the tabular
pipeline. Three complementary strategies are provided:

1. :func:`sweep_threshold_7class`  — tune the mel-probability threshold on the
   tuned 7-class model.
2. :func:`train_binary_mel_vs_rest` — dedicated LightGBM binary classifier
   with aggressive SMOTE / ``scale_pos_weight``.
3. :func:`two_tier_predict`        — binary gate first, 7-class model second.
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .config import RANDOM_STATE


def mel_class_index(class_names: np.ndarray) -> int:
    """Return the column index of the melanoma class inside ``y_proba``."""
    arr = np.asarray(class_names)
    idx = np.where(arr == "mel")[0]
    if len(idx) == 0:
        raise ValueError("No 'mel' class in class_names.")
    return int(idx[0])


def sweep_threshold_7class(y_proba: np.ndarray, y_true: np.ndarray,
                           class_names: np.ndarray,
                           thresholds: np.ndarray | None = None
                           ) -> pd.DataFrame:
    """Vary the mel-probability threshold on the 7-class model.

    A sample is labelled ``mel`` iff ``y_proba[:, mel_idx] >= t``; otherwise
    the argmax is used. Returns one row per threshold with mel recall /
    precision, macro-F1 and accuracy.
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.50, 9)
    mel_idx = mel_class_index(class_names)
    argmax_pred = np.argmax(y_proba, axis=1)
    rows = []
    for t in thresholds:
        y_pred = argmax_pred.copy()
        y_pred[y_proba[:, mel_idx] >= t] = mel_idx
        is_mel_true = (y_true == mel_idx).astype(int)
        is_mel_pred = (y_pred == mel_idx).astype(int)
        rows.append({
            "threshold": float(t),
            "mel_recall":    float(recall_score(is_mel_true, is_mel_pred, zero_division=0)),
            "mel_precision": float(precision_score(is_mel_true, is_mel_pred, zero_division=0)),
            "accuracy":      float(accuracy_score(y_true, y_pred)),
            "macro_f1":      float(f1_score(y_true, y_pred, average="macro")),
        })
    return pd.DataFrame(rows)


def train_binary_mel_vs_rest(X_train_scaled: pd.DataFrame,
                             y_train: np.ndarray,
                             class_names: np.ndarray,
                             *,
                             gpu: bool = False,
                             random_state: int = RANDOM_STATE) -> Any:
    """Binary LightGBM (mel vs everything else) with aggressive SMOTE."""
    mel_idx = mel_class_index(class_names)
    y_bin = (y_train == mel_idx).astype(int)
    pos = int(y_bin.sum())
    neg = int(len(y_bin) - pos)
    scale_pos_weight = max(1.0, neg / max(pos, 1))

    sm = SMOTE(random_state=random_state, k_neighbors=3,
               sampling_strategy={1: min(neg, pos * 5)})
    X_res, y_res = sm.fit_resample(X_train_scaled, y_bin)

    params: dict[str, Any] = dict(
        n_estimators=800, max_depth=8, learning_rate=0.03,
        num_leaves=64, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
        objective="binary", scale_pos_weight=scale_pos_weight,
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    if gpu:
        params["device"] = "gpu"

    model = lgb.LGBMClassifier(**params)
    model.fit(X_res, y_res)
    return model


def two_tier_predict(binary_model: Any,
                     multiclass_model: Any,
                     X_test: pd.DataFrame,
                     class_names: np.ndarray,
                     *, binary_gate_threshold: float = 0.5) -> np.ndarray:
    """Apply the two-tier pipeline.

    * Run the binary mel-vs-rest model. Samples whose mel probability is
      ``>= binary_gate_threshold`` are labelled ``mel`` immediately.
    * The remaining samples go through the 7-class model.
    """
    mel_idx = mel_class_index(class_names)
    bin_proba = binary_model.predict_proba(X_test)[:, 1]
    is_gated = bin_proba >= binary_gate_threshold

    y_pred = np.empty(len(X_test), dtype=int)
    y_pred[is_gated] = mel_idx

    if np.any(~is_gated):
        mc_pred = multiclass_model.predict(X_test.iloc[~is_gated] if hasattr(X_test, "iloc")
                                           else X_test[~is_gated])
        y_pred[~is_gated] = mc_pred
    return y_pred


def melanoma_headline_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: np.ndarray) -> dict[str, float]:
    """Return ``recall`` / ``precision`` / ``f1`` for the melanoma class."""
    mel_idx = mel_class_index(class_names)
    is_mel_true = (y_true == mel_idx).astype(int)
    is_mel_pred = (y_pred == mel_idx).astype(int)
    return {
        "recall":    float(recall_score(is_mel_true, is_mel_pred, zero_division=0)),
        "precision": float(precision_score(is_mel_true, is_mel_pred, zero_division=0)),
        "f1":        float(f1_score(is_mel_true, is_mel_pred, zero_division=0)),
    }
