"""LightGBM randomized hyperparameter search with checkpointing.

Key design choices
------------------
* ``StratifiedKFold`` on the **pre-SMOTE** training set.
* SMOTE is applied **inside each fold** only (via :mod:`imblearn.pipeline`
  or a manual ``fit_resample`` per fold) to prevent train/val leakage.
* Each iteration is checkpointed to a pickle file so long searches are
  resumable without wasting compute.
"""
from __future__ import annotations

import os
import pickle
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from .config import RANDOM_STATE

DEFAULT_PARAM_DIST: dict[str, list] = {
    "n_estimators":       [300, 500, 700, 1000],
    "max_depth":          [5, 8, 12, 15, -1],
    "learning_rate":      [0.01, 0.03, 0.05, 0.1],
    "num_leaves":         [31, 50, 70, 100],
    "subsample":          [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":   [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_samples":  [5, 10, 20, 30],
    "reg_alpha":          [0, 0.01, 0.1, 0.5],
    "reg_lambda":         [0, 0.01, 0.1, 1.0, 5.0],
}


def _idx(data, idx):
    return data.iloc[idx] if hasattr(data, "iloc") else data[idx]


def _eval_fold(X_train: pd.DataFrame, y_train: np.ndarray,
               train_idx: np.ndarray, val_idx: np.ndarray,
               hp: dict[str, Any], base_params: dict[str, Any],
               smote_k: int, early_stop: int) -> float:
    """One CV fold: SMOTE inside, early stopping on the held-out fold."""
    X_tr, X_vl = _idx(X_train, train_idx), _idx(X_train, val_idx)
    y_tr, y_vl = _idx(y_train, train_idx), _idx(y_train, val_idx)
    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    mdl = lgb.LGBMClassifier(**{**base_params, **hp})
    mdl.fit(
        X_tr_sm, y_tr_sm,
        eval_set=[(X_vl, y_vl)],
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return float(f1_score(y_vl, mdl.predict(X_vl), average="macro"))


def randomized_search_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray,
                               *,
                               n_iter: int = 15,
                               cv_folds: int = 5,
                               smote_k: int = 3,
                               early_stop_rounds: int = 50,
                               param_pool: int = 50,
                               param_dist: dict[str, list] | None = None,
                               gpu: bool = False,
                               checkpoint_path: str | None = None,
                               random_state: int = RANDOM_STATE,
                               verbose: bool = True) -> dict[str, Any]:
    """Run a resumable randomized search.

    Returns a dict with ``best_params``, ``best_score``, ``cv_results``
    (list of per-iteration dicts), and ``completed`` (the set of iteration
    indices). If ``checkpoint_path`` is given, progress is saved after each
    iteration.
    """
    param_dist = param_dist or DEFAULT_PARAM_DIST
    all_params = list(ParameterSampler(param_dist, n_iter=param_pool,
                                       random_state=random_state))

    completed: set[int] = set()
    cv_results: list[dict[str, Any]] = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            state = pickle.load(f)
        completed = state["completed"]
        cv_results = state["cv_results"]
        if verbose:
            print(f"Resumed from checkpoint: {len(completed)} iterations already done")

    base_params: dict[str, Any] = dict(
        class_weight="balanced", random_state=random_state,
        n_jobs=1, verbose=-1,
    )
    if gpu:
        base_params["device"] = "gpu"

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_splits = list(cv.split(X_train, y_train))

    for i in range(n_iter):
        if i in completed:
            continue
        hp = all_params[i]
        t0 = time.time()
        scores = [
            _eval_fold(X_train, y_train, tr, vl, hp, base_params,
                       smote_k, early_stop_rounds)
            for tr, vl in cv_splits
        ]
        mean_sc, std_sc = float(np.mean(scores)), float(np.std(scores))
        cv_results.append({
            "idx": i, "params": hp,
            "mean_score": mean_sc, "std_score": std_sc,
        })
        completed.add(i)

        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"completed": completed, "cv_results": cv_results}, f)

        if verbose:
            best = max(cv_results, key=lambda x: x["mean_score"])["mean_score"]
            print(f"  [{len(completed):>2}/{n_iter}] F1={mean_sc:.4f} "
                  f"(best so far: {best:.4f}) [{time.time() - t0:.0f}s]")

    best_entry = max(cv_results, key=lambda x: x["mean_score"])
    return {
        "best_params": best_entry["params"],
        "best_score": best_entry["mean_score"],
        "cv_results": cv_results,
        "completed": sorted(completed),
    }


def fit_final_lightgbm(best_params: dict[str, Any],
                       X_train_resampled: pd.DataFrame,
                       y_train_resampled: np.ndarray,
                       *, gpu: bool = False,
                       random_state: int = RANDOM_STATE):
    """Retrain a LightGBM on the full SMOTE-resampled training set."""
    base = dict(class_weight="balanced", random_state=random_state,
                n_jobs=1, verbose=-1)
    if gpu:
        base["device"] = "gpu"
    model = lgb.LGBMClassifier(**{**base, **best_params})
    model.fit(X_train_resampled, y_train_resampled)
    return model
