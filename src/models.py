"""Baseline model factory + unified evaluator.

The six models used in §6 of the Report are built here. Keeping them in one
module means the exact same configuration can be invoked from the notebook,
from ``scripts/train_baselines.py``, or from a future test suite.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import lightgbm as lgb
import xgboost as xgb

from .config import RANDOM_STATE


def build_baseline_models(n_classes: int, gpu: bool = False,
                          random_state: int = RANDOM_STATE
                          ) -> dict[str, Any]:
    """Return the six baseline classifiers used in §6 of the Report."""
    lr = LogisticRegression(
        max_iter=2000, solver="lbfgs",
        class_weight="balanced", random_state=random_state, C=1.0,
    )
    knn = KNeighborsClassifier(
        n_neighbors=7, weights="distance", metric="minkowski", n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=20,
        min_samples_split=5, min_samples_leaf=2,
        class_weight="balanced", random_state=random_state, n_jobs=-1,
    )
    svm = SVC(
        kernel="rbf", C=10, gamma="scale", class_weight="balanced",
        probability=True, random_state=random_state,
    )

    xgb_params: dict[str, Any] = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=n_classes,
        random_state=random_state, eval_metric="mlogloss",
    )
    if gpu:
        xgb_params.update(tree_method="hist", device="cuda")
    else:
        xgb_params["n_jobs"] = -1
    xgb_model = xgb.XGBClassifier(**xgb_params)

    lgb_params: dict[str, Any] = dict(
        n_estimators=500, max_depth=8, learning_rate=0.05, num_leaves=50,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=1.0, class_weight="balanced",
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    if gpu:
        lgb_params["device"] = "gpu"
    lgb_model = lgb.LGBMClassifier(**lgb_params)

    return {
        "LR":   ("Logistic Regression", lr),
        "KNN":  ("K-Nearest Neighbors", knn),
        "RF":   ("Random Forest", rf),
        "SVM":  ("SVM (RBF)", svm),
        "XGB":  ("XGBoost", xgb_model),
        "LGBM": ("LightGBM", lgb_model),
    }


def evaluate_model(model: Any,
                   X_train: pd.DataFrame, y_train: np.ndarray,
                   X_test: pd.DataFrame, y_test: np.ndarray,
                   name: str, class_names: np.ndarray,
                   *, verbose: bool = True) -> dict[str, Any]:
    """Train, predict, and collect headline metrics.

    Returns a dict with ``model_name``, ``model``, ``accuracy``, ``macro_f1``,
    ``weighted_f1``, ``roc_auc``, ``y_pred``, ``y_proba``.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_proba: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_proba = softmax(model.decision_function(X_test), axis=1)

    acc = accuracy_score(y_test, y_pred)
    mf1 = f1_score(y_test, y_pred, average="macro")
    wf1 = f1_score(y_test, y_pred, average="weighted")
    roc: float | None = None
    if y_proba is not None:
        try:
            roc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except Exception:
            roc = None

    if verbose:
        print(f"\n{'=' * 60}\nModel: {name}\n{'=' * 60}")
        print(f"Accuracy:     {acc:.4f}")
        print(f"Macro F1:     {mf1:.4f}")
        print(f"Weighted F1:  {wf1:.4f}")
        if roc is not None:
            print(f"ROC-AUC:      {roc:.4f}")
        print("\n" + classification_report(y_test, y_pred, target_names=list(class_names)))

    return {
        "model_name": name,
        "model": model,
        "accuracy": float(acc),
        "macro_f1": float(mf1),
        "weighted_f1": float(wf1),
        "roc_auc": None if roc is None else float(roc),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def train_all_baselines(td, *, gpu: bool = False) -> dict[str, dict]:
    """Fit and evaluate all six baseline models on the resampled training set."""
    models = build_baseline_models(n_classes=len(td.class_names), gpu=gpu)
    results: dict[str, dict] = {}
    for key, (name, mdl) in models.items():
        results[key] = evaluate_model(
            mdl,
            td.X_train_resampled, td.y_train_resampled,
            td.X_test_scaled, td.y_test,
            name, td.class_names,
        )
    return results
