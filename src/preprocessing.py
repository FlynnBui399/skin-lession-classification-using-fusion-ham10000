"""Preprocessing: merge, impute, encode, split, scale, and SMOTE.

The public entry point is :func:`build_training_matrices`. It takes the raw
metadata DataFrame and the cached image-feature table, and returns all the
matrices the modelling code expects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import CATEGORICAL_METADATA_COLS, RANDOM_STATE


@dataclass
class TrainingData:
    """Everything the baseline-training loop needs, bundled together."""
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_resampled: pd.DataFrame
    y_train_resampled: np.ndarray
    class_names: np.ndarray
    numerical_cols: list[str]
    image_feat_cols: list[str]
    scaler: StandardScaler
    label_encoder: LabelEncoder


def merge_and_impute(metadata: pd.DataFrame,
                     image_features: pd.DataFrame) -> pd.DataFrame:
    """Join metadata with image features, dedupe lesions, impute ``age``."""
    data = metadata.merge(image_features, on="image_id", how="inner")
    data = data.drop_duplicates(subset="lesion_id", keep="first").copy()
    data["age"] = data.groupby("dx")["age"].transform(lambda x: x.fillna(x.median()))
    data["age"] = data["age"].fillna(data["age"].median())
    data = data.dropna(subset=["sex"])
    return data


def build_training_matrices(metadata: pd.DataFrame,
                            image_features: pd.DataFrame,
                            *,
                            test_size: float = 0.2,
                            smote_k: int = 3,
                            random_state: int = RANDOM_STATE) -> TrainingData:
    """One-shot preprocessing.

    Steps
    -----
    1. Merge metadata with the cached image-feature table on ``image_id``.
    2. Drop duplicate lesions (keep the first image of each lesion).
    3. Impute missing ``age`` with the per-class median.
    4. One-hot encode ``sex``, ``dx_type``, ``localization``.
    5. Stratified 80/20 split on ``dx``.
    6. ``StandardScaler`` on all numerical columns (``age`` + image features).
    7. SMOTE on the training split only.
    """
    image_feat_cols = [c for c in image_features.columns if c != "image_id"]
    data = merge_and_impute(metadata, image_features)

    data_encoded = pd.get_dummies(
        data, columns=list(CATEGORICAL_METADATA_COLS), drop_first=False
    )
    drop_cols = {"lesion_id", "image_id", "dx"}
    feature_cols = [c for c in data_encoded.columns if c not in drop_cols]

    X = data_encoded[feature_cols].copy()
    le = LabelEncoder()
    y = le.fit_transform(data_encoded["dx"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numerical_cols = ["age"] + [c for c in image_feat_cols if c in X.columns]
    scaler = StandardScaler()
    X_train_sc = X_train.copy()
    X_test_sc = X_test.copy()
    X_train_sc[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_sc[numerical_cols] = scaler.transform(X_test[numerical_cols])

    smote = SMOTE(random_state=random_state, k_neighbors=smote_k)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)

    return TrainingData(
        X_train_scaled=X_train_sc,
        X_test_scaled=X_test_sc,
        y_train=y_train,
        y_test=y_test,
        X_train_resampled=X_train_res,
        y_train_resampled=y_train_res,
        class_names=le.classes_,
        numerical_cols=numerical_cols,
        image_feat_cols=image_feat_cols,
        scaler=scaler,
        label_encoder=le,
    )


def feature_set_columns(td: TrainingData) -> dict[str, list[str]]:
    """Return column lists for the metadata-only / image-only / combined ablation."""
    meta_only = ["age"] + [
        c for c in td.X_train_scaled.columns
        if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_METADATA_COLS)
    ]
    image_only = [c for c in td.image_feat_cols if c in td.X_train_scaled.columns]
    combined = list(td.X_train_scaled.columns)
    return {
        "Metadata Only": meta_only,
        "Image Only": image_only,
        "Combined (Meta+Image)": combined,
    }
