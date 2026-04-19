"""Handcrafted image-feature extraction for dermoscopy.

Every image is summarised by ~30 descriptors grouped into four families:

+-------------+-----------------------------------------------------------+
| Family      | Descriptors                                               |
+=============+===========================================================+
| Colour      | RGB mean/std (6), HSV mean/std (6), ``blue_white_ratio``  |
| Luminance   | ``brightness``, ``contrast``, ``color_variance``          |
| Shape       | ``asymmetry_h``, ``asymmetry_v``, ``edge_density``        |
| Texture     | LBP (mean/std/entropy), GLCM (contrast, dissimilarity,    |
|             | homogeneity, energy, correlation)                         |
+-------------+-----------------------------------------------------------+

The extractor returns a plain ``dict[str, float]`` per image; the driver
script in ``scripts/extract_features.py`` aggregates them into a DataFrame
and caches the result to ``results/image_features.csv``.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm.auto import tqdm

from .config import TARGET_IMAGE_SIZE


def extract_image_features(image_path: str,
                           target_size: tuple[int, int] = TARGET_IMAGE_SIZE
                           ) -> Optional[dict[str, float]]:
    """Extract handcrafted features from a single dermoscopy image.

    Returns ``None`` if ``image_path`` cannot be read (OpenCV returns
    ``None`` for non-existing or corrupted files).
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features: dict[str, float] = {}

    for i, ch in enumerate("rgb"):
        c = img_rgb[:, :, i].astype(np.float64)
        features[f"{ch}_mean"] = float(np.mean(c))
        features[f"{ch}_std"] = float(np.std(c))

    for i, ch in enumerate("hsv"):
        c = img_hsv[:, :, i].astype(np.float64)
        features[f"{ch}_mean"] = float(np.mean(c))
        features[f"{ch}_std"] = float(np.std(c))

    gray_f = img_gray.astype(np.float64)
    features["brightness"] = float(np.mean(gray_f))
    features["contrast"] = float(np.std(gray_f))

    h, w = img_gray.shape
    half_w = w // 2
    left = gray_f[:, :half_w]
    right = np.flip(gray_f[:, w - half_w:], axis=1)
    features["asymmetry_h"] = float(np.mean(np.abs(left - right)))

    half_h = h // 2
    top = gray_f[:half_h, :]
    bottom = np.flip(gray_f[h - half_h:, :], axis=0)
    features["asymmetry_v"] = float(np.mean(np.abs(top - bottom)))

    lbp = local_binary_pattern(img_gray, P=8, R=1, method="uniform")
    features["lbp_mean"] = float(np.mean(lbp))
    features["lbp_std"] = float(np.std(lbp))
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
    nonzero = lbp_hist[lbp_hist > 0]
    features["lbp_entropy"] = float(-np.sum(nonzero * np.log2(nonzero)))

    gray_q = (img_gray // 4).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=[1, 3], angles=[0, np.pi / 4],
                        levels=64, symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        features[f"glcm_{prop}"] = float(np.mean(graycoprops(glcm, prop)))

    features["color_variance"] = float(
        np.mean([np.var(img_rgb[:, :, i].astype(np.float64)) for i in range(3)])
    )
    edges = cv2.Canny(img_gray, 50, 150)
    features["edge_density"] = float(np.sum(edges > 0) / (h * w))
    features["blue_white_ratio"] = float(
        (features["b_mean"] + 1) / (features["r_mean"] + 1)
    )
    return features


def build_feature_table(df: pd.DataFrame, image_path_dict: dict[str, str],
                        *, target_size: tuple[int, int] = TARGET_IMAGE_SIZE,
                        show_progress: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Run :func:`extract_image_features` for every row in ``df``.

    Parameters
    ----------
    df
        Metadata DataFrame (must contain ``image_id``).
    image_path_dict
        Mapping from ``image_id`` to filesystem path.

    Returns
    -------
    (features_df, failed_ids)
        ``features_df`` has one row per successfully extracted image and a
        ``image_id`` column for joining with the metadata.
    """
    all_feats: list[dict] = []
    failed: list[str] = []
    iterable = df.iterrows()
    if show_progress:
        iterable = tqdm(iterable, total=len(df), desc="Extracting image features")
    for _, row in iterable:
        iid = row["image_id"]
        path = image_path_dict.get(iid)
        if path is None:
            failed.append(iid)
            continue
        f = extract_image_features(path, target_size=target_size)
        if f is None:
            failed.append(iid)
            continue
        f["image_id"] = iid
        all_feats.append(f)
    return pd.DataFrame(all_feats), failed


def load_or_build_features(df: pd.DataFrame, image_path_dict: dict[str, str],
                           cache_path: str,
                           *, force_rebuild: bool = False) -> pd.DataFrame:
    """Load cached features from ``cache_path`` or build and cache them.

    This is the function the notebooks call: it makes the image-feature
    extraction idempotent and fast on subsequent runs.
    """
    if os.path.exists(cache_path) and not force_rebuild:
        return pd.read_csv(cache_path)
    features_df, failed = build_feature_table(df, image_path_dict)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    features_df.to_csv(cache_path, index=False)
    if failed:
        print(f"[warning] failed to extract features for {len(failed)} images")
    return features_df
