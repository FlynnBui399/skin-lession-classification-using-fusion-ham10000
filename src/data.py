"""Dataset I/O and indexing for HAM10000.

Responsibilities
----------------
* ``load_metadata``      — read ``HAM10000_metadata.csv`` as a typed DataFrame.
* ``build_image_index``  — map each ``image_id`` to its JPEG path.
* ``audit_dataset``      — print a short sanity-check summary (shape, missing
  values, duplicate ``lesion_id``).
"""
from __future__ import annotations

import os
from glob import glob
from typing import Optional

import pandas as pd

from .config import resolve_paths


def load_metadata(metadata_path: Optional[str] = None) -> pd.DataFrame:
    """Load ``HAM10000_metadata.csv`` into a DataFrame.

    Parameters
    ----------
    metadata_path
        Optional explicit path. When omitted, :func:`resolve_paths` is used.
    """
    if metadata_path is None:
        metadata_path = resolve_paths()["metadata_path"]
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found at '{metadata_path}'. "
            "See data/README.md for download instructions."
        )
    return pd.read_csv(metadata_path)


def build_image_index(base_dir: Optional[str] = None) -> dict[str, str]:
    """Walk ``base_dir`` recursively and return ``{image_id: absolute_path}``.

    On HAM10000 each image appears in exactly one of
    ``HAM10000_images_part_1/`` or ``HAM10000_images_part_2/``; the two parts
    share a naming space, so a single dictionary keyed by ``image_id`` is
    enough.
    """
    if base_dir is None:
        base_dir = resolve_paths()["base_dir"]
    paths = glob(os.path.join(base_dir, "**", "*.jpg"), recursive=True)
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def audit_dataset(df: pd.DataFrame) -> None:
    """Print a short human-readable audit of the loaded metadata."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Rows / Cols:                         {df.shape[0]} / {df.shape[1]}")
    print(f"Unique lesions (lesion_id):          {df['lesion_id'].nunique()}")
    print(f"Unique images  (image_id):           {df['image_id'].nunique()}")
    print(f"Multi-image lesions (duplicates):    {df.shape[0] - df['lesion_id'].nunique()}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        pct = (missing / len(df) * 100).round(2)
        print("\n--- Missing values ---")
        print(pd.DataFrame({"Count": missing, "Pct (%)": pct}).to_string())
    else:
        print("\nNo missing values detected.")

    print("\n--- Class distribution (dx) ---")
    counts = df["dx"].value_counts()
    pct = (counts / len(df) * 100).round(2)
    print(pd.DataFrame({"Count": counts, "Pct (%)": pct}).to_string())
    print(f"\nImbalance ratio (max/min): {counts.max() / counts.min():.1f}x")
