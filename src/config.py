"""Project-wide configuration: paths, constants, and diagnosis lookup.

The module auto-detects whether the code is running on Kaggle or on a local
machine, and returns the right ``DATA_DIR`` / ``OUTPUT_DIR`` accordingly. All
other modules import paths from here so that every script writes its outputs
to the same ``results/`` folder.
"""
from __future__ import annotations

import os
from glob import glob
from pathlib import Path

RANDOM_STATE: int = 42
"""Global seed reused across splits, SMOTE, and model fits."""

TARGET_IMAGE_SIZE: tuple[int, int] = (128, 128)
"""Resolution used by the handcrafted feature extractor."""

DX_FULL: dict[str, str] = {
    "nv":    "Melanocytic nevi",
    "mel":   "Melanoma",
    "bkl":   "Benign keratosis",
    "bcc":   "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc":  "Vascular lesions",
    "df":    "Dermatofibroma",
}
"""Human-readable names for the seven HAM10000 diagnostic classes."""

DX_CLASSES: tuple[str, ...] = tuple(DX_FULL.keys())

CATEGORICAL_METADATA_COLS: tuple[str, ...] = ("sex", "dx_type", "localization")
"""Metadata columns that get one-hot encoded."""


def _project_root() -> Path:
    """Return the repository root (the folder that contains ``data/``)."""
    here = Path(__file__).resolve()
    # src/config.py -> src/ -> project root
    return here.parent.parent


PROJECT_ROOT: Path = _project_root()


def resolve_paths(prefer_kaggle: bool = True) -> dict[str, str]:
    """Return a dictionary of canonical project paths.

    Parameters
    ----------
    prefer_kaggle
        If ``True`` (default) and ``/kaggle/input`` exists, use Kaggle paths.
        Set to ``False`` to always use the local ``data/`` and ``results/``.

    Returns
    -------
    dict
        Keys: ``base_dir`` (folder holding the metadata CSV and images),
        ``metadata_path`` (full path to ``HAM10000_metadata.csv``),
        ``output_dir`` (where figures and JSON summaries are written),
        ``data_dir``, ``results_dir``, ``project_root``.
    """
    on_kaggle = prefer_kaggle and os.path.isdir("/kaggle/input")
    project_root = PROJECT_ROOT
    local_data_dir = project_root / "data"
    local_results_dir = project_root / "results"

    if on_kaggle:
        output_dir = "/kaggle/working"
        candidates = glob("/kaggle/input/**/HAM10000_metadata.csv", recursive=True)
        if candidates:
            metadata_path = candidates[0]
            base_dir = os.path.dirname(metadata_path)
        else:
            base_dir = "/kaggle/input/skin-cancer-mnist-ham10000"
            metadata_path = os.path.join(base_dir, "HAM10000_metadata.csv")
    else:
        output_dir = str(local_results_dir)
        os.makedirs(output_dir, exist_ok=True)
        candidates = glob(str(local_data_dir / "**" / "HAM10000_metadata.csv"), recursive=True)
        if candidates:
            metadata_path = candidates[0]
            base_dir = os.path.dirname(metadata_path)
        else:
            base_dir = str(local_data_dir)
            metadata_path = str(local_data_dir / "HAM10000_metadata.csv")

    return {
        "project_root":  str(project_root),
        "data_dir":      str(local_data_dir),
        "results_dir":   str(local_results_dir),
        "base_dir":      base_dir,
        "metadata_path": metadata_path,
        "output_dir":    output_dir,
    }
