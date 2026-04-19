"""Extract handcrafted image features and cache them to CSV.

Usage
-----
    python -m scripts.extract_features
    python -m scripts.extract_features --force
"""
from __future__ import annotations

import argparse
import os

from src.config import resolve_paths
from src.data import audit_dataset, build_image_index, load_metadata
from src.features import load_or_build_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if a cached CSV exists.")
    args = parser.parse_args()

    paths = resolve_paths()
    df = load_metadata(paths["metadata_path"])
    audit_dataset(df)

    image_path_dict = build_image_index(paths["base_dir"])
    print(f"\nIndexed {len(image_path_dict)} images under {paths['base_dir']}")

    cache = os.path.join(paths["output_dir"], "image_features.csv")
    features_df = load_or_build_features(
        df, image_path_dict, cache, force_rebuild=args.force
    )
    print(f"\nSaved: {cache}  (shape={features_df.shape})")


if __name__ == "__main__":
    main()
