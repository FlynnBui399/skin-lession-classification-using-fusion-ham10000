"""Run every EDA plot and hypothesis test from the Report.

Usage
-----
    python -m scripts.run_eda
"""
from __future__ import annotations

from src.config import resolve_paths
from src.data import audit_dataset, load_metadata
from src.eda import run_full_eda
from src.stats import (
    chi_square_test,
    dunn_posthoc,
    kruskal_wallis_age,
)


def main() -> None:
    paths = resolve_paths()
    df = load_metadata(paths["metadata_path"])
    audit_dataset(df)

    outs = run_full_eda(df, paths["output_dir"])
    print("\nEDA figures saved:")
    for p in outs:
        print(" -", p)

    print("\n--- Hypothesis Tests ---")
    chi_square_test(df, "sex")
    chi_square_test(df, "localization")
    kruskal_wallis_age(df)
    dunn = dunn_posthoc(df, output_dir=paths["output_dir"])
    print("\nDunn p-value matrix (first 5 rows):")
    print(dunn.round(4).iloc[:5])


if __name__ == "__main__":
    main()
