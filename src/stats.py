"""Statistical hypothesis tests used in the Report.

* :func:`chi_square_test` — contingency Chi-square with Cramér's V effect size.
* :func:`kruskal_wallis_age` — test the equality of age distributions across
  diagnostic classes (non-parametric ANOVA).
* :func:`dunn_posthoc` — pairwise Dunn test with Bonferroni correction.

All functions print their own human-readable report and return a dictionary
of the raw numbers so that the caller can store them in JSON.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import chi2_contingency, kruskal, shapiro

from .config import RANDOM_STATE

ALPHA: float = 0.05
"""Significance level used throughout the report."""


def _cramers_v(chi2: float, n: int, min_dim: int) -> float:
    """Cramér's V effect size from a Chi-square statistic."""
    return float(np.sqrt(chi2 / (n * min_dim)))


def _interpret_v(v: float) -> str:
    if v < 0.1:
        return "Negligible"
    if v < 0.3:
        return "Weak"
    if v < 0.5:
        return "Moderate"
    return "Strong"


def chi_square_test(df: pd.DataFrame, col: str, target: str = "dx",
                    *, verbose: bool = True) -> dict:
    """Run a Chi-square test of independence between ``col`` and ``target``.

    Returns a dict with ``chi2``, ``p_value``, ``dof``, ``cramers_v`` and
    ``low_expected_cells_pct``.
    """
    ct = pd.crosstab(df[col].dropna(), df[target])
    chi2, p_val, dof, expected = chi2_contingency(ct)
    n = int(ct.sum().sum())
    v = _cramers_v(chi2, n, min(ct.shape) - 1)
    low_exp_pct = float((expected < 5).sum() / expected.size * 100)

    if verbose:
        print("=" * 60)
        print(f"Chi-Square — {col} vs {target}")
        print("=" * 60)
        print(ct)
        print(f"\nchi2 = {chi2:.4f} | dof = {dof} | p = {p_val:.3e}")
        print(f"Cramér's V = {v:.4f}  ({_interpret_v(v)})")
        print(f"Cells with expected < 5: {low_exp_pct:.1f}%")
        verdict = "REJECT H0" if p_val < ALPHA else "FAIL TO REJECT H0"
        print(f"Verdict at alpha={ALPHA}: {verdict}")

    return {
        "column": col,
        "target": target,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_val),
        "cramers_v": v,
        "low_expected_cells_pct": low_exp_pct,
        "reject_h0": bool(p_val < ALPHA),
    }


def kruskal_wallis_age(df: pd.DataFrame, group_col: str = "dx",
                       *, verbose: bool = True) -> dict:
    """Kruskal–Wallis on ``age`` across groups of ``group_col``."""
    groups = [g["age"].dropna().values for _, g in df.groupby(group_col)]
    h_stat, p_val = kruskal(*groups)
    shapiro_results = {}
    for name, group in df.groupby(group_col):
        ages = group["age"].dropna()
        sample = ages.sample(min(500, len(ages)), random_state=RANDOM_STATE)
        stat_sw, p_sw = shapiro(sample)
        shapiro_results[str(name)] = {"W": float(stat_sw), "p": float(p_sw)}

    if verbose:
        print("=" * 60)
        print(f"Kruskal–Wallis — age across {group_col}")
        print("=" * 60)
        print(f"H = {h_stat:.4f} | p = {p_val:.3e}")
        verdict = "REJECT H0" if p_val < ALPHA else "FAIL TO REJECT H0"
        print(f"Verdict at alpha={ALPHA}: {verdict}")

    return {
        "H": float(h_stat),
        "p_value": float(p_val),
        "reject_h0": bool(p_val < ALPHA),
        "shapiro_per_group": shapiro_results,
    }


def dunn_posthoc(df: pd.DataFrame, group_col: str = "dx",
                 *, output_dir: Optional[str] = None) -> pd.DataFrame:
    """Dunn's pairwise test with Bonferroni correction on ``age``.

    Returns the p-value matrix (DataFrame). If ``output_dir`` is given, also
    saves ``dunns_test.png`` there.
    """
    dunn = sp.posthoc_dunn(
        df.dropna(subset=["age"]), val_col="age",
        group_col=group_col, p_adjust="bonferroni",
    )

    if output_dir:
        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(dunn, dtype=bool))
        sns.heatmap(dunn, annot=True, fmt=".3f", cmap="RdYlGn_r", mask=mask,
                    ax=ax, vmin=0, vmax=1, linewidths=0.5,
                    cbar_kws={"label": "p-value (Bonferroni)"})
        ax.set_title("Dunn Post-Hoc — Pairwise p-values", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dunns_test.png"), bbox_inches="tight")
        plt.close(fig)

    return dunn
