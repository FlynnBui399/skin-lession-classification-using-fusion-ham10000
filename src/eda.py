"""Plotting helpers for exploratory data analysis.

Each function saves its figure under ``output_dir`` with a predictable name
(e.g. ``class_distribution.png``). The filenames match what the Report
notebook expects, so the ``results/`` folder stays consistent whether the
plots were produced by the notebook or by a standalone script.
"""
from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import DX_FULL


def plot_class_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """Bar + pie chart of the seven diagnostic classes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    dx_counts = df["dx"].value_counts()
    colors = sns.color_palette("Set2", n_colors=len(dx_counts))

    bars = axes[0].bar(dx_counts.index, dx_counts.values, color=colors)
    axes[0].set_title("Distribution of Skin-Lesion Types", fontweight="bold")
    axes[0].set_xlabel("Diagnosis")
    axes[0].set_ylabel("Count")
    for bar, count in zip(bars, dx_counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count}\n({count/len(df)*100:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )

    axes[1].pie(
        dx_counts.values,
        labels=[DX_FULL.get(x, x) for x in dx_counts.index],
        autopct="%1.1f%%", colors=colors, startangle=90,
    )
    axes[1].set_title("Proportion of Skin-Lesion Types", fontweight="bold")

    out = os.path.join(output_dir, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_age_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """Histogram (density) + boxplot of ``age`` by diagnosis."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for dx_val in df["dx"].unique():
        ages = df[df["dx"] == dx_val]["age"].dropna()
        axes[0].hist(ages, bins=20, alpha=0.5, density=True,
                     label=DX_FULL.get(dx_val, dx_val))
    axes[0].set_title("Age Distribution by Diagnosis (Density)", fontweight="bold")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=7)

    order = df.groupby("dx")["age"].median().sort_values().index
    sns.boxplot(data=df, x="dx", y="age", order=order, palette="Set2", ax=axes[1])
    axes[1].set_title("Age Distribution by Diagnosis (Boxplot)", fontweight="bold")

    out = os.path.join(output_dir, "age_distribution.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sex_distribution(df: pd.DataFrame, output_dir: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sex_counts = df["sex"].value_counts()
    axes[0].bar(sex_counts.index, sex_counts.values, color=["#66c2a5", "#fc8d62", "#8da0cb"][:len(sex_counts)])
    axes[0].set_title("Overall Sex Distribution", fontweight="bold")
    for i, (_, val) in enumerate(sex_counts.items()):
        axes[0].text(i, val + 30, f"{val} ({val/len(df)*100:.1f}%)",
                     ha="center", fontsize=10)

    ct_sex = pd.crosstab(df["dx"], df["sex"], normalize="index") * 100
    ct_sex.plot(kind="bar", stacked=True, ax=axes[1],
                color=["#66c2a5", "#fc8d62", "#8da0cb"][:ct_sex.shape[1]],
                edgecolor="white")
    axes[1].set_title("Sex Distribution Within Each Diagnosis", fontweight="bold")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend(title="Sex")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

    out = os.path.join(output_dir, "sex_distribution.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_localization_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    loc_counts = df["localization"].value_counts()
    axes[0].barh(loc_counts.index, loc_counts.values,
                 color=sns.color_palette("viridis", len(loc_counts)))
    axes[0].set_title("Distribution of Lesion Localization", fontweight="bold")

    ct = pd.crosstab(df["localization"], df["dx"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    sns.heatmap(ct_norm, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[1],
                linewidths=0.5, cbar_kws={"label": "Percentage (%)"})
    axes[1].set_title("Diagnosis Distribution by Localization (Row-normalised %)",
                      fontweight="bold")

    out = os.path.join(output_dir, "localization_distribution.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def run_full_eda(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Run every EDA plot and return the list of output PNG paths."""
    return [
        plot_class_distribution(df, output_dir),
        plot_age_distribution(df, output_dir),
        plot_sex_distribution(df, output_dir),
        plot_localization_heatmap(df, output_dir),
    ]
