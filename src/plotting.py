"""Result plots: confusion matrices, ROC curves, feature importance, comparison bars.

These helpers take the ``results`` dict produced by :mod:`src.models` and
re-produce exactly the figures expected by the Report notebook.
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)


def results_to_dataframe(results: dict[str, dict]) -> pd.DataFrame:
    """Flatten a ``results`` dict into a tidy comparison DataFrame."""
    rows = []
    for r in results.values():
        rows.append({
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Macro F1": r["macro_f1"],
            "Weighted F1": r["weighted_f1"],
            "ROC-AUC": r["roc_auc"] if r["roc_auc"] is not None else np.nan,
        })
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)


def plot_model_comparison(comp_df: pd.DataFrame, output_dir: str,
                          title: str = "Model Performance Comparison") -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, metric in zip(axes, ["Accuracy", "Macro F1", "Weighted F1"]):
        plot_data = comp_df.sort_values(metric, ascending=True)
        bars = ax.barh(plot_data["Model"], plot_data[metric],
                       color=sns.color_palette("Set2", len(plot_data)))
        ax.set_title(metric, fontweight="bold")
        ax.set_xlim(0, 1)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", fontsize=10)

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrices(results: dict[str, dict], y_test: np.ndarray,
                            class_names: np.ndarray, output_dir: str) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes_flat = axes.flatten()
    labels = list(class_names)
    for idx, r in enumerate(results.values()):
        if idx >= len(axes_flat):
            break
        cm = confusion_matrix(y_test, r["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=axes_flat[idx], cmap="Blues", values_format="d", colorbar=False,
        )
        axes_flat[idx].set_title(
            f"{r['model_name']}\n(Acc={r['accuracy']:.3f}, F1={r['macro_f1']:.3f})",
            fontweight="bold", fontsize=11,
        )
        axes_flat[idx].tick_params(axis="x", rotation=45)
    for j in range(len(results), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_roc_curves(results: dict[str, dict], y_test: np.ndarray,
                    class_names: np.ndarray, output_dir: str) -> str:
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    for ci in range(n_classes):
        ax = axes[ci // 4][ci % 4]
        for r in results.values():
            if r["y_proba"] is None:
                continue
            y_bin = (y_test == ci).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, r["y_proba"][:, ci])
            ax.plot(fpr, tpr, label=f"{r['model_name']} ({auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(f"ROC — {class_names[ci]}", fontweight="bold")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=7)

    if n_classes < 8:
        axes[1][3].set_visible(False)

    plt.suptitle("ROC Curves per Class (One-vs-Rest)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_importance(model: Any, feature_names: list[str],
                            image_feat_cols: list[str], output_dir: str,
                            top_n: int = 20) -> str:
    """LightGBM / any tree model with ``feature_importances_``."""
    imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    top = imp.head(top_n)
    axes[0].barh(range(top_n), top["Importance"].values[::-1],
                 color=sns.color_palette("viridis", top_n))
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top["Feature"].values[::-1])
    axes[0].set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    axes[0].set_xlabel("Importance")

    imp["Category"] = imp["Feature"].apply(
        lambda x: "Image" if x in image_feat_cols else
                  "Age" if x == "age" else "Metadata (categorical)"
    )
    cat_imp = imp.groupby("Category")["Importance"].sum()
    bars = axes[1].bar(cat_imp.index, cat_imp.values,
                       color=["#66c2a5", "#fc8d62", "#8da0cb"][:len(cat_imp)])
    axes[1].set_title("Total Importance by Feature Category", fontweight="bold")
    for bar in bars:
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 5, f"{bar.get_height():.0f}",
                     ha="center", fontsize=11)

    out = os.path.join(output_dir, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_set_comparison(fs_df: pd.DataFrame, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(fs_df))
    width = 0.2
    for i, metric in enumerate(["Accuracy", "Macro F1", "Weighted F1"]):
        bars = ax.bar(x + i * width, fs_df[metric], width, label=metric)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005, f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Performance by Feature Set (LightGBM)", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(fs_df["Feature Set"])
    ax.legend()
    ax.set_ylim(0, 1.1)
    out = os.path.join(output_dir, "feature_set_comparison.png")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
