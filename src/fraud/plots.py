"""Figure generation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from . import config


def _save(fig, name: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=120)
    plt.close(fig)


def pr_curves(y, scores: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=n)
    ax.axhline(np.mean(y), color="gray", ls="--", label=f"base rate {np.mean(y):.4f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test set)", ylim=(0, 1.02))
    ax.legend(fontsize=8)
    _save(fig, "pr_curves.png")


def cost_curve(grid: np.ndarray, costs: np.ndarray, chosen: float, no_model: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, costs)
    ax.axvline(chosen, color="C3", ls="--", label=f"cost-optimal threshold {chosen:.3f}")
    ax.axhline(no_model, color="gray", ls=":", label="flag nothing")
    ax.set(xlabel="Score threshold", ylabel="Total cost on validation (OOF)", title="Cost-based threshold selection")
    ax.legend()
    _save(fig, "cost_vs_threshold.png")


def class_balance(df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df.is_fraud.value_counts().sort_index()
    axes[0].bar(["legit", "fraud"], counts.values, color=["C0", "C3"])
    axes[0].set_yscale("log")
    axes[0].set(title="Class balance (log scale)", ylabel="transactions")
    for lab, c in [(0, "C0"), (1, "C3")]:
        axes[1].hist(np.log1p(df.loc[df.is_fraud == lab, "amount"]), bins=40, alpha=0.6, density=True, color=c,
                     label=["legit", "fraud"][lab])
    axes[1].set(title="log(1+amount) by class", xlabel="log(1+amount)")
    axes[1].legend()
    _save(fig, "class_balance.png")


def importance(names, values, errs) -> None:
    order = np.argsort(values)[-12:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(names)[order], np.array(values)[order], xerr=np.array(errs)[order])
    ax.set(title="Permutation importance (drop in PR-AUC, test)", xlabel="PR-AUC decrease")
    _save(fig, "feature_importance.png")


def strategy_cv(folds: dict[str, list[float]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(list(folds.values()), tick_labels=list(folds.keys()))
    ax.set(ylabel="PR-AUC per CV fold", title="Imbalance strategies: cross-validated PR-AUC")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    _save(fig, "cv_strategies.png")
