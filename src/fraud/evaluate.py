"""Metrics, cost model and threshold selection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from . import config


def total_cost(y: np.ndarray, flagged: np.ndarray, amount: np.ndarray, review_cost: float = config.REVIEW_COST) -> float:
    """Missed fraud costs its amount; every flagged transaction costs a review fee."""
    y, flagged = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    missed = amount[y & ~flagged].sum()
    return float(missed + review_cost * flagged.sum())


def best_cost_threshold(y: np.ndarray, score: np.ndarray, amount: np.ndarray,
                        review_cost: float = config.REVIEW_COST, n_grid: int = 400) -> tuple[float, float]:
    """Return (threshold, cost) minimising total cost over a quantile grid of scores."""
    grid = np.unique(np.quantile(score, np.linspace(0.5, 0.99999, n_grid)))
    costs = [total_cost(y, score >= t, amount, review_cost) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def summarize(y: np.ndarray, score: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    """Threshold-free and thresholded metrics."""
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(precision_score(y, flagged, zero_division=0)),
        "recall": float(recall_score(y, flagged, zero_division=0)),
        "f1": float(f1_score(y, flagged, zero_division=0)),
        "n_flagged": int(np.sum(flagged)),
    }
