"""Model zoo, imbalance strategies and cross-validated scoring."""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor

log = logging.getLogger(__name__)


def undersample(X: pd.DataFrame, y: np.ndarray, ratio: int = config.UNDERSAMPLE_RATIO,
                seed: int = config.SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep all positives and ``ratio`` negatives per positive (random). Apply to TRAIN folds only."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    keep_neg = rng.choice(neg, size=min(len(neg), ratio * len(pos)), replace=False)
    idx = np.sort(np.concatenate([pos, keep_neg]))
    return X.iloc[idx], y[idx]


def make_model(name: str) -> Pipeline:
    """Return an unfitted pipeline for a supervised strategy name."""
    if name == "baseline_prior":
        return Pipeline([("pre", build_preprocessor()), ("clf", DummyClassifier(strategy="prior"))])
    if name == "logreg_class_weight":
        clf = LogisticRegression(class_weight="balanced", max_iter=500, C=0.5)
        return Pipeline([("pre", build_preprocessor()), ("clf", clf)])
    if name == "hgb_unweighted":
        clf = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.12, max_leaf_nodes=15, early_stopping=False, random_state=config.SEED)
    elif name in ("hgb_class_weight",):
        clf = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.12, max_leaf_nodes=15, early_stopping=False, class_weight="balanced",
                                             random_state=config.SEED)
    elif name == "hgb_undersample":
        clf = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.12, max_leaf_nodes=15, early_stopping=False, random_state=config.SEED)
    else:
        raise ValueError(f"unknown model {name}")
    return Pipeline([("pre", build_preprocessor(scale=False)), ("clf", clf)])


def fit_model(name: str, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """Fit a strategy; ``*_undersample`` models resample the training data first."""
    model = make_model(name)
    if name.endswith("_undersample"):
        X, y = undersample(X, y)
    return model.fit(X, y)


def fit_isolation_forest(X: pd.DataFrame, seed: int = config.SEED) -> Pipeline:
    """Unsupervised anomaly detector: never sees labels."""
    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1)
    return Pipeline([("pre", build_preprocessor()), ("clf", clf)]).fit(X)


def anomaly_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous."""
    return -model.score_samples(X)


def proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def cv_scores(names: list[str], X: pd.DataFrame, y: np.ndarray, n_folds: int = config.N_FOLDS,
              seed: int = config.SEED) -> tuple[dict[str, list[float]], dict[str, np.ndarray]]:
    """Stratified CV. Returns per-fold PR-AUC and out-of-fold scores for each strategy."""
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    folds: dict[str, list[float]] = {n: [] for n in names}
    oof = {n: np.zeros(len(y)) for n in names}
    for tr, va in skf.split(X, y):
        for n in names:
            m = fit_model(n, X.iloc[tr], y[tr])
            s = proba(m, X.iloc[va])
            oof[n][va] = s
            folds[n].append(float(average_precision_score(y[va], s)))
        log.info("fold done")
    return folds, oof
