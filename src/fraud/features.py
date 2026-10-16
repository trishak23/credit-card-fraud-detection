"""Feature engineering and preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from raw columns (row-wise only, so no leakage)."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"])
    out["amount_ratio"] = out["amount"] / out["avg_amount_30d"].clip(lower=1.0)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["is_night"] = ((out["hour"] >= 23) | (out["hour"] <= 4)).astype(int)
    return out


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Impute + (optionally) scale numerics, one-hot the merchant category."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), config.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
    ])
