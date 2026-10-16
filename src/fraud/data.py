"""Seeded synthetic transaction generator (NOT real card data)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)
CATEGORIES = ["grocery", "restaurant", "fuel", "travel", "electronics", "online_retail", "entertainment", "other"]
CAT_P = [0.24, 0.16, 0.12, 0.05, 0.06, 0.20, 0.07, 0.10]
# Additive log-odds contribution of each category to fraud risk.
CAT_RISK = {"grocery": -0.4, "restaurant": -0.2, "fuel": 0.0, "travel": 0.8, "electronics": 1.0,
            "online_retail": 0.5, "entertainment": 0.2, "other": 0.0}


def generate_transactions(n: int = config.N_TRANSACTIONS, fraud_rate: float = config.TARGET_FRAUD_RATE,
                          seed: int = config.SEED) -> pd.DataFrame:
    """Generate ``n`` transactions whose fraud probability follows a latent logistic model.

    The intercept is calibrated by bisection so the realised base rate is close to ``fraud_rate``.
    """
    rng = np.random.default_rng(seed)
    cat = rng.choice(CATEGORIES, size=n, p=CAT_P)
    is_online = ((cat == "online_retail") | (rng.random(n) < 0.15)).astype(int)
    hour = np.where(rng.random(n) < 0.5, rng.normal(14, 4, n), rng.normal(19, 3, n)) % 24
    hour = np.floor(hour).astype(int)
    avg_amount_30d = rng.lognormal(3.6, 0.5, n)
    amount = avg_amount_30d * rng.lognormal(0, 0.8, n)
    distance = rng.gamma(1.5, 6.0, n) * np.where(is_online == 1, 3.0, 1.0)
    tx_24h = rng.poisson(1.8, n)
    card_age = rng.integers(10, 3500, n)
    foreign = (rng.random(n) < 0.04).astype(int)
    dow = rng.integers(0, 7, n)
    new_merchant = (rng.random(n) < 0.12).astype(int)  # first purchase in this category for the card
    burst = (rng.random(n) < 0.05).astype(int)  # rapid-velocity burst
    tx_24h = tx_24h + burst * rng.integers(3, 8, n)

    is_night = ((hour >= 23) | (hour <= 4)).astype(int)
    z = (2.2 * np.log(amount / avg_amount_30d).clip(-2, 4) + 2.6 * is_night + 1.3 * is_online
         + 0.02 * np.minimum(distance, 200) + 0.3 * tx_24h + 3.0 * burst + 3.0 * foreign
         + 2.2 * new_merchant - 0.0004 * card_age + np.array([CAT_RISK[c] for c in cat]))
    z = 1.8 * (z + rng.normal(0, 0.5, n))  # irreducible noise: fraud is not perfectly separable

    lo, hi = -30.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > fraud_rate:
            hi = mid
        else:
            lo = mid
    p = 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    is_fraud = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "amount": amount.round(2), "hour": hour, "day_of_week": dow, "merchant_category": cat,
        "is_online": is_online, "distance_from_home_km": distance.round(1), "tx_last_24h": tx_24h,
        "avg_amount_30d": avg_amount_30d.round(2), "card_age_days": card_age, "foreign_country": foreign,
        "new_merchant": new_merchant,
        "is_fraud": is_fraud,
    })
    # Realistic missingness in a field that is often unavailable.
    df.loc[rng.random(n) < 0.03, "distance_from_home_km"] = np.nan
    log.info("Generated %d rows, fraud rate %.4f%%", n, 100 * df.is_fraud.mean())
    return df


def save_sample(df: pd.DataFrame, path: Path = config.SAMPLE_CSV, n_legit: int = 6000, seed: int = config.SEED) -> Path:
    """Write a small committed sample (all fraud rows + a random legit subset)."""
    legit = df[df.is_fraud == 0].sample(n_legit, random_state=seed)
    out = pd.concat([df[df.is_fraud == 1], legit]).sample(frac=1, random_state=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load a CSV with the expected schema, or generate the full synthetic set."""
    if path is not None:
        return pd.read_csv(path)
    return generate_transactions()
