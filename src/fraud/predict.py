"""Score new transactions with the saved model: ``python -m fraud.predict in.csv out.csv``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from . import config
from .features import add_features


def load_artifact(path: Path = config.MODEL_DIR / "fraud_model.joblib") -> dict:
    return joblib.load(path)


def predict_df(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """Return df with fraud probability and a flag using the cost-optimal threshold."""
    artifact = artifact or load_artifact()
    X = add_features(df)
    p = artifact["model"].predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_score"] = p
    out["flagged"] = (p >= artifact["threshold"]).astype(int)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    a = ap.parse_args()
    res = predict_df(pd.read_csv(a.input_csv))
    res.to_csv(a.output_csv, index=False)
    print(json.dumps({"rows": len(res), "flagged": int(res.flagged.sum())}))


if __name__ == "__main__":
    main()
