"""Central configuration (paths, seeds, cost assumptions)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_CSV = DATA_DIR / "sample_transactions.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
METRICS_JSON = REPORT_DIR / "metrics.json"

SEED = 42
N_TRANSACTIONS = 60_000
TARGET_FRAUD_RATE = 0.007
TEST_SIZE = 0.25
N_FOLDS = 4
UNDERSAMPLE_RATIO = 10  # legit : fraud after undersampling

# Business cost model: a missed fraud costs the transaction amount, a false
# alarm costs a fixed manual-review fee.
REVIEW_COST = 5.0

CATEGORICAL = ["merchant_category"]
NUMERIC = [
    "log_amount", "hour_sin", "hour_cos", "is_night", "is_online", "distance_from_home_km",
    "tx_last_24h", "amount_ratio", "card_age_days", "foreign_country", "day_of_week", "new_merchant",
]
