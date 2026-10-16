"""Single entry point: data -> CV of strategies -> test evaluation -> reports."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from fraud import config, plots
from fraud.data import generate_transactions, save_sample
from fraud.evaluate import best_cost_threshold, summarize, total_cost
from fraud.features import add_features
from fraud.train import anomaly_score, cv_scores, fit_isolation_forest, fit_model, proba

log = logging.getLogger("pipeline")
SUPERVISED = ["baseline_prior", "logreg_class_weight", "hgb_unweighted", "hgb_class_weight", "hgb_undersample"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = generate_transactions()
    save_sample(raw)
    if args.data_only:
        return
    df = add_features(raw)
    X, y = df.drop(columns="is_fraud"), df["is_fraud"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)
    log.info("train %d (fraud %d) | test %d (fraud %d)", len(y_tr), y_tr.sum(), len(y_te), y_te.sum())

    # 1) Cross-validated comparison of imbalance strategies on the training set.
    folds, oof = cv_scores(SUPERVISED, X_tr, y_tr)

    # 2) Fit on full train, score the held-out test set.
    models = {n: fit_model(n, X_tr, y_tr) for n in SUPERVISED}
    test_scores = {n: proba(m, X_te) for n, m in models.items()}
    iso = fit_isolation_forest(X_tr)
    test_scores["isolation_forest (unsupervised)"] = anomaly_score(iso, X_te)

    # 3) Cost-based threshold chosen on out-of-fold train scores, applied once to test.
    best = max(SUPERVISED, key=lambda n: np.mean(folds[n]))
    amt_tr, amt_te = X_tr["amount"].to_numpy(), X_te["amount"].to_numpy()
    thr, val_cost = best_cost_threshold(y_tr, oof[best], amt_tr)
    flagged_cost = test_scores[best] >= thr
    flagged_default = test_scores[best] >= 0.5
    grid = np.unique(np.quantile(oof[best], np.linspace(0.5, 0.99999, 300)))
    costs = np.array([total_cost(y_tr, oof[best] >= t, amt_tr) for t in grid])
    plots.cost_curve(grid, costs, thr, total_cost(y_tr, np.zeros_like(y_tr), amt_tr))

    # Isolation forest flags the same number of transactions as the cost-optimal supervised model.
    iso_thr = np.sort(test_scores["isolation_forest (unsupervised)"])[::-1][int(flagged_cost.sum()) - 1]
    iso_flag = test_scores["isolation_forest (unsupervised)"] >= iso_thr

    res = {}
    for n, s in test_scores.items():
        res[n] = {"pr_auc": float(average_precision_score(y_te, s))}
    res[best + "@cost_threshold"] = summarize(y_te, test_scores[best], flagged_cost)
    res[best + "@0.5"] = summarize(y_te, test_scores[best], flagged_default)
    res["isolation_forest@same_alert_budget"] = summarize(y_te, test_scores["isolation_forest (unsupervised)"], iso_flag)
    cost = {
        "no_model": total_cost(y_te, np.zeros_like(y_te), amt_te),
        "default_0.5_threshold": total_cost(y_te, flagged_default, amt_te),
        "cost_optimal_threshold": total_cost(y_te, flagged_cost, amt_te),
        "isolation_forest_same_budget": total_cost(y_te, iso_flag, amt_te),
    }

    # 4) Plots + importance.
    plots.class_balance(raw)
    plots.pr_curves(y_te, test_scores)
    plots.strategy_cv(folds)
    pi = permutation_importance(models[best], X_te, y_te, scoring="average_precision", n_repeats=3,
                                random_state=config.SEED, n_jobs=1)
    plots.importance(list(X_te.columns), pi.importances_mean, pi.importances_std)
    top = sorted(zip(X_te.columns, pi.importances_mean), key=lambda t: -t[1])[:5]

    metrics = {
        "data": {"synthetic": True, "n_rows": int(len(raw)), "n_fraud": int(y.sum()),
                 "fraud_rate": float(y.mean()), "n_test": int(len(y_te)), "n_test_fraud": int(y_te.sum())},
        "cv_pr_auc_mean": {n: float(np.mean(v)) for n, v in folds.items()},
        "cv_pr_auc_std": {n: float(np.std(v)) for n, v in folds.items()},
        "best_strategy": best,
        "cost_threshold": float(thr),
        "validation_cost_at_threshold": val_cost,
        "test": res,
        "test_cost": cost,
        "test_cost_saving_vs_no_model_pct": float(100 * (1 - cost["cost_optimal_threshold"] / cost["no_model"])),
        "top_features": [[a, float(b)] for a, b in top],
    }
    config.REPORT_DIR.mkdir(exist_ok=True)
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2))
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": models[best], "threshold": float(thr), "name": best}, config.MODEL_DIR / "fraud_model.joblib")
    log.info("done: best=%s", best)


if __name__ == "__main__":
    main()
