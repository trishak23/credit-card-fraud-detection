import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from fraud.data import generate_transactions
from fraud.evaluate import best_cost_threshold, total_cost
from fraud.features import add_features
from fraud.train import fit_model, proba, undersample


class TestFraud(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_transactions(n=8000, fraud_rate=0.02, seed=1)

    def test_generator_deterministic_and_imbalanced(self):
        again = generate_transactions(n=8000, fraud_rate=0.02, seed=1)
        self.assertTrue(self.df.equals(again))
        rate = self.df.is_fraud.mean()
        self.assertTrue(0.005 < rate < 0.05)

    def test_undersample_keeps_all_positives_and_ratio(self):
        df = add_features(self.df)
        y = df.is_fraud.to_numpy()
        Xs, ys = undersample(df.drop(columns="is_fraud"), y, ratio=5)
        self.assertEqual(ys.sum(), y.sum())
        self.assertEqual((ys == 0).sum(), 5 * y.sum())

    def test_cost_function(self):
        y = np.array([1, 0, 1, 0]); amt = np.array([100., 10., 50., 10.])
        self.assertEqual(total_cost(y, np.array([0, 0, 0, 0]), amt, 5.0), 150.0)
        self.assertEqual(total_cost(y, np.array([1, 1, 0, 0]), amt, 5.0), 50.0 + 10.0)

    def test_threshold_beats_flagging_nothing(self):
        rng = np.random.default_rng(0)
        y = (rng.random(5000) < 0.05).astype(int)
        s = rng.random(5000) * 0.5 + 0.5 * y
        amt = np.full(5000, 100.0)
        thr, cost = best_cost_threshold(y, s, amt)
        self.assertLess(cost, total_cost(y, np.zeros(5000), amt))

    def test_model_probabilities_valid_and_better_than_chance(self):
        from sklearn.metrics import average_precision_score
        df = add_features(self.df)
        y = df.is_fraud.to_numpy(); X = df.drop(columns="is_fraud")
        m = fit_model("hgb_class_weight", X.iloc[:6000], y[:6000])
        p = proba(m, X.iloc[6000:])
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertGreater(average_precision_score(y[6000:], p), 2 * y[6000:].mean())


if __name__ == "__main__":
    unittest.main()
