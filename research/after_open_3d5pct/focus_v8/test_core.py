import unittest

import numpy as np
import pandas as pd
import torch

from .core import FocusC, mature_prior, splits, within_auc, path_summary, focus_weights


class CausalityTests(unittest.TestCase):
    def test_prior_never_uses_current_future_or_late_label(self):
        dates = pd.date_range("2026-01-01", periods=15, tz="UTC")
        rows = pd.DataFrame({"symbol": ["A"]*15, "decision_at": dates,
                             "label_end_at": dates+pd.Timedelta(days=5),
                             "label_available_at": dates+pd.Timedelta(days=5, seconds=1)})
        y = np.tile(np.arange(15) % 2, (9, 1)).T.astype(float)
        p, n = mature_prior(rows, y)
        self.assertEqual(n[6], 1)
        self.assertEqual(n[5], 0)
        changed = y.copy()
        changed[1:] = 1-changed[1:]
        p2, _ = mature_prior(rows, changed)
        np.testing.assert_array_equal(p[:7], p2[:7])
        shortened, _ = mature_prior(rows.iloc[:8], y[:8])
        np.testing.assert_array_equal(p[:8], shortened)
        rows.loc[0, "label_available_at"] = dates[-1]
        _, late_n = mature_prior(rows, y)
        self.assertEqual(late_n[6], 0)

    def test_purge_uses_both_actual_end_and_availability(self):
        rows = pd.DataFrame({"session_date": ["2026-04-10"]*3,
            "label_end_at": ["2026-04-19T20:00Z", "2026-04-21T20:00Z", "2026-04-19T20:00Z"],
            "label_available_at": ["2026-04-19T20:00Z", "2026-04-19T20:00Z", "2026-04-21T20:00Z"]})
        f = splits(rows, {"tune": "2026-04-20", "cal": "2026-05-11", "eval": "2026-06-01", "end": "2026-06-27"})
        self.assertEqual(f["fit"].tolist(), [0])

    def test_same_stock_auc_ignores_cross_stock_level(self):
        s = np.array(["A", "A", "B", "B"])
        y = np.array([0, 1, 0, 1])
        self.assertEqual(within_auc(s, y, np.array([.8, .9, .1, .2])), 1.)
        self.assertEqual(within_auc(s, y, np.array([.9, .8, .2, .1])), 0.)

    def test_group_weight_overlap_does_not_duplicate_rows(self):
        rows = pd.DataFrame({"symbol": ["ALAB", "AMD", "LITE", "MU", "STX", "OTHER"]})
        w = focus_weights(rows)
        self.assertEqual(len(w), len(rows))
        self.assertAlmostEqual(w.sum(), 1.)
        self.assertEqual(w[-1], 0.)

    def test_route_feature_isolation(self):
        torch.set_num_threads(1)
        x5 = torch.zeros(2, 8, 192, 14)
        x60 = torch.zeros(2, 31, 17, 14)
        day = torch.zeros(2, 126, 14)
        group = torch.randn(2, 6, 6, 10)
        for x in (x5, x60, day):
            x[..., 10] = 1
        p = torch.full((2,9), .5)
        for recipe in ({}, {"representation": True, "capacity": True, "curves": True, "prior": True}):
            model = FocusC("C_no_daily", recipe).eval()
            with torch.no_grad():
                a = model(x5, x60, torch.empty(0), group, p)
                b = model(x5, x60, torch.full_like(day, float("nan")), group, p)
            torch.testing.assert_close(a, b)
            model = FocusC("C_no_group", recipe).eval()
            with torch.no_grad():
                a = model(x5, x60, day, torch.empty(0), p)
                b = model(x5, x60, day, torch.full_like(group, float("nan")), p)
            torch.testing.assert_close(a, b)

    def test_path_summary_accounts_for_gap_and_mask(self):
        x = np.zeros((1, 5, 14), np.float32)
        x[0, :3, 10] = 1
        x[0, :3, 12] = np.log([100, 110, 121])/5
        x[0, 3:, 12] = 100  # Padding must not create a price jump.
        data = {k: x.copy() for k in ("x5", "x60", "xday")}
        z = path_summary(data)
        self.assertAlmostEqual(z[0, 0], np.log(1.21), places=5)
        self.assertEqual(z[0, 2], 0)


if __name__ == "__main__":
    unittest.main()
