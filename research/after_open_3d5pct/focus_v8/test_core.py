import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .core import FocusC, mature_prior, splits, within_auc, path_summary, focus_weights
from .support import build_c


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

    def test_training_support_is_frozen_and_checkpointed(self):
        torch.set_num_threads(1)
        a = torch.zeros(12, 8, 192, 14)
        b = torch.zeros(12, 31, 17, 14)
        d = torch.zeros(12, 126, 14)
        g = torch.zeros(12, 6, 6, 10)
        a[..., 10] = 1; b[..., 10] = 1
        d[:, 60:, 10] = 1; g[:, :, [0, 3, 4, 5], 9] = 1
        p = torch.full((12, 9), .5)
        rows = pd.DataFrame({"session_date": pd.date_range("2026-03-01", periods=12).strftime("%Y-%m-%d")})
        for route in ("C_group", "C_no_group", "C_no_daily"):
            recipe = {"support": True, "representation": True}
            model = build_c(route, recipe).eval()
            model.fit_support((a,b,d,g,p), rows, np.arange(10))
            changed_d, changed_g = d.clone(), g.clone()
            changed_d[:, :60] = float("nan")
            changed_g[:, :, 1:3] = float("nan")
            with torch.no_grad():
                original = model(a,b,d,g,p)
                revised = model(a,b,changed_d,changed_g,p)
            torch.testing.assert_close(original, revised)
            replay = build_c(route, recipe).eval()
            replay.load_state_dict(model.state_dict())
            with torch.no_grad():
                torch.testing.assert_close(replay(a,b,changed_d,changed_g,p), original)

    def test_reject_changed_dataset_before_reading_rows(self):
        from .run import load_data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/"dataset").mkdir()
            for name in ("features.npz", "rows.parquet", "manifest.json"):
                (root/"dataset"/name).write_bytes(b"changed")
            (root/"dataset_identity.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "dataset contents changed"):
                load_data(root)

    def test_reject_cache_content_change(self):
        from .run import load_data, digest, HERE
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ds = root/"dataset"; ds.mkdir()
            rows = pd.DataFrame({"symbol": ["A"], "decision_at": ["2026-03-01T15:30Z"],
                                 "label_end_at": ["2026-03-06T15:30Z"], "label_available_at": ["2026-03-06T15:30Z"]})
            rows.to_parquet(ds/"rows.parquet")
            np.savez(ds/"features.npz", x5=np.zeros((1,8,192,14),np.float32),
                     x60=np.zeros((1,31,17,14),np.float32), xday=np.zeros((1,126,14),np.float32),
                     group_seq=np.zeros((1,6,6,10),np.float32), y=np.zeros((1,3,3),np.float32))
            (ds/"manifest.json").write_text('{}')
            identity = {n: digest(ds/n) for n in ("features.npz", "rows.parquet", "manifest.json")}
            np.save(root/"path_summary.npy", np.zeros((1,84)))
            meta = {"dataset": identity, "transform": digest(HERE/"core.py"), "cache_sha256": digest(root/"path_summary.npy")}
            (root/"path_summary_identity.json").write_text(json.dumps(meta))
            np.save(root/"path_summary.npy", np.ones((1,84)))
            with self.assertRaisesRegex(ValueError, "cache contents changed"):
                load_data(root)


if __name__ == "__main__":
    unittest.main()
