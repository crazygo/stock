import unittest

import numpy as np
import torch

from research.after_open_3d5pct.train_multiscale_v6 import CModel, project_monotone, violations
from research.after_open_3d5pct.v6_data import (SymbolData, _encode, _group_state,
                                                 _outcomes, _prior_median, inputs_available)


class V6ContractTests(unittest.TestCase):
    def test_nested_probabilities_project_both_axes(self):
        p = np.array([[[.9, .1, .8], [.2, .4, .1], [.3, .2, .7]]])
        self.assertEqual(violations(project_monotone(p)), 0)
        self.assertEqual(violations(p), 1)

    def test_late_historical_bar_is_rejected_and_future_bar_is_irrelevant(self):
        arr = np.full((3, 192), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        raw = np.zeros((3, 192, 6), np.float32)
        seq = np.zeros((3, 192, 12), np.float32)
        v = SymbolData(raw, arr, seq, np.zeros((3, 17, 12), np.float32),
                       np.zeros((3, 12), np.float32), np.ones(3, bool),
                       np.zeros(3), np.ones(3), "synthetic", 0)
        session = {"open_at": "2026-09-10T13:30:00+00:00"}
        self.assertTrue(inputs_available(v, session, 1, 30))
        arr[0, 10] = np.datetime64("2026-09-10T16:00:00")
        self.assertFalse(inputs_available(v, session, 1, 30))
        arr[0, 10] = np.datetime64("2026-09-09T15:00:00")
        arr[1, 90] = np.datetime64("2026-09-10T20:00:00")
        self.assertTrue(inputs_available(v, session, 1, 30))

    def test_group_effective_first_week_day_and_late_version(self):
        day = "2026-09-14"  # Monday
        sessions = [{"open_at": "2026-09-14T13:30:00+00:00"}]
        own = {("2026-W38", "trend_15", "AAA"):
               ("trend_15:up", {"history_end":"2026-09-11"},
                {"effective_from":"2026-09-14T13:30:00+00:00",
                 "effective_to":"2026-09-21T13:30:00+00:00",
                 "feature_cutoff_at":"2026-09-11T20:00:00+00:00"})}
        views = {}
        for symbol, ret in (("AAA", .1), ("BBB", .02)):
            raw = np.zeros((1, 192, 6), np.float32)
            views[symbol] = SymbolData(raw, np.full((1,192), np.datetime64("NaT", "ns")),
                np.zeros((1,192,12)), np.zeros((1,17,12)), np.zeros((1,12)),
                np.ones(1,bool), np.array([ret]), np.ones(1), "synthetic", 0)
        peers = {("2026-W38", "trend_15:up"): {"AAA", "BBB"}}
        g = _group_state("AAA", 0, [day], sessions, views, own, peers)
        self.assertEqual(g[0, 9], 1)
        self.assertAlmostEqual(g[0, 3], .02)
        own[("2026-W38", "trend_15", "AAA")][2]["feature_cutoff_at"] = "2026-09-14T16:00:00+00:00"
        self.assertEqual(_group_state("AAA", 0, [day], sessions, views, own, peers)[0, 9], 0)

    def test_dual_share_classes_have_one_stable_issuer_representative(self):
        day = "2026-09-14"
        sessions = [{"open_at": "2026-09-14T13:30:00+00:00"}]
        version = {"effective_from":"2026-09-14T13:30:00+00:00",
                   "effective_to":"2026-09-21T13:30:00+00:00",
                   "feature_cutoff_at":"2026-09-11T20:00:00+00:00"}
        own = {("2026-W38", "trend_15", "AAA"):
               ("trend_15:up", {"history_end":"2026-09-11"}, version)}
        views = {}
        for symbol, ret in (("AAA", .1), ("GOOG", .02), ("GOOGL", .50)):
            views[symbol] = SymbolData(np.zeros((1,192,6)),
                np.full((1,192), np.datetime64("NaT", "ns")), np.zeros((1,192,14)),
                np.zeros((1,17,14)), np.zeros((1,14)), np.ones(1,bool),
                np.array([ret]), np.ones(1), "synthetic", 0)
        peers = {("2026-W38", "trend_15:up"): {"AAA", "GOOGL", "GOOG"}}
        for _ in range(3):
            g = _group_state("AAA", 0, [day], sessions, views, own, peers)
            self.assertAlmostEqual(g[0, 3], .02)
            self.assertAlmostEqual(g[0, 7], np.log1p(1)/5)

    def test_group_strategy_identity_survives_set_aggregation(self):
        torch.manual_seed(11)
        model = CModel(group=True, daily=False, input_channels=14, type_identity=True).eval()
        x5 = torch.zeros(1, 8, 192, 14)
        x60 = torch.zeros(1, 31, 17, 14)
        xday = torch.zeros(1, 126, 14)
        groups = torch.zeros(1, 6, 6, 10)
        # The first slot is trend15 and the fourth is volatility. Their
        # first category bit has different semantics despite the same bit.
        groups[:, :, 0, [0, 3, 9]] = torch.tensor([1.0, .05, 1.0])
        groups[:, :, 3, [0, 3, 9]] = torch.tensor([1.0, -.05, 1.0])
        swapped = groups.clone()
        swapped[:, :, 0] = groups[:, :, 3]
        swapped[:, :, 3] = groups[:, :, 0]
        with torch.no_grad():
            a = model(x5, x60, xday, groups)
            b = model(x5, x60, xday, swapped)
        self.assertGreater(float(torch.max(torch.abs(a-b))), 1e-8)

    def test_label_waits_for_latest_observed_bar(self):
        sessions = []
        for d in range(6):
            day = f"2026-09-{7+d:02d}"
            sessions.append({"session_date":day, "open_at":f"{day}T13:30:00+00:00",
                             "duration_minutes":390})
        raw = np.full((6,192,6), np.nan, np.float32)
        raw[:,66:144] = [100, 106, 99, 101, 1, 101]
        available = np.full((6,192), np.datetime64("2026-09-20T00:00:00"))
        available[5, 66+24] = np.datetime64("2026-09-21T00:00:00")
        v = SymbolData(raw, available, np.zeros((6,192,12)), np.zeros((6,17,12)),
                       np.zeros((6,12)), np.ones(6,bool), np.zeros(6), np.ones(6), "synthetic", 0)
        out = _outcomes(v, sessions, 0, {"thresholds":[.03,.05,.08],
                                         "availability_assumption_seconds":1}, set())
        self.assertIsNotNone(out)
        self.assertEqual(out[3], "2026-09-21T00:00:00+00:00")
        self.assertEqual(out[0][1,1], 1)

    def test_appended_future_day_cannot_rewrite_past_features(self):
        raw = np.array([[[100,101,99,100.5,10,1000], [100.5,102,100,101,11,1100]],
                        [[101,102,100,101.5,12,1200], [101.5,103,101,102,13,1300]],
                        [[102,110,101,109,100,10000], [109,120,108,119,100,20000]]],
                       dtype=np.float32)
        duration = np.full((3,2), 5, np.float32)
        types = np.ones((3,2), np.float32)
        position = np.tile(np.array([0,.5],np.float32),(3,1))
        starts = np.arange(6,dtype=np.float64).reshape(3,2)*300e9
        two = _encode(raw[:2], duration[:2], types[:2], position[:2], starts[:2],
                      _prior_median(raw[:2,:,5]))
        three = _encode(raw, duration, types, position, starts, _prior_median(raw[:,:,5]))
        np.testing.assert_array_equal(two, three[:2])


if __name__ == "__main__":
    unittest.main()
