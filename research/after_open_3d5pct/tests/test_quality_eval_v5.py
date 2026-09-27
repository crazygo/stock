"""Quality action coverage and date-axis uncertainty contracts."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.hourly_v3_data import Bundle
from research.after_open_3d5pct.evaluate_quality_v5 import _weighted_es_loss
from research.after_open_3d5pct.quality_eval_v5 import (
    _block_ci, choose_quality_recommendations, matched_history_rows)
from research.after_open_3d5pct.quality_pool_v5 import FEATURE_COLUMNS
from research.after_open_3d5pct.train_quality_v5 import (
    _static_v5, _tabular_v5, _training_indices)


class QualityEvalV5Tests(unittest.TestCase):
    def test_same_time_weighted_tail_uses_full_pool_and_fractional_boundary(self):
        # Two date/hour cross sections have different quality weights. The
        # worst half of weighted *all-pool* returns uses 0.5 of the first loss
        # and 1.0 of the second; failed-touch-only ES would be a different set.
        returns = np.array([-.20, .10, -.05, .05])
        weights = np.array([.5, .5, 1., 1.])
        self.assertAlmostEqual(_weighted_es_loss(returns, weights, .5), .10)
        self.assertAlmostEqual(_weighted_es_loss(np.array([.02, .04]),
                                                  np.ones(2), .5), -.02)

    def test_pooled_control_excludes_quality_and_both_families_get_ordered_features(self):
        row = {"cutoff_et": "10:30", "h_range_actual": .01, "h_range_relative": 1.,
               "h_dollar_actual_log": 15., "h_dollar_relative": 1.,
               "b_range_actual": .02, "b_range_relative": 1.,
               "b_dollar_actual_log": 14., "b_dollar_relative": 1.}
        rows = pd.DataFrame([row, row])
        seq = np.zeros((2, 1, 7), dtype=np.float32)
        bundle = Bundle(rows=rows, h_seq=seq, post_seq=seq, pre_seq=seq,
                        b_seq=seq, exclusions={}, coverage={}, source_paths=[])
        quality = pd.DataFrame([{c: 0. for c in FEATURE_COLUMNS} for _ in range(2)])
        quality.loc[0, "h_quality_rate_63"] = .7
        quality.loc[0, "h_quality_mature_days_63"] = 63
        quality.loc[0, "h_quality_rate_available_mask"] = 1
        quality.loc[1, "h_quality_rate_63"] = np.nan
        quality.loc[1, "h_quality_rate_available_mask"] = 0
        control = _tabular_v5(bundle, quality, "HB", False)
        aware = _tabular_v5(bundle, quality, "HB", True)
        self.assertFalse(any(c in control for c in FEATURE_COLUMNS))
        self.assertEqual(aware.columns[-len(FEATURE_COLUMNS):].tolist(), list(FEATURE_COLUMNS))
        base_static, base_names = _static_v5(bundle, quality, "HB", False)
        new_static, names = _static_v5(bundle, quality, "HB", True)
        self.assertEqual(names[-len(FEATURE_COLUMNS):], list(FEATURE_COLUMNS))
        self.assertEqual(new_static.shape[1], base_static.shape[1] + len(FEATURE_COLUMNS))
        self.assertEqual(float(new_static[0, names.index("h_quality_mature_days_63")]), 1.)
        self.assertEqual(float(new_static[1, names.index("h_quality_rate_available_mask")]), 0.)
        self.assertEqual(len(base_names), base_static.shape[1])
        bundle.rows["h_quality_unregistered"] = 1
        with self.assertRaisesRegex(ValueError, "leaked"):
            _tabular_v5(bundle, quality, "HB", False)

    def test_quality_only_training_scope_uses_each_row_own_asof_identity(self):
        indices = {"train": np.array([0, 1, 2, 3]), "early": np.array([4, 5])}
        quality = np.array([False, True, False, True, True, False])
        train, train_q, early_q = _training_indices(indices, quality, "decision_time_quality_rows_only")
        self.assertEqual(train.tolist(), [1, 3])
        self.assertEqual(train_q.tolist(), [1, 3])
        self.assertEqual(early_q.tolist(), [4])
        pooled, _, _ = _training_indices(indices, quality, "pooled")
        self.assertEqual(pooled.tolist(), [0, 1, 2, 3])

    def test_zero_recommendation_and_same_time_history_coverage(self):
        frame = pd.DataFrame({"session_date": ["2026-07-01"] * 3,
                              "cutoff_et": ["10:30"] * 3,
                              "symbol": ["A", "B", "C"],
                              "quality_eligible": [True] * 3,
                              "p_touch": [.7, .59, .1],
                              "expected_net": [.01, .01, .01],
                              "failure_q10_net": [-.1, -.1, -.1],
                              "h_quality_rate_63": [.62, .8, .61]})
        scenario = {"opportunity_budget_fraction_max_each_quality_date_hour": .05,
                    "candidate_calibrated_p_touch_min": .6,
                    "predicted_expected_net_min": 0,
                    "predicted_failure_q10_net_min": -.15}
        chosen = choose_quality_recommendations(frame, scenario)
        self.assertEqual(chosen.tolist(), [True, False, False])
        self.assertEqual(matched_history_rows(frame, chosen).symbol.tolist(), ["B"])
        frame.loc[0, "p_touch"] = .59
        self.assertFalse(choose_quality_recommendations(frame, scenario).any())

    def test_sparse_recommendations_keep_missing_sessions_in_blocks(self):
        days = pd.bdate_range("2026-07-01", periods=20).strftime("%Y-%m-%d").tolist()
        sparse = pd.DataFrame({"session_date": [days[0], days[-1]], "lift": [.2, -.2]})
        result = _block_ci(sparse, "lift", days, 5, 500, 11)
        self.assertEqual(result["ET_dates"], 20)
        self.assertEqual(result["status"], "bootstrap_empty_selected_draw")
        self.assertIsNone(result["lower"])


if __name__ == "__main__":
    unittest.main()
