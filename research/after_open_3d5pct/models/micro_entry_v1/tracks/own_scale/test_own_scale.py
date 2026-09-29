"""The past-60 level ignores the current score. No market data."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.micro_entry_v1.tracks.common import first_trigger
from research.after_open_3d5pct.models.micro_entry_v1.tracks.own_scale.run import (
    past_60_level,
    prior_impulse_gap,
    score_levels,
    with_impulse_gap60,
)


class Past60LevelTests(unittest.TestCase):
    def test_shifted_60_quantile_does_not_use_the_current_score(self):
        score = pd.Series(np.arange(120, dtype=float))
        level = past_60_level(score)
        index = 80
        prior = score.iloc[index - 60:index]
        self.assertEqual(len(prior), 60)
        self.assertAlmostEqual(level.iloc[index], prior.quantile(0.80))
        self.assertNotAlmostEqual(level.iloc[index], score.iloc[index - 59:index + 1].quantile(0.80))
        self.assertNotAlmostEqual(level.iloc[index], score.iloc[index - 61:index].quantile(0.80))
        changed = score.copy()
        changed.iloc[index] = 0.0
        self.assertAlmostEqual(past_60_level(changed).iloc[index], level.iloc[index])
        inclusive = changed.rolling(60, min_periods=40).quantile(0.80)
        base_inclusive = score.rolling(60, min_periods=40).quantile(0.80)
        self.assertNotAlmostEqual(inclusive.iloc[index], base_inclusive.iloc[index])

    def test_thirty_nine_priors_produce_no_buy(self):
        score = pd.Series(np.full(40, 0.99))
        level = past_60_level(score)
        self.assertTrue(np.isposinf(level.iloc[39]))
        self.assertFalse(bool(score.iloc[39] >= level.iloc[39]))
        frame = pd.DataFrame({
            "symbol": "SYN",
            "session_date": "2026-01-02",
            "decision_at": pd.date_range("2026-01-02 11:30", periods=40, freq="5min"),
            "score": score,
            "level": level.to_numpy(),
            "y_3d_5pct": 1.0,
        })
        self.assertEqual(first_trigger(frame)["buys"], 0)
        longer = pd.Series(np.arange(41, dtype=float))
        level_40 = past_60_level(longer)
        self.assertTrue(np.isfinite(level_40.iloc[40]))
        self.assertAlmostEqual(level_40.iloc[40], longer.iloc[:40].quantile(0.80))
        self.assertNotAlmostEqual(level_40.iloc[40], longer.iloc[:40].quantile(0.50))

    def test_level_does_not_cross_symbols(self):
        b_score = np.arange(90, dtype=float)
        mixed = pd.concat([
            pd.DataFrame({
                "symbol": "A",
                "decision_at": pd.date_range("2026-01-02 11:30", periods=40, freq="5min"),
                "score": np.full(40, 0.99),
            }),
            pd.DataFrame({
                "symbol": "B",
                "decision_at": pd.date_range("2026-01-02 11:30", periods=90, freq="5min"),
                "score": b_score,
            }),
        ], ignore_index=True).sample(frac=1.0, random_state=0)
        leveled = score_levels(mixed)
        got = leveled.loc[leveled["symbol"] == "B"].sort_values("decision_at")["level"].to_numpy()
        expect = past_60_level(pd.Series(b_score)).to_numpy()
        self.assertTrue(np.allclose(got, expect))
        last_a = leveled.loc[leveled["symbol"] == "A"].sort_values("decision_at")["level"].iloc[-1]
        self.assertTrue(np.isposinf(last_a))


class ImpulseGapTests(unittest.TestCase):
    def test_gap_excludes_the_current_impulse_and_needs_20_priors(self):
        impulse = pd.Series(np.arange(80, dtype=float))
        gap = prior_impulse_gap(impulse)
        self.assertTrue(gap.iloc[:20].isna().all())
        self.assertAlmostEqual(gap.iloc[20], 20 - impulse.iloc[:20].mean())
        index = 70
        self.assertAlmostEqual(gap.iloc[index], impulse.iloc[index] - impulse.iloc[index - 60:index].mean())
        changed = impulse.copy()
        changed.iloc[index] = 500.0
        gap_changed = prior_impulse_gap(changed)
        self.assertAlmostEqual(impulse.iloc[index] - gap.iloc[index], changed.iloc[index] - gap_changed.iloc[index])

    def test_gap_does_not_cross_symbols(self):
        mixed = pd.concat([
            pd.DataFrame({
                "symbol": "B",
                "decision_at": pd.date_range("2026-01-02 11:30", periods=30, freq="5min"),
                "impulse": np.full(30, 1000.0),
            }),
            pd.DataFrame({
                "symbol": "A",
                "decision_at": pd.date_range("2026-01-02 11:30", periods=30, freq="5min"),
                "impulse": np.arange(30, dtype=float),
            }),
        ], ignore_index=True)
        out = with_impulse_gap60(mixed)
        got = out.loc[out["symbol"] == "B"].sort_values("decision_at")["impulse_gap60"]
        expect = prior_impulse_gap(pd.Series(np.full(30, 1000.0)))
        self.assertTrue(np.allclose(got.to_numpy(dtype=float), expect.to_numpy(dtype=float), equal_nan=True))
        self.assertAlmostEqual(got.iloc[20], 0.0)


if __name__ == "__main__":
    unittest.main()
