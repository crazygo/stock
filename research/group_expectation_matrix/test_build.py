import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from .build import calendar_grid, classify_history, outcome_at, corporate_dates, paired_baseline, es95_loss, prepare_bars, validate_inputs, validate_refinement, unavailable_outcome
from .sync_futud_etf import make_universe


class MatrixContractTests(unittest.TestCase):
    def arrays(self, n=200):
        start = pd.Timestamp("2026-01-02T14:30:00Z").value
        starts = start + np.arange(n)*300*10**9
        ends = starts+300*10**9
        arr = np.tile([100., 101., 99., 100., 100000.], (n, 1))
        return arr, np.ones(n, bool), np.ones(n, bool), ends+10**9, starts, ends, np.array(["2026-01-02"]*n)

    def outcome(self, arrays, asof=None, actions=None):
        return outcome_at(*arrays, 0, {"trading_days": 1, "target_return": .05}, asof or int(arrays[5][-1]+10**9), actions or [], .0006)

    def test_early_hit_must_wait_full_window(self):
        a = self.arrays()
        a[0][0, 1] = 106
        self.assertEqual(self.outcome(a, int(a[5][50]))["status"], "pending")
        self.assertEqual(self.outcome(a)["hit"], 1)

    def test_missing_bar_even_after_hit_is_not_success(self):
        a = self.arrays()
        a[0][0, 1] = 106
        a[1][50] = False
        self.assertEqual(self.outcome(a)["status"], "missing")

    def test_exact_horizon_no_outside_high(self):
        a = self.arrays()
        a[0][78, 1] = 106
        self.assertEqual(self.outcome(a)["hit"], 0)
        a[0][77, 1] = 106
        self.assertEqual(self.outcome(a)["hit_minutes"], 390)

    def test_future_append_cannot_change_label(self):
        a = self.arrays()
        before = self.outcome(a)
        a[0][78:, 1] = 900
        a[0][78:, 2] = 1
        self.assertEqual(before, self.outcome(a))

    def test_zero_volume_entry_rejected(self):
        a = self.arrays()
        a[2][0] = False
        self.assertEqual(self.outcome(a)["reason"], "entry_missing_invalid_or_zero_volume")

    def test_late_label_stays_pending(self):
        a = self.arrays()
        a[3][10] = a[5][-1] + 100*10**9
        self.assertEqual(self.outcome(a)["reason"], "horizon_not_yet_available")

    def test_target_and_terminal_proxy_differ(self):
        a = self.arrays()
        a[0][1, 1] = 106
        a[0][77, 3] = 80
        x = self.outcome(a)
        self.assertEqual(x["hit"], 1)
        self.assertAlmostEqual(x["terminal_return"], -.2)
        self.assertAlmostEqual(x["net_proxy"], 1.05*.9994/1.0006-1)
        self.assertEqual(x["holding_minutes_proxy"], 10)

    def test_corporate_action_cash_distinction(self):
        d = pd.DataFrame({"ex_div_date": ["2026-01-02", "2026-01-03"], "per_cash_div": [1, None], "split_ratio": [None, 2]})
        self.assertEqual(corporate_dates(d), ["2026-01-03"])

    def test_calendar_half_day_weekend_and_delay(self):
        g = calendar_grid([
            {"session_date": "2026-11-27", "open_at": "2026-11-27T14:30:00Z", "close_at": "2026-11-27T18:00:00Z"},
            {"session_date": "2026-11-30", "open_at": "2026-11-30T14:30:00Z", "close_at": "2026-11-30T21:00:00Z"}])
        self.assertEqual(len(g), 42+78)
        earliest = pd.Timestamp("2026-11-27T16:30:00Z") + pd.Timedelta(seconds=150)
        self.assertEqual(earliest.ceil("5min"), pd.Timestamp("2026-11-27T16:35:00Z"))
        self.assertEqual(g.iloc[42].date, "2026-11-30")

    def test_trend_requires_full_window_and_is_prefix_invariant(self):
        dates = pd.bdate_range("2026-01-01", periods=130).strftime("%Y-%m-%d").tolist()
        d = pd.DataFrame({"close": np.exp(np.arange(130)*.01), "complete": True,
                          "available": pd.to_datetime(dates, utc=True)+pd.Timedelta(hours=22), "dollar_volume": 100000.}, index=dates)
        strategy = {"lookback_sessions": 126, "family": "trend", "threshold": 1.}
        cutoff = pd.Timestamp(dates[127], tz="UTC")
        x = classify_history(d, dates[:127], cutoff, strategy, [])
        self.assertEqual(x[0], "up")
        d.loc[dates[128:], "close"] = 1e9
        self.assertEqual(x, classify_history(d, dates[:127], cutoff, strategy, []))
        self.assertEqual(classify_history(d, dates[:126], cutoff, strategy, [])[1], "insufficient_history")

    def test_late_history_and_split_rejected(self):
        dates = pd.bdate_range("2026-01-01", periods=17).strftime("%Y-%m-%d").tolist()
        d = pd.DataFrame({"close": 100., "complete": True, "available": pd.to_datetime(dates, utc=True), "dollar_volume": 100000.}, index=dates)
        st = {"lookback_sessions": 15, "family": "trend", "threshold": 1.}
        cutoff = pd.Timestamp(dates[-1], tz="UTC")
        self.assertEqual(classify_history(d, dates[:16], cutoff, st, [])[0], "range")
        self.assertEqual(classify_history(d, dates[:16], cutoff, st, [dates[5]])[1], "corporate_action_in_lookback")
        d.loc[dates[15], "available"] = cutoff+pd.Timedelta(days=1)
        self.assertEqual(classify_history(d, dates[:16], cutoff, st, [])[1], "history_not_available")

    def test_date_matched_comparison_does_not_use_early_period(self):
        group = pd.DataFrame({"status": ["mature"]*2, "date": ["2026-02-01"]*2, "hit": [1, 0], "symbol": ["A", "B"]})
        old = pd.DataFrame({"status": ["mature"]*8, "date": ["2026-01-01"]*8, "hit": [0]*8, "symbol": list("ABCDEFGH")})
        result = paired_baseline(group, pd.concat([old, group]), [], {}, {}, False)
        self.assertEqual(result["baseline_rate"], .5)
        self.assertEqual(result["lift"], 0.)

    def test_fractional_tail_mass(self):
        self.assertAlmostEqual(es95_loss([-.2]+[0]*20), .2/1.05)

    def test_real_preparation_path_distinguishes_late_from_missing(self):
        g = calendar_grid([{"session_date": "2026-01-02", "open_at": "2026-01-02T14:30:00Z", "close_at": "2026-01-02T21:00:00Z"}])
        raw = pd.DataFrame({"start_at": g.start, "end_at": g.end, "available_at": g.end+pd.Timedelta(seconds=1), "open": 100., "high": 101., "low": 99., "close": 100., "turnover": 10000., "volume": 100., "price_basis": "NONE"})
        asof = pd.Timestamp("2026-01-02T21:00:01Z")
        raw.loc[10, "available_at"] = asof+pd.Timedelta(days=1)
        aligned, arr, valid, entry_ok, _, _ = prepare_bars(raw, g, asof)
        x = outcome_at(arr, valid, entry_ok, pd.DatetimeIndex(aligned.available).as_unit("ns").asi8,
                       pd.DatetimeIndex(g.start).as_unit("ns").asi8, pd.DatetimeIndex(g.end).as_unit("ns").asi8,
                       g.date.to_numpy(), 0, {"trading_days": 1, "target_return": .05}, asof.value, [], .0006)
        self.assertEqual(x["status"], "pending")
        self.assertTrue(np.isnan(arr[10, 0]))
        # Restore a known original, append an unknown future revision, and ensure original survives.
        raw.loc[10, "available_at"] = g.end.iloc[10]+pd.Timedelta(seconds=1)
        revision = raw.iloc[[10]].copy()
        revision["available_at"] = asof+pd.Timedelta(days=1)
        revision["close"] = 900.
        base = prepare_bars(raw, g, asof)
        appended = prepare_bars(pd.concat([raw, revision]), g, asof)
        np.testing.assert_array_equal(base[1], appended[1])
        np.testing.assert_array_equal(base[2], appended[2])

    def test_duplicate_expectation_ids_fail_fast(self):
        cfg = {"expectations": [{"expectation_id": "same"}, {"expectation_id": "same"}]}
        with self.assertRaisesRegex(ValueError, "Duplicate expectation_id"):
            validate_inputs(cfg, {}, {}, {})

    def test_paired_baseline_weights_group_counts(self):
        group = pd.DataFrame({"status": ["mature"]*4, "date": ["A"]*3+["B"], "hit": [1, 1, 1, 0], "symbol": list("ABCD")})
        base = pd.DataFrame({"status": ["mature"]*2, "date": ["A", "B"], "hit": [1, 0], "symbol": ["X", "X"]})
        x = paired_baseline(group, base, [], {}, {}, False)
        self.assertEqual(x["baseline_rate"], .75)
        self.assertEqual(x["lift"], 0.)

    def refinement_fixture(self):
        return {"groups": [
            {"group_id": f"industry:{i}", "name": f"原类 - 细分{i}", "segment": f"细分{i}",
             "derived_from_group_id": "industry:old", "symbols": symbols,
             "sources": [{"url": "https://example.com", "observed_at": "2026-09-25", "supports": symbols}]}
            for i, symbols in [(1, ["A", "B"]), (2, ["B"])]],
            "refinement": [{"source_group_id": "industry:old", "source_name": "原类", "source_symbols": ["A", "B"], "subgroup_ids": ["industry:1", "industry:2"]}]}

    def test_flat_refinement_preserves_overlap_and_coverage(self):
        taxonomy = self.refinement_fixture()
        validate_refinement(taxonomy)
        for mutation, message in [
            (lambda t: t["refinement"][0].update(source_symbols=["A", "B", "C"]), "membership union"),
            (lambda t: t["refinement"][0].update(subgroup_ids=["industry:1"]), "2–3"),
            (lambda t: t["groups"][0].update(name="nested name"), "name"),
            (lambda t: t["groups"][0].update(sources=[]), "observed source"),
            (lambda t: t["groups"].append({"group_id": "industry:old"}), "absent"),
            (lambda t: t["groups"][0].update(children=["nested"]), "flat"),
        ]:
            with self.subTest(message=message):
                changed = deepcopy(taxonomy)
                mutation(changed)
                with self.assertRaisesRegex(ValueError, message):
                    validate_refinement(changed)

    def test_parent_entity_requires_union_and_live_relations(self):
        taxonomy = self.refinement_fixture()
        taxonomy["refinement_mode"] = "parent_entities"
        for g in taxonomy["groups"]:
            g.update(group_kind="subgroup", parent_group_id="industry:old", child_group_ids=[])
        taxonomy["groups"].append({"group_id": "industry:old", "name": "原类", "group_kind": "parent", "parent_group_id": None,
                                   "child_group_ids": ["industry:1", "industry:2"], "symbols": ["A", "B"],
                                   "membership_rule": "union_of_child_memberships"})
        validate_refinement(taxonomy)
        for mutation, message in [
            (lambda t: t["groups"][-1].update(symbols=["A"]), "deduplicated child union"),
            (lambda t: t["groups"][0].update(parent_group_id="missing"), "parent entity relation"),
            (lambda t: t["groups"].pop(), "root group entity"),
            (lambda t: t["groups"][-1].update(child_group_ids=[]), "child relations"),
        ]:
            with self.subTest(message=message):
                changed = deepcopy(taxonomy)
                mutation(changed)
                with self.assertRaisesRegex(ValueError, message):
                    validate_refinement(changed)

    def test_etf_security_entity_preserves_benchmark_identity(self):
        source = {"members": [
            {"security_id": "US.A", "symbol": "A", "role": "candidate"},
            {"security_id": "US.QQQ", "symbol": "QQQ", "role": "benchmark"}],
            "universe_id": "original"}
        snapshot = {"group_id": "watchlist:ETF", "observed_at": "2026-09-25T20:00:00Z",
                    "members": [{"security_id": "US.QQQ", "symbol": "QQQ", "name": "QQQ", "listing_date": "1970-01-01"},
                                {"security_id": "US.SOXS", "symbol": "SOXS", "name": "SOXS", "listing_date": "1970-01-01"}]}
        result = make_universe(source, snapshot)
        rows = {m["symbol"]: m for m in result["members"]}
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows["A"]["instrument_type"], "stock")
        self.assertEqual(rows["QQQ"]["instrument_type"], "etf")
        self.assertEqual(rows["QQQ"]["role"], "candidate")
        self.assertIn("benchmark", rows["QQQ"]["roles"])
        self.assertEqual(rows["SOXS"]["watchlist_group_ids"], ["watchlist:ETF"])
        self.assertEqual(result["benchmarks_count"], 1)
        snapshot["members"].append({"security_id": "US.A", "symbol": "A", "name": "A", "listing_date": "1970-01-01"})
        with self.assertRaisesRegex(ValueError, "conflicts with a stock"):
            make_universe(source, snapshot)

    def test_unavailable_etf_label_remains_unknown(self):
        row = unavailable_outcome()
        self.assertEqual((row["status"], row["reason"]), ("missing", "market_data_unavailable"))
        self.assertIsNone(row["hit"])
        self.assertIsNone(row["net_proxy"])


if __name__ == "__main__":
    unittest.main()
