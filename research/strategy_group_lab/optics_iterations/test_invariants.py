import unittest
import numpy as np
import pandas as pd
from .train import temporal_split, inner_split
from .features import peer_context
from .prepare import GROUPS
from .portfolio import simulate
from research.strategy_group_lab.portfolio import simulate as legacy_simulate

class Invariants(unittest.TestCase):
    def test_target_maturity_and_inner_purge(self):
        dates = pd.bdate_range("2026-01-02", periods=100)
        rows = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
        labels = pd.DataFrame({"status": "mature", "label_available_at": dates+pd.Timedelta(days=21)})
        fit, cal, cut = temporal_split(rows, labels, "2026-06-01")
        ready = pd.to_datetime(labels.label_available_at, utc=True)
        self.assertTrue((ready[fit] < pd.Timestamp(cut, tz="UTC")).all())
        self.assertFalse((fit & cal).any())
        tr, va, inner = inner_split(rows, labels, fit)
        self.assertTrue((ready.iloc[tr] < pd.Timestamp(inner, tz="UTC")).all())
        self.assertTrue((rows.date.iloc[va] >= inner).all())
        self.assertTrue((ready[cal] < pd.Timestamp("2026-06-01", tz="UTC")).all())

    def test_peer_self_exclusion_and_future_invariance(self):
        groups = [{"group_id": gid} for gid in GROUPS if gid not in ["optics:chain", "optics:core"]]
        rows = pd.DataFrame({"symbol": ["LITE", "COHR", "AAOI", "QQQ"], "date": ["2026-07-01"]*4, "cutoff_at": ["same"]*4})
        own = np.arange(4*42, dtype=np.float32).reshape(4,42)/100
        mem = np.zeros((4,5), bool); mem[:3,0] = True
        a = peer_context(rows, own, mem, groups)
        changed = own.copy(); changed[0] += 100
        b = peer_context(rows, changed, mem, groups)
        np.testing.assert_array_equal(a[0,6:30], b[0,6:30])
        self.assertFalse(np.array_equal(a[1,6:12], b[1,6:12]))
        future = rows.copy(); future["date"] = "2026-07-02"; future["cutoff_at"] = "later"
        c = peer_context(pd.concat([rows,future],ignore_index=True), np.r_[own, own+1000], np.r_[mem,mem], groups)
        np.testing.assert_array_equal(a, c[:4])

    def portfolio_fixture(self, target=.5, days=3):
        starts = pd.DatetimeIndex([t for day in pd.bdate_range("2026-07-01", periods=5) for t in pd.date_range(day+pd.Timedelta(hours=13,minutes=30), periods=78, freq="5min", tz="UTC")])
        grid = pd.DataFrame({"start":starts,"end":starts+pd.Timedelta(minutes=5),"date":starts.strftime("%Y-%m-%d")})
        path = np.ones((len(grid),6)); path[:,:4] *= 100
        path[:,1] = 102; path[:,2] = 99; path[:,3] = 101
        rows = pd.DataFrame({"row_id":[0], "symbol":["X"], "date":["2026-07-01"], "pos":[0]})
        signals = pd.DataFrame({"row_id":[0], "p":[.7], "binding_id":["test"], "expectation_id":["e"]})
        cfg = {"threshold":.5,"source_end":"2026-07-07", "folds":[{"outer_start":"2026-07-01","outer_end":"2026-07-01"}],
               "expectations":[{"id":"e","days":days,"target":target}],
               "portfolio":{"initial_cash":100000,"cost_bps":6,"max_weight":.1,"top_k":3,"max_positions":10}}
        return signals, rows, {"X":path}, grid, cfg

    def test_pending_mark_and_future_price_invariance(self):
        args = self.portfolio_fixture()
        r,d = simulate(*args,"2026-07-01","2026-07-01","2026-07-02")
        self.assertEqual(r["open_positions"],1)
        self.assertEqual(r["closed_trade_n"],0)
        self.assertGreater(r["unrealized_pnl"],0)
        self.assertAlmostEqual(r["end_balance"],100000+r["unrealized_pnl"])
        self.assertAlmostEqual(r["cost"],d["trades"][0]["shares"]*100*.0006)
        args[2]["X"][156:,:4] *= 10
        changed,_ = simulate(*args,"2026-07-01","2026-07-01","2026-07-02")
        self.assertEqual(r,changed)

    def test_mature_parity_with_original_simulation(self):
        for target, hit in [(.01,True),(.5,False)]:
            args = self.portfolio_fixture(target=target)
            labels = pd.DataFrame({"row_id":[0],"expectation_id":["e"],"status":["mature"],"hit":[hit],"hit_minutes":[5 if hit else np.nan],"reason":[None]})
            r,d = simulate(*args,"2026-07-01","2026-07-01","2026-07-07")
            old,od = legacy_simulate(args[0],args[1],labels,args[2],args[3],args[4])
            for key in ["end_balance","net_return","max_drawdown","cost","trade_n"]:
                self.assertAlmostEqual(r[key],old[key])
            self.assertAlmostEqual(d["trades"][0]["pnl"],od["trades"][0]["pnl"])

    def test_missing_selected_path_invalidates_account(self):
        args = self.portfolio_fixture()
        args[2]["X"][40,4] = 0
        r,_ = simulate(*args,"2026-07-01","2026-07-01","2026-07-07")
        self.assertFalse(r["backtest_complete"])
        self.assertIsNone(r["net_return"])
        self.assertEqual(r["unresolved_n"],1)

if __name__ == "__main__":
    unittest.main()
