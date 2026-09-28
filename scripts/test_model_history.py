"""Contract checks for historical data, not model performance tests."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.model_history_calendar import calendar
from scripts.backfill_model_history import normalize, quality
from scripts.build_model_history import aggregate, load_bars


def fixture(day, n=78):
    return normalize(pd.DataFrame({
        "time_key":pd.date_range(day+" 09:35",periods=n,freq="5min").strftime("%Y-%m-%d %H:%M:%S"),
        "open":10.,"high":11.,"low":9.,"close":10.5,"volume":1.,"turnover":10.}),"TEST")


class HistoryContractTests(unittest.TestCase):
    def test_official_special_closure_and_session_counts(self):
        c=calendar("2023-01-01","2025-12-31")["sessions"]
        dates=[s["session_date"] for s in c]
        self.assertNotIn("2025-01-09",dates)
        self.assertEqual([sum(d.startswith(str(y)) for d in dates) for y in (2023,2024,2025)],[250,252,250])

    def test_midnight_and_early_close(self):
        f=fixture("2024-07-03",43)
        self.assertEqual(f.session_type.iloc[41],"regular")
        self.assertEqual(f.session_type.iloc[42],"post_market")
        raw=f.iloc[:1].copy();raw["time_key"]="2024-07-04 00:00:00"
        self.assertEqual(normalize(raw,"TEST").session_type.iloc[0],"overnight")

    def test_halfday_hour_and_day(self):
        f=fixture("2024-07-03",42)
        self.assertTrue(quality(f,"2024-07-03","2024-07-03")["rth_complete"])
        self.assertEqual(aggregate(f).duration_minutes.tolist(),[60,60,60,30])
        d=aggregate(f,True).iloc[0]
        self.assertTrue(d.is_complete)
        self.assertEqual(d.volume,42)
        self.assertEqual(d.duration_minutes,210)

    def test_missing_bar_is_not_filled(self):
        f=fixture("2024-07-03",42).drop(index=5)
        self.assertFalse(quality(f,"2024-07-03","2024-07-03")["rth_complete"])
        d=aggregate(f,True).iloc[0]
        self.assertFalse(d.is_complete)
        self.assertTrue(np.isnan(d.open))

    def test_cross_year_loading_and_dst(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"us_5m/TEST";p.mkdir(parents=True)
            fixture("2024-12-31").to_parquet(p/"2024.parquet")
            fixture("2025-01-02").to_parquet(p/"2025.parquet")
            f=load_bars("TEST","2024-12-31","2025-01-02",Path(tmp))
            self.assertEqual(len(f),156)
        self.assertEqual(pd.Timestamp(fixture("2024-03-08").start_at.iloc[0]).hour,14)
        self.assertEqual(pd.Timestamp(fixture("2024-03-11").start_at.iloc[0]).hour,13)

    def test_prelisting_exemption_does_not_hide_postlisting_gaps(self):
        q=quality(pd.DataFrame(),"2024-01-01","2024-12-31",listed="2025-01-02")
        self.assertEqual(q["missing_rth_bars"],0)
        q=quality(pd.DataFrame(),"2025-01-02","2025-01-03",listed="2025-01-03")
        self.assertEqual(q["missing_rth_bars"],78)


if __name__=="__main__":unittest.main()
