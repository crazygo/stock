"""Boundary regressions for the actual timestamp/action integration risks."""
import unittest
from decimal import Decimal
from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v2'))
from acquire import normalize
sys.path.insert(0,str(ROOT))
from scripts.model_history_calendar import calendar


class DataContractTests(unittest.TestCase):
    def bars(self, keys):
        return normalize(pd.DataFrame(dict(time_key=keys,open=1.,high=1.1,low=.9,close=1.,volume=10.)), 'SOXL','fixture')

    def test_nine_thirty_is_last_premarket_bar(self):
        f=self.bars(['2023-04-03 09:30:00','2023-04-03 09:35:00'])
        self.assertEqual(list(f.session_type),['pre_market','regular'])
        self.assertEqual(f.start_at.iloc[1],pd.Timestamp('2023-04-03 09:30',tz='America/New_York').tz_convert('UTC'))

    def test_completed_bar_delay_excludes_future(self):
        f=self.bars(['2023-04-03 09:25:00','2023-04-03 09:30:00'])
        at=pd.Timestamp('2023-04-03 09:25:30',tz='America/New_York').tz_convert('UTC')
        self.assertEqual(list(f.loc[f.available_at<=at,'time_key']),['2023-04-03 09:25:00'])

    def test_sunday_night_attributed_to_monday(self):
        f=self.bars(['2024-10-13 20:05:00'])
        self.assertEqual(f.session_date.iloc[0],'2024-10-14')
        self.assertEqual(f.session_type.iloc[0],'overnight')

    def test_official_half_day_has_42_regular_bars(self):
        row=calendar('2023-07-03','2023-07-03')['sessions'][0]
        self.assertEqual(row['duration_minutes'],210)
        self.assertEqual(len(pd.date_range(row['open_at'],row['close_at'],freq='5min',inclusive='right')),42)

    def test_holiday_has_no_synthetic_regular_session(self):
        self.assertEqual(calendar('2023-07-04','2023-07-04')['sessions'],[])

    def test_dst_changes_utc_not_wall_clock(self):
        f=self.bars(['2024-03-08 09:35:00','2024-03-11 09:35:00'])
        self.assertEqual(list(f.end_at.dt.hour),[14,13])

    def test_invalid_high_is_rejected(self):
        raw=pd.DataFrame(dict(time_key=['2023-04-03 09:35:00'],open=[2.],high=[1.],low=[.5],close=[1.],volume=[1.]))
        with self.assertRaises(ValueError):normalize(raw,'SOXL','fixture')


if __name__=='__main__':unittest.main()
