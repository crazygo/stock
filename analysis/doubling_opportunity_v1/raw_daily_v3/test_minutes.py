import unittest
import pandas as pd
from compare_native_minutes import inspect_minutes
from scripts.model_history_calendar import calendar

class MinuteSourceTests(unittest.TestCase):
    def fixture(self):
        cal=calendar('2025-11-28','2025-11-28');s=cal['sessions'][0]
        start=pd.date_range(s['open_at'],pd.Timestamp(s['close_at'])-pd.Timedelta(minutes=5),freq='5min');end=start+pd.Timedelta(minutes=5)
        q=pd.DataFrame({'symbol':'TEST','price_basis':'NONE','time_key':end.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M:%S'),
          'start_at':start.astype(str),'end_at':end.astype(str),'available_at':(end+pd.Timedelta(seconds=1)).astype(str),
          'open':10.,'high':11.,'low':9.,'close':10.5,'volume':100.})
        return q,cal
    def test_official_half_day_requires_the_entire_42_bar_prefix(self):
        q,cal=self.fixture();r=inspect_minutes(q,'TEST',cal,'2025-11-28')[0]
        self.assertTrue(r['complete_rth_prefix']);self.assertEqual(r['expected_5m'],42);self.assertEqual(r['minute_high'],11.)
        r=inspect_minutes(q.drop(index=10),'TEST',cal,'2025-11-28')[0]
        self.assertFalse(r['complete_rth_prefix']);self.assertEqual(r['actual_5m'],41)
    def test_basis_duplicates_and_premature_availability_are_rejected(self):
        q,cal=self.fixture()
        for invalid in [q.assign(price_basis='QFQ'),pd.concat([q,q.iloc[:1]]),q.assign(available_at=q.start_at)]:
            with self.assertRaises(ValueError):inspect_minutes(invalid,'TEST',cal,'2025-11-28')
    def test_invalid_high_cannot_count_as_complete_rth(self):
        q,cal=self.fixture();q.loc[3,'high']=8.
        self.assertFalse(inspect_minutes(q,'TEST',cal,'2025-11-28')[0]['complete_rth_prefix'])

if __name__=='__main__':unittest.main()
