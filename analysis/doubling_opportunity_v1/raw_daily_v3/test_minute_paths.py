import unittest
import pandas as pd
from minute_paths import classify,records
from scripts.model_history_calendar import calendar

class NativeMinutePathTests(unittest.TestCase):
    def test_sunday_night_belongs_to_monday_and_weekend_fabrication_is_rejected(self):
        c=calendar('2026-09-25','2026-09-29');m={s['session_date']:s for s in c['sessions']};days=sorted(m)
        self.assertEqual(classify(pd.Timestamp('2026-09-27 20:00',tz='America/New_York'),m,days),('2026-09-28','夜盘'))
        self.assertIsNone(classify(pd.Timestamp('2026-09-26 20:00',tz='America/New_York'),m,days))
        self.assertIsNone(classify(pd.Timestamp('2026-09-25 20:00',tz='America/New_York'),m,days))
    def test_official_half_day_postmarket_and_regular_boundaries(self):
        c=calendar('2025-11-28','2025-11-28');m={s['session_date']:s for s in c['sessions']};days=sorted(m)
        self.assertEqual(classify(pd.Timestamp('2025-11-28 12:55',tz='America/New_York'),m,days)[1],'常规盘')
        self.assertEqual(classify(pd.Timestamp('2025-11-28 13:00',tz='America/New_York'),m,days)[1],'盘后')
    def test_reverse_split_does_not_rewrite_raw_minute_dollars_or_guess_premarket_timing(self):
        c=calendar('2025-11-26','2025-11-28');start=pd.to_datetime(['2025-11-26T14:30:00Z','2025-11-28T14:00:00Z','2025-11-28T14:30:00Z'],utc=True)
        q=pd.DataFrame({'time_key':(start+pd.Timedelta(minutes=5)).tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M:%S'),
           'start_at':start.astype(str),'open':[10.,10.,250.],'high':[11.,11.,260.],'low':[9.,9.,240.],'close':[10.,10.,250.],'volume':100.})
        e=pd.DataFrame([{'ex_div_date':'2025-11-28','split_ratio':25.}]);r,_=records(q,c,e,'2025-11-28')
        self.assertEqual(r[0][1],10.);self.assertEqual(r[0][6],25.);self.assertIsNone(r[1][6]);self.assertEqual(r[2][1],250.);self.assertEqual(r[2][6],1.)
        # A $20 old-unit barrier becomes $500 after 25-for-1; raw $260 High is not a double.
        self.assertEqual(20*r[0][6]/r[2][6],500.)

if __name__=='__main__':unittest.main()
