import unittest
from types import SimpleNamespace
from censored_learning import *
from verify_censored_v2 import independent_paths,reference_observation


class RecordRepairTests(unittest.TestCase):
    def test_missing_future_bars_remain_right_censored(self):
        day=pd.Timestamp('2026-01-05');at=pd.Timestamp('2026-01-05 14:25:30Z')
        ends=pd.date_range('2026-01-05 14:35Z',periods=2,freq='5min')
        f=pd.DataFrame({'start_at':ends-pd.Timedelta(minutes=5),'end_at':ends,'available_at':ends+pd.Timedelta(seconds=30),
            'session_type':'regular','open':100.,'high':101.,'low':99.,'close':100.,'share_factor':1.,
            'econ_open':100.,'econ_high':101.,'econ_low':99.,'econ_close':100.},index=ends)
        market=SimpleNamespace(calendar={day:{'open_at':'2026-01-05 14:30Z','close_at':'2026-01-05 14:50Z'}},
            sessions=pd.DatetimeIndex([day]),bars={s:f for s in core.ETFS})
        base={at.isoformat()+'|'+s:{'exit_at':None} for s in core.ETFS}
        paths,count=independent_paths(market,pd.DataFrame(index=[at]),base)
        for path in paths.values():
            o=reference_observation(path,pd.Timestamp('2026-01-05 14:41Z'))
            self.assertEqual(o['event'],0);self.assertEqual(o['duration'],.5);self.assertIsNone(o['price_log_return'])

    def test_source_event_dates_decode_without_changing_serialized_values(self):
        at=pd.Timestamp('2026-01-05 14:25:30Z');day=pd.Timestamp('2026-01-05')
        original={'at':str(at),'date':str(day),'ready':True,'action':'SOXL','allocation':1.}
        fixed={**original,'at':at,'date':day}
        self.assertEqual(str(fixed['at']),original['at']);self.assertEqual(str(fixed['date']),original['date'])
        self.assertEqual(fixed['action'],original['action']);self.assertEqual(fixed['date'].date().isoformat(),'2026-01-05')


if __name__=='__main__':unittest.main()
