"""Meaningful leakage, unknown-denominator and calendar boundary checks."""
import unittest
import numpy as np
import pandas as pd
import research
from prepare import daily_features

class Contracts(unittest.TestCase):
    def frame(self):
        rows=[]
        for minute in [245,250,575]:
            for symbol,score in [('AAA',.9),('BBB',.85)]:
                rows.append(dict(symbol=symbol,day='2026-07-01',minute=minute,score=score,baseline=.5,y=0.,entry=100.,target=105.,label_end='2026-07-08 16:00:00',future_status='mature',reference=99.,feature_available='2026-07-01 04:05:01',h_rv_30=.1))
        return pd.DataFrame(rows)
    def test_future_labels_do_not_change_emission_or_dedup(self):
        f=self.frame();a,t=research.replay(f,dict(pre=.8,regular=.8));f['y']=np.nan;f['future_status']='pending_window';b,_=research.replay(f,dict(pre=.8,regular=.8))
        self.assertEqual([(x['day'],x['minute'],x['symbol']) for x in a],[(x['day'],x['minute'],x['symbol']) for x in b])
        self.assertEqual(len(a),2);self.assertEqual(len({e['symbol'] for e in a}),2)
    def test_unknown_and_failure_are_retained(self):
        f=self.frame();e,t=research.replay(f,dict(pre=.8,regular=.8));e[0]['y']=1;e[1]['y']=None;m=research.metrics(e,t)
        self.assertEqual((m['signals'],m['mature'],m['pending'],m['tp'],m['fp']),(2,1,1,1,0));self.assertEqual(m['conservative_precision'],.5)
        e[1]['y']=0;m=research.metrics(e,t);self.assertEqual(m['precision'],.5);self.assertEqual(m['fp'],1)
    def test_no_threshold_means_no_signal(self):
        e,t=research.replay(self.frame(),dict(pre=None,regular=None));m=research.metrics(e,t)
        self.assertEqual(e,[]);self.assertIsNone(m['precision']);self.assertEqual(m['abstention_ratio'],1.)
    def test_next_block_has_all_previous_labels_mature(self):
        research.PANEL=pd.DataFrame({'day':research.DATES})
        td,cd,hd,sd=research.split('2026-08')
        for a,b in [(td,cd),(cd,hd),(hd,sd),(sd,['2026-08-01'])]:
            end=research.DATES[research.DATES.index(a[-1])+4]
            self.assertLess(end,b[0])
    def test_daily_feature_never_reads_current_day(self):
        days=research.DATES[:80];rows=[]
        for j,day in enumerate(days):
            close=570+research.SESSIONS[day]['duration_minutes']
            for minute in range(570,close,5):rows.append(dict(day=day,minute=minute,open=100+j,high=102+j,low=99+j,close=101+j,volume=100))
        raw=pd.DataFrame(rows);before=daily_features(raw).set_index('day');raw.loc[raw.day==days[-1],['open','high','low','close']]*=10;after=daily_features(raw).set_index('day')
        pd.testing.assert_series_equal(before.loc[days[-1]],after.loc[days[-1]])

if __name__=='__main__':unittest.main()
