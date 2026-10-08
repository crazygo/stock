"""Focused checks for leakage, ambiguity and frozen replay, using real source bars."""
import json, sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build
import evaluate
import adaptive

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars,cls.sources,cls.audit=build.load('AMD')
        payload=json.loads((build.HISTORY/'calendar.json').read_text())
        cls.calendar={s['session_date']:s for s in payload['sessions']}
        cls.target='2026-08-03'
        with patch.object(build,'load',return_value=(cls.bars.copy(),cls.sources,cls.audit)):
            cls.rows,cls.chart,_,_=build.build_symbol('AMD',cls.calendar)

    def test_mutated_future_never_changes_features_or_breadth(self):
        changed=self.bars.copy();mask=changed.start>=pd.Timestamp(self.target+' 09:30')
        changed.loc[mask,['open','high','low','close']]*=3
        changed.loc[mask,'volume']*=8
        with patch.object(build,'load',return_value=(changed,self.sources,self.audit)):
            rows,chart,_,_=build.build_symbol('AMD',self.calendar)
        base=next(r for r in self.rows if r['day']==self.target)
        other=next(r for r in rows if r['day']==self.target)
        fields=set(sum(evaluate.FEATURES.values(),[]))-{'relative_pre_ret','relative_pre_gap','relative_pre_late','relative_night_ret','qqq_pre_gap','qqq_pre_ret','qqq_night_ret','breadth','dispersion','available_pool'}
        for key in fields:
            np.testing.assert_allclose(base[key],other[key],equal_nan=True,err_msg=key)
        b=next(c for c in self.chart['contexts'] if c['day']==self.target)
        o=next(c for c in chart['contexts'] if c['day']==self.target)
        self.assertEqual(b,o)
        self.assertNotEqual(base['mfe'],other['mfe'])

    def test_missing_future_rth_does_not_change_contemporaneous_context(self):
        changed=self.bars[~((self.bars.start>=pd.Timestamp(self.target+' 10:00'))&(self.bars.start<pd.Timestamp(self.target+' 10:05')))].copy()
        with patch.object(build,'load',return_value=(changed,self.sources,self.audit)):
            rows,chart,_,_=build.build_symbol('AMD',self.calendar)
        self.assertNotIn(self.target,[r['day'] for r in rows])
        self.assertEqual(next(c for c in self.chart['contexts'] if c['day']==self.target),next(c for c in chart['contexts'] if c['day']==self.target))

    def test_exact_cutoff_and_available_at(self):
        for r in self.rows:
            self.assertEqual(r['feature_end'],r['day']+'T09:25:00')
            self.assertLessEqual(pd.Timestamp(r['max_source_available']),pd.Timestamp(r['decision_at']))
        changed=self.bars.copy();mask=changed.end==pd.Timestamp(self.target+' 09:25')
        changed.loc[mask,'available']=pd.Timestamp(self.target+' 09:26')
        with patch.object(build,'load',return_value=(changed,self.sources,self.audit)):
            rows,_,_,_=build.build_symbol('AMD',self.calendar)
        self.assertNotIn(self.target,[r['day'] for r in rows])

    def test_unavailable_middle_bar_volume_is_not_a_feature(self):
        late=self.bars.copy();mask=late.end==pd.Timestamp(self.target+' 07:00')
        self.assertTrue(mask.any());late.loc[mask,'available']=pd.Timestamp(self.target+' 09:26')
        variant=late.copy();variant.loc[mask,'volume']*=100
        rows=[]
        for f in [late,variant]:
            with patch.object(build,'load',return_value=(f,self.sources,self.audit)):
                r,_,_,_=build.build_symbol('AMD',self.calendar)
            rows.append(next(r for r in r if r['day']==self.target))
        self.assertEqual(rows[0]['pre_rvol'],rows[1]['pre_rvol'])
        self.assertEqual(rows[0]['pre_volume'],rows[1]['pre_volume'])

    def test_same_bar_double_touch_is_stop(self):
        g=pd.DataFrame([{'open':100.,'high':104.,'low':98.,'close':101.}])
        result,reason=build.first_barrier(g,100.)
        self.assertEqual(reason,'stop_both');self.assertAlmostEqual(result,-.012)

    def test_zero_supply_does_not_create_precision(self):
        g=pd.DataFrame({'day':['2026-01-02'],'symbol':['A'],'y':[1],'mfe':[.02],'mae':[-.01],
                        'close_ret':[.01],'cost_proxy':[.008],'delayed_mfe':[.01]})
        m=evaluate.stats(g,np.array([False]));self.assertIsNone(m['precision']);self.assertEqual(m['n'],0)
        ci=evaluate.block_ci(g,np.zeros(1),np.zeros(1));self.assertEqual(ci['ci'],[None,None])

    def test_historical_rate_excludes_current_and_future_labels(self):
        daily=pd.DataFrame({'o':np.full(100,100.),'h':np.linspace(101,105,100)})
        before=adaptive.historical_rate(daily)
        altered=daily.copy();altered.loc[70:,'h']=100.
        after=adaptive.historical_rate(altered)
        self.assertAlmostEqual(before.iloc[70],after.iloc[70])
        np.testing.assert_allclose(before.iloc[:71],after.iloc[:71],equal_nan=True)
        self.assertNotEqual(before.iloc[71],after.iloc[71])

    def test_weekend_night_attaches_by_elapsed_time(self):
        day=next(r for r in self.rows if r['day']=='2026-08-03')
        self.assertGreaterEqual(day['n_night'],6)
        chosen=self.bars[(self.bars.start>=pd.Timestamp('2026-08-02 20:00'))&(self.bars.end<=pd.Timestamp('2026-08-03 04:00'))]
        self.assertAlmostEqual(day['night_ret'],chosen.close.iloc[-1]/chosen.open.iloc[0]-1)

if __name__=='__main__':unittest.main()
