import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from clock_training import *
from test_early import EarlyTrainingTests
from allocation_ledger import replay


class ClockTests(unittest.TestCase):
    def volume_fixture(self):
        days=pd.DatetimeIndex(['2025-11-28','2025-12-01'])
        calendar={d:{'close_at':pd.Timestamp(str(d.date())+(' 13:00' if i==0 else ' 16:00'),tz=core.ET).tz_convert('UTC')} for i,d in enumerate(days)}
        rows=[]
        for d in days:
            for t in pd.date_range(d.tz_localize(core.ET)+pd.Timedelta(hours=4,minutes=5),d.tz_localize(core.ET)+pd.Timedelta(hours=13,minutes=10),freq='5min'):
                end=t.tz_convert('UTC');rows.append({'date':d,'end_at':end,'start_at':end-pd.Timedelta(minutes=5),'available_at':end+pd.Timedelta(seconds=1),'econ_volume':1.})
        f=pd.DataFrame(rows).set_index('end_at',drop=False)
        return days,calendar,{s:f.copy() for s in PANEL}

    def test_half_day_omitted_only_after_its_official_close(self):
        days,cal,bars=self.volume_fixture();cut=pd.Timestamp('2025-12-02',tz=core.ET)
        pre,_=clock_volumes(bars,days,cal,cut,CLOCKS[0]);late,_=clock_volumes(bars,days,cal,cut,CLOCKS[3])
        self.assertEqual(list(pre.index),list(days));self.assertEqual(list(late.index),list(days[1:]))
        self.assertEqual(pre.iloc[0,0],65);self.assertEqual(late.iloc[0,0],109)

    def test_late_bar_and_missing_prefix_abstain(self):
        days,cal,bars=self.volume_fixture();cut=pd.Timestamp('2025-12-02',tz=core.ET)
        t=pd.Timestamp('2025-12-01 10:00',tz=core.ET).tz_convert('UTC')
        bars[PANEL[0]].loc[t,'available_at']=t+pd.Timedelta(minutes=7)
        v,_=clock_volumes(bars,days,cal,cut,CLOCKS[1]);self.assertTrue(pd.isna(v.iloc[-1,0]))
        bars[PANEL[1]]=bars[PANEL[1]].drop(t)
        v,_=clock_volumes(bars,days,cal,cut,CLOCKS[1]);self.assertTrue(pd.isna(v.iloc[-1,1]))

    def test_future_same_day_volume_does_not_change_prefix(self):
        days,cal,bars=self.volume_fixture();cut=pd.Timestamp('2025-12-02',tz=core.ET)
        a,_=clock_volumes(bars,days,cal,cut,CLOCKS[1])
        for f in bars.values():
            local=f.end_at.dt.tz_convert(core.ET);f.loc[local.dt.hour>=11,'econ_volume']=1e12
        b,_=clock_volumes(bars,days,cal,cut,CLOCKS[1]);pd.testing.assert_frame_equal(a,b)

    def test_future_truth_cannot_change_clock_model(self):
        p=EarlyTrainingTests().panel();rng=np.random.default_rng(23)
        for col in EXTRA:p[col]=rng.normal(size=len(p))
        at=p.index[160];p.loc[p.index[140:],'terminal_release']=pd.Timestamp('2030-01-01',tz='UTC')
        p.loc[p.index[140:],'release']=pd.Timestamp('2030-01-01',tz='UTC')
        _,a=forecasts(p,at,CLOCKS[1],'context',{'early':[],'full':[]})
        changed=p.copy();changed.loc[p.index[140:],['C','D','M']]=999.
        _,b=forecasts(changed,at,CLOCKS[1],'context',{'early':[],'full':[]})
        self.assertEqual(a,b)

    def test_intraday_columns_do_not_enter_premarket_model(self):
        self.assertEqual(len(columns(CLOCKS[0],'context')),19)
        self.assertEqual(len(columns(CLOCKS[1],'context')),31)
        self.assertEqual(len(columns(CLOCKS[1],'base')),19)

    def test_same_day_exit_cannot_reenter_despite_idle_cash(self):
        days=pd.DatetimeIndex(['2026-01-05','2026-01-06','2026-01-07']);ats=[]
        for d in days[:2]:
            ats.extend(pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for c in CLOCKS[:2])
        f=pd.DataFrame({'date':[days[0]]*2+[days[1]]*2,'clock':CLOCKS[:2]*2},index=ats)
        labs=[]
        for at,d in zip(ats,f.date):
            labs.append({'decision_at':at,'symbol':'SOXL','entry_at':at+pd.Timedelta(minutes=5),
                'exit_at':at+pd.Timedelta(minutes=15),'entry_fill_economic':100.,'exit_fill_economic':105.,
                'entry_share_factor':1.,'dividends':[],'net_return':.049,'reason':'target','attached_order':{}})
        m=SimpleNamespace(sessions=days,daily={'SOXL':pd.DataFrame({'close':[100.]*3,'complete':[True]*3},index=days)})
        book=replay(m,f,labs,{t:'SOXL' for t in ats},str(days[0].date()),str(days[1].date()),allocations={t:.5 for t in ats})
        self.assertEqual(list(book[2].decision),['buy_SOXL','daily_entry_limit','buy_SOXL','daily_entry_limit'])
        self.assertEqual(len(book[1]),2);self.assertGreater(book[0].cash_unsettled.iloc[0],0.)
        self.assertGreater(book[1].capital.iloc[1],book[1].capital.iloc[0])


if __name__=='__main__':unittest.main()
