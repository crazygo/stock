import unittest
from types import SimpleNamespace
from clock_reliability import *
from planning import COLUMNS
from cycle_ledger import replay


class ClockReliabilityTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(27);days=pd.bdate_range('2025-01-01',periods=95)
        times=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in days for c in CLOCKS]
        p=pd.DataFrame(rng.normal(size=(len(times),len(COLUMNS))),index=times,columns=COLUMNS)
        p['date']=[d for d in days for c in CLOCKS];p['clock']=CLOCKS*len(days);p['feature_ready']=True
        p['C']=rng.normal(-.01,.03,len(p));p['D']=rng.normal(0,.04,len(p))
        p['terminal_release']=p.index+pd.Timedelta(days=3);p['release']=p.index+pd.Timedelta(days=14)
        history={t:rng.normal(0,.02,2) for t in p.index}
        return p,history

    def test_other_clocks_cannot_change_current_clock_fit(self):
        p,h=self.fixture();at=p.index[-4];first=issue(p,h,at,h[at]);q=p.copy()
        q.loc[q.clock!=CLOCKS[0],['C','D']]=999
        other={t:(np.array([999.,999.]) if q.loc[t,'clock']!=CLOCKS[0] else v) for t,v in h.items()}
        self.assertEqual(first,issue(q,other,at,h[at]))

    def test_same_day_truth_and_future_forecasts_cannot_change_fit(self):
        p,h=self.fixture();at=p.index[-3];first=issue(p,h,at,h[at]);q=p.copy();q.loc[q.date>=q.loc[at,'date'],['C','D']]=999
        other={t:(np.array([999.,999.]) if t>at else v) for t,v in h.items()}
        self.assertEqual(first,issue(q,other,at,h[at]))

    def test_one_parent_per_date_and_same_clock(self):
        p,h=self.fixture();at=p.index[-2];r=issue(p,h,at,h[at]);ids=r['estimate']['members']
        self.assertEqual(len(ids),p.loc[ids,'date'].nunique())
        self.assertTrue(p.loc[ids,'clock'].eq(p.loc[at,'clock']).all())
        self.assertTrue((pd.DatetimeIndex(ids)<day_cutoff(at)).all())

    def test_route_masks_preserve_chronology_and_point_control(self):
        d={'band':{'ready':True,'action':'SOXS','allocation':1.},'point':{'ready':True,'action':'SOXL','allocation':1.}}
        day=pd.Timestamp('2026-01-02')
        for c in CLOCKS:
            at=pd.Timestamp('2026-01-02 '+c,tz=core.ET);r=routes(at,day,c,d)
            self.assertEqual(r['score_sequential']['action'],'SOXS');self.assertEqual(r['double_cost'],r['score_sequential'])
            self.assertEqual(r['point_sequential']['action'],'SOXL')
            self.assertEqual(r['premarket_only']['action'],'SOXS' if c==CLOCKS[0] else 'CASH')
            self.assertEqual(r['score_after_open']['action'],'CASH' if c==CLOCKS[0] else 'SOXS')

    def test_early_missing_parent_blocks_only_its_clock(self):
        p,h=self.fixture();at=p.index[-4];p.loc[p.index[40],'terminal_release']=pd.NaT
        a=issue(p,h,at,h[at]);b=issue(p,h,p.index[-3],h[p.index[-3]])
        self.assertTrue(a['feedback']['expired_unresolved_prefix']);self.assertEqual(a['band']['action'],'CASH')
        self.assertFalse(b['feedback']['expired_unresolved_prefix'])

    def test_exit_before_next_clock_does_not_allow_same_day_reentry(self):
        days=pd.bdate_range('2026-01-05',periods=3)
        ats=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in days[:2] for c in CLOCKS[:2]]
        f=pd.DataFrame({'date':[days[0],days[0],days[1],days[1]],'clock':CLOCKS[:2]*2},index=ats)
        labs=[]
        for at in ats:
            day=pd.Timestamp(at.tz_convert(core.ET).date());exit_at=(day.tz_localize(core.ET)+pd.Timedelta(hours=10)).tz_convert('UTC')
            if at>exit_at:exit_at+=pd.Timedelta(hours=1)
            labs.append({'decision_at':at,'symbol':'SOXL','entry_at':at+pd.Timedelta(minutes=5),'entry_fill_economic':100.,
                'exit_at':exit_at,'exit_fill_economic':105.,'entry_share_factor':1.,'dividends':[],
                'net_return':.05,'reason':'target','attached_order':{}})
        market=SimpleNamespace(sessions=days,daily={'SOXL':pd.DataFrame({'close':[100.]*3,'complete':[True]*3},index=days)})
        eq,trades,decisions,orders=replay(market,f,labs,{at:'SOXL' for at in ats},str(days[0].date()),str(days[1].date()),require_full_cash=False)
        self.assertEqual(list(decisions.decision),['buy_SOXL','daily_entry_limit','buy_SOXL','daily_entry_limit'])
        self.assertEqual(len(trades),2)


if __name__=='__main__':unittest.main()
