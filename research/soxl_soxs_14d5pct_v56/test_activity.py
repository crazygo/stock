"""Causal activity definition, missingness, matched variants and routing."""
import unittest,copy,json
from types import SimpleNamespace
from activity_learning import *
from audit_dense_calibration import solve,model_check
from run_planning import json_ready


def bars():
    day=pd.Timestamp('2026-01-05');ends=pd.date_range('2026-01-05 04:05',periods=140,freq='5min',tz=core.ET).tz_convert('UTC')
    f=pd.DataFrame({'start_at':ends-pd.Timedelta(minutes=5),'end_at':ends,'available_at':ends+pd.Timedelta(seconds=1),'high':102.,'low':98.,'close':100.,'volume':10.},index=ends)
    return day,f


def prefix_fixture():
    days=pd.bdate_range('2025-01-02',periods=22);records=[]
    for s in SYMBOLS:
        for i,d in enumerate(days):
            at=pd.Timestamp(str(d.date())+' 09:25:30',tz=core.ET).tz_convert('UTC');records.append({'symbol':s,'at':at,'date':d,'clock':CLOCKS[0],'complete':True,'activity':100.+i})
    p=pd.DataFrame(records).set_index(['symbol','at']);at=p.loc['SOXL'].index[-1];base=pd.DataFrame({'date':[days[-1]],'clock':[CLOCKS[0]],'sector_premarket':[2.],'feature_ready':[True]},index=[at]);return p,base


class Tests(unittest.TestCase):
    def test_known_future_calendar_does_not_create_future_prefix_rows(self):
        d,f=bars();days=pd.DatetimeIndex([d,d+pd.Timedelta(days=1)]);calendar={day:{'close_at':pd.Timestamp(str(day.date())+' 16:00',tz=core.ET)} for day in days};f['date']=d
        market=SimpleNamespace(bars={symbol:f for symbol in SYMBOLS},sessions=days,calendar=calendar);cutoff=pd.Timestamp('2026-01-06',tz=core.ET);p=build_prefixes(market,cutoff)
        self.assertTrue((p.index.get_level_values('at')<cutoff).all());self.assertEqual(len(p),32)
    def test_exact_completed_prefix(self):
        d,f=bars();r=prefix_activity(f,d,CLOCKS[0],pd.Timestamp('2026-01-05 16:00',tz=core.ET));self.assertEqual(r['rows'],65);self.assertEqual(r['activity'],65000.)
    def test_later_bars_cannot_change_activity(self):
        d,f=bars();c=pd.Timestamp('2026-01-05 16:00',tz=core.ET);a=prefix_activity(f,d,CLOCKS[0],c);g=f.copy();g.loc[g.end_at>pd.Timestamp('2026-01-05 09:25',tz=core.ET),'volume']=1e12;self.assertEqual(a,prefix_activity(g,d,CLOCKS[0],c))
    def test_missing_bar_stays_unknown(self):
        d,f=bars();r=prefix_activity(f.drop(f.index[3]),d,CLOCKS[0],pd.Timestamp('2026-01-05 16:00',tz=core.ET));self.assertFalse(r['complete']);self.assertTrue(np.isnan(r['activity']))
    def test_delayed_bar_not_discarded_to_hide_gap(self):
        d,f=bars();f.loc[f.index[3],'available_at']=pd.Timestamp('2026-01-05 10:00',tz=core.ET);self.assertFalse(prefix_activity(f,d,CLOCKS[0],pd.Timestamp('2026-01-05 16:00',tz=core.ET))['complete'])
    def test_negative_volume_unknown(self):
        d,f=bars();f.loc[f.index[3],'volume']=-1;self.assertFalse(prefix_activity(f,d,CLOCKS[0],pd.Timestamp('2026-01-05 16:00',tz=core.ET))['complete'])
    def test_zero_individual_bar_retained(self):
        d,f=bars();f.loc[f.index[3],'volume']=0;r=prefix_activity(f,d,CLOCKS[0],pd.Timestamp('2026-01-05 16:00',tz=core.ET));self.assertTrue(r['complete']);self.assertEqual(r['activity'],64000.)
    def test_all_zero_unknown(self):
        d,f=bars();f['volume']=0;self.assertFalse(prefix_activity(f,d,CLOCKS[0],pd.Timestamp('2026-01-05 16:00',tz=core.ET))['complete'])
    def test_halfday_late_clock_absent(self):
        d,f=bars();self.assertIsNone(prefix_activity(f,d,'14:05:30',pd.Timestamp('2026-01-05 13:00',tz=core.ET)))
    def test_split_units_leave_activity_unchanged(self):
        d,f=bars();c=pd.Timestamp('2026-01-05 16:00',tz=core.ET);a=prefix_activity(f,d,CLOCKS[0],c);g=f.copy();g[['high','low','close']]*=7;g['volume']/=7;self.assertAlmostEqual(a['activity'],prefix_activity(g,d,CLOCKS[0],c)['activity'])
    def test_clock_definition_not_daily_total(self):
        d,f=bars();c=pd.Timestamp('2026-01-05 16:00',tz=core.ET);self.assertLess(prefix_activity(f,d,CLOCKS[0],c)['activity'],prefix_activity(f,d,CLOCKS[1],c)['activity'])
    def test_prior_twenty_excludes_current(self):
        p,b=prefix_fixture();f,r=build_features(b,p);self.assertEqual(len(r[0]['members']['SOXL']['prior']),20);self.assertNotIn(b.index[0],r[0]['members']['SOXL']['prior']);self.assertAlmostEqual(r[0]['members']['SOXL']['prior_mean'],110.5)
    def test_missing_prior_is_not_skipped(self):
        p,b=prefix_fixture();p.loc[('SOXL',p.loc['SOXL'].index[-2]),'complete']=False;f,_=build_features(b,p);self.assertFalse(f.activity_ready.iloc[0]);self.assertTrue(f[FEATURES].isna().all().all())
    def test_future_prefix_does_not_change_features(self):
        p,b=prefix_fixture();f,r=build_features(b,p);q=p.copy();day=b.date.iloc[0]+pd.Timedelta(days=1);at=b.index[0]+pd.Timedelta(days=1)
        for s in SYMBOLS:q.loc[(s,at),:]=[day,CLOCKS[0],True,1e12]
        g,rr=build_features(b,q);pd.testing.assert_frame_equal(f,g);self.assertEqual(r,rr)
    def test_clipping_defined_before_interactions(self):
        p,b=prefix_fixture();p.loc[('SOXL',b.index[0]),'activity']=1e20;f,r=build_features(b,p);self.assertEqual(r[0]['members']['SOXL']['log_relative'],3.);self.assertAlmostEqual(f.pair_activity_price.iloc[0],f.pair_activity.iloc[0]*2)
    def test_no_direction_sign_imposed(self):
        a=combine_activity([2,-2,1,-1],-2);self.assertEqual(a[1],2);self.assertEqual(a[3],-4);self.assertEqual(a[-1],-2)
    def test_matched_variant_columns(self):
        for clock in CLOCKS:
            original=prior.columns(clock,'full');self.assertEqual(columns(clock,'full'),original+FEATURES);self.assertEqual(columns(clock,'pair'),original+PAIR);self.assertEqual(columns(clock,'sector'),original+SECTOR)
    def test_normalized_calibration_independent(self):
        r=np.random.default_rng(34);x=np.column_stack([r.normal(size=(320,2)),np.tile(np.arange(8),40)]);y=r.normal(size=(320,2));w=r.uniform(.1,2,320);model_check(fit_map(x,y,w,40),solve(x,y,w,40,8))
    def test_actions_use_own_variant(self):
        estimates={'full':{'ready':True,'point':[.1,.02],'lower':[.02,-.01]},'pair':{'ready':True,'point':[.01,.1],'lower':[-.03,.04]},'sector':{'ready':True,'point':[-.01,-.02],'lower':[-.1,-.2]}};daily={k:{'feedback':{'expired_unresolved_prefix':False}} for k in VARIANTS};at=pd.Timestamp('2026-01-05 14:05:30',tz=core.ET);e=events(at,pd.Timestamp('2026-01-05'),CLOCKS[-2],estimates,daily,{'action':'CASH','ready':True,'allocation':0.})
        self.assertEqual(e['activity_band']['action'],'SOXL');self.assertEqual(e['pair_band']['action'],'SOXS');self.assertEqual(e['sector_band']['action'],'CASH');self.assertEqual(e['four_only']['action'],'CASH');self.assertEqual(e['double_cost'],e['activity_band'])
    def test_new_base_only_past_same_clock(self):
        r=np.random.default_rng(334);days=pd.bdate_range('2025-01-02',periods=124);ix=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' 09:25:30',tz=core.ET).tz_convert('UTC') for d in days]);p=pd.DataFrame(r.normal(size=(len(days),len(columns(CLOCKS[0],'full')))),index=ix,columns=columns(CLOCKS[0],'full'));p['date']=days;p['clock']=CLOCKS[0];p['feature_ready']=True;p['C']=r.normal(0,.01,len(p));p['D']=r.normal(0,.03,len(p));p['M']=abs(p.D);p['terminal_release']=ix+pd.Timedelta(minutes=30);p['release']=p.terminal_release
        _,a,_=fit_bases(p,ix[-1]);q=p.copy();q.loc[ix[-1],['C','D','M']]=999;_,b,_=fit_bases(q,ix[-1]);self.assertEqual(json_ready(a),json_ready(b));self.assertTrue(all(t<ix[-1] for t in a['full']['fit']['members']))
    def test_negative_month_still_fails(self):
        p=read(SOURCE/'protocol.json');s={'completed_trades':30,'monthly':[{'month':m,'return':v,'mark_complete':True} for m,v in zip(p['development']['months'],[.3,.3,-.01,.3])]};self.assertFalse(development_gate(s,True,p)['passed'])

if __name__=='__main__':unittest.main()
