"""Corporate-action arithmetic and causal daily information boundary tests."""
import unittest,copy,json
from threadpoolctl import threadpool_limits
from macro_learning import *
from macro_features import SYMBOLS,daily_returns,build_features
from audit_macro_models import weighted_equations
from run_planning import json_ready


def wire(x):return json.dumps(json_ready(list(x) if isinstance(x,tuple) else x),sort_keys=True,default=str)


def raw_fixture():
    cal=calendar('2024-09-01','2024-12-31');dates=pd.DatetimeIndex([x['session_date'] for x in cal['sessions']]);raw={}
    for k,s in enumerate(SYMBOLS):
        price=100*np.exp(np.arange(len(dates))*(k+1)*.0001)
        frame=pd.DataFrame({'time_key':dates.strftime('%Y-%m-%d 00:00:00'),'open':price,'close':price,'high':price+1,'low':price-1,'volume':1000})
        acts=pd.DataFrame(columns=['ex_div_date','split_ratio','per_cash_div','per_share_div_ratio','per_share_trans_ratio','allotment_ratio','stk_spo_ratio','spin_off_ratio'])
        raw[s]={'daily':frame,'actions':acts}
    queries=[]
    for d in dates[25:]:
        for c in CLOCKS:queries.append({'at':pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC'),'date':d})
    return raw,cal['sessions'],pd.DataFrame(queries).set_index('at'),dates


def event(day,**kwargs):return {'ex_div_date':str(day.date()),'split_ratio':np.nan,'per_cash_div':np.nan,'per_share_div_ratio':np.nan,'per_share_trans_ratio':np.nan,'allotment_ratio':np.nan,'stk_spo_ratio':np.nan,'spin_off_ratio':np.nan,**kwargs}


def model_fixture(n=145):
    rng=np.random.default_rng(427);dates=pd.bdate_range('2024-01-02',periods=n);at=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC') for d in dates]);cols=columns(CLOCKS[0],'macro')
    p=pd.DataFrame(rng.normal(size=(n,len(cols))),columns=cols,index=at);p['date']=dates;p['clock']=CLOCKS[0];p['feature_ready']=True;p['release']=at+pd.Timedelta(minutes=5);p['terminal_release']=p.release;p['C']=rng.normal(-.01,.005,n);p['D']=np.where(p.iloc[:,0]>0,.05,-.06)
    return p


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.limits=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def test_known_constant_growth_and_relative_differences(self):
        raw,cal,f,dates=raw_fixture();x,_,a=build_features(f,raw,cal)
        for h in [5,20]:
            np.testing.assert_allclose(x['macro_spy_'+str(h)],.0001*h,atol=1e-12)
            np.testing.assert_allclose(x['macro_small_large_'+str(h)],.0001*h,atol=1e-12)
            np.testing.assert_allclose(x['macro_long_short_'+str(h)],-.0001*h,atol=1e-12)
    def test_dividend_does_not_create_false_price_drop(self):
        raw,cal,f,d=raw_fixture();t=30;a=raw['SPY']['daily'];prev=a.loc[t-1,'close'];a.loc[t,['open','high','low','close']]=prev-2;raw['SPY']['actions']=pd.DataFrame([event(d[t],per_cash_div=2.)]);r,_=daily_returns(raw,cal);self.assertAlmostEqual(r.loc[d[t],'SPY'],0,places=12)
    def test_split_units_are_reciprocal_to_futu_ratio(self):
        raw,cal,f,d=raw_fixture();t=30;a=raw['SPY']['daily'];prev=a.loc[t-1,'close'];a.loc[t,['open','high','low','close']]=prev/5;raw['SPY']['actions']=pd.DataFrame([event(d[t],split_ratio=.2)]);r,_=daily_returns(raw,cal);self.assertAlmostEqual(r.loc[d[t],'SPY'],0,places=12)
    def test_current_future_prices_actions_cannot_change_current_features(self):
        raw,cal,f,d=raw_fixture();day=d[40];past=f[f.date<=day];x=build_features(past,raw,cal)[0]
        for s in SYMBOLS:
            a=raw[s]['daily'];a.loc[40:,['open','high','low','close']]*=3
            raw[s]['actions']=pd.DataFrame([event(day,per_cash_div=99.),event(d[50],split_ratio=.5)])
        y=build_features(past,raw,cal)[0];pd.testing.assert_frame_equal(x,y)
    def test_current_future_missing_prices_not_needed(self):
        raw,cal,f,d=raw_fixture();day=d[40];past=f[f.date<=day];x=build_features(past,raw,cal)[0]
        for s in SYMBOLS:raw[s]['daily']=raw[s]['daily'].iloc[:40]
        pd.testing.assert_frame_equal(x,build_features(past,raw,cal)[0])
    def test_missing_past_price_preserves_unknown(self):
        raw,cal,f,d=raw_fixture();raw['SPY']['daily']=raw['SPY']['daily'].drop(35);x,_,_=build_features(f,raw,cal)
        self.assertFalse(x.loc[f.date==d[40],'macro_ready'].any());self.assertTrue(x.loc[f.date==d[35],'macro_ready'].all())
    def test_unsupported_action_is_unknown_not_filled(self):
        raw,cal,f,d=raw_fixture();raw['SPY']['actions']=pd.DataFrame([event(d[35],spin_off_ratio=.5)]);x,_,_=build_features(f,raw,cal);self.assertFalse(x.loc[f.date==d[40],'macro_ready'].any())
    def test_split_plus_cash_ambiguous_is_unknown(self):
        raw,cal,f,d=raw_fixture();raw['SPY']['actions']=pd.DataFrame([event(d[35],split_ratio=.5,per_cash_div=2)]);r,_=daily_returns(raw,cal);self.assertTrue(pd.isna(r.loc[d[35],'SPY']))
    def test_needs_twenty_one_prior_closes(self):
        raw,cal,f,d=raw_fixture();queries=pd.DataFrame({'date':[d[20],d[21]]},index=[pd.Timestamp(str(x.date())+' 09:25:30',tz=core.ET).tz_convert('UTC') for x in [d[20],d[21]]]);x,_,_=build_features(queries,raw,cal);self.assertEqual(x.macro_ready.tolist(),[False,True])
    def test_halfday_and_holiday_dates_respected(self):
        raw,cal,f,d=raw_fixture();x,_,a=build_features(f,raw,cal);r=next(z for z in a['days'] if z['date']==pd.Timestamp('2024-12-02'));self.assertEqual(pd.Timestamp(r['latest_close_at']).tz_convert(core.ET).hour,13);self.assertNotIn(pd.Timestamp('2024-11-28'),r['used_close_dates'])
    def test_all_clock_macro_features_identical(self):
        raw,cal,f,d=raw_fixture();x,_,_=build_features(f,raw,cal)
        for day,g in f.groupby('date'):self.assertEqual(len(x.loc[g.index,MACRO].drop_duplicates()),1)
    def test_all_model_equations_and_dimensions(self):
        p=model_fixture();at=p.index[-1];models,fits,raws=fit_bases(p,at)
        for k in VARIANTS:
            info=fits[k]['fit'];ids=pd.to_datetime(info['members'],utc=True);w=np.array(info['parent_weights']);values,n=weighted_equations(p,ids,at,models[k],info,columns(CLOCKS[0],k),w);self.assertEqual(n,3);np.testing.assert_allclose(raws[k],[values[s] for s in core.ETFS],atol=1e-12)
        self.assertEqual([len(columns(CLOCKS[0],k)) for k in VARIANTS],[35,31,31,27])
    def test_future_current_labels_excluded(self):
        p=model_fixture(150);at=p.index[140];a=fit_bases(p,at);p.loc[p.index>=at,['C','D']]=999;p.loc[p.index>at,columns(CLOCKS[0],'macro')]=999;self.assertEqual(wire(a[1:]),wire(fit_bases(p,at)[1:]))
    def test_unresolved_prefix_and_minimum_not_bypassed(self):
        p=model_fixture();p.loc[p.index[50],'terminal_release']=pd.NaT;models,_,_=fit_bases(p,p.index[-1]);self.assertTrue(all(m is None for m in models.values()))
        p=model_fixture(120);self.assertTrue(all(m is None for m in fit_bases(p,p.index[-1])[0].values()))
    def test_baseline_reproduces_original_fit(self):
        p=model_fixture();at=p.index[-1];_,f,_=fit_bases(p,at);same=p[p.clock==CLOCKS[0]];dates,_=training_dates(same,day_cutoff(at));_,info=fit_clock(same,dates['early'],columns(CLOCKS[0],'baseline'));self.assertEqual(wire(f['baseline']['fit']),wire(info))
    def test_current_unready_abstains(self):
        p=model_fixture();p.loc[p.index[-1],'feature_ready']=False;self.assertTrue(all(r is None for r in fit_bases(p,p.index[-1])[2].values()))

if __name__=='__main__':unittest.main()
