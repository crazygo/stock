"""Causal feature windows, daily label maturity, and independent model equations."""
import unittest
import copy
import json
import tempfile
from threadpoolctl import threadpool_limits
from option_learning import *
from option_features import SYMBOLS, daily_measurements
from audit_option_models import weighted_equations
from run_planning import json_ready


def fixture(n=145):
    rng=np.random.default_rng(407)
    dates=pd.bdate_range('2024-01-02',periods=n)
    at=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC') for d in dates])
    cols=columns(CLOCKS[0])+option_columns(2)+option_columns(5)
    p=pd.DataFrame(rng.normal(size=(n,len(cols))),columns=cols,index=at)
    p['date']=dates;p['clock']=CLOCKS[0];p['feature_ready']=True
    p['terminal_release']=at+pd.Timedelta(minutes=5)
    p['release']=p.terminal_release
    p['C']=rng.normal(-.01,.005,n);p['D']=np.where(p.iloc[:,0]+p[option_columns(2)[0]]>0,.05,-.06)
    return p


def wire(x):
    return json.dumps(json_ready(list(x) if isinstance(x,tuple) else x),sort_keys=True,default=str)


def raw_fixture():
    cal=calendar('2025-10-01','2025-12-31')
    days=[s['session_date'] for s in cal['sessions']]
    n=len(days);raw={}
    for i,s in enumerate(SYMBOLS):
        c=np.arange(n)+100+i;put=np.arange(n)/2+40
        raw[s]={'options':pd.DataFrame({'source_day':days,'call_volume':c,'put_volume':put,'option_volume':c+put}),
                'volatility':pd.DataFrame({'source_day':days,'iv':np.arange(n)/100+1,'hv':np.repeat(.8,n)})}
    dates=pd.to_datetime(days)
    features=pd.DataFrame({'date':dates},index=pd.DatetimeIndex([d.tz_localize(core.ET).tz_convert('UTC') for d in dates]))
    return raw,cal,features


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.limits=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def test_independent_all_model_equations(self):
        p=fixture();at=p.index[-1];models,fits,raws=fit_bases(p,at)
        for k in VARIANTS:
            info=fits[k]['fit'];ids=list(pd.to_datetime(info['members'],utc=True));w=np.asarray(info['parent_weights'])
            values,n=weighted_equations(p,ids,at,models[k],info,columns(CLOCKS[0]),w,k)
            self.assertEqual(n,3);np.testing.assert_allclose(raws[k],[values[s] for s in core.ETFS],atol=1e-12)
    def test_future_current_outcomes_and_future_inputs_do_not_rewrite_past(self):
        p=fixture(150);at=p.index[140];a=fit_bases(p,at)
        p.loc[p.index>=at,['C','D']]=999;p.loc[p.index>at,q_columns(CLOCKS[0],'options')+option_columns(5)]=999
        b=fit_bases(p,at);self.assertEqual(wire(a[1:]),wire(b[1:]))
    def test_unresolved_prefix_not_skipped(self):
        p=fixture();p.loc[p.index[50],'terminal_release']=pd.NaT
        models,fits,_=fit_bases(p,p.index[-1]);self.assertTrue(all(m is None for m in models.values()))
        self.assertTrue(all(len(fits[k]['fit']['members'])==50 for k in VARIANTS))
    def test_minimum120_mature_dates(self):
        p=fixture(120);models,_,_=fit_bases(p,p.index[-1]);self.assertTrue(all(m is None for m in models.values()))
    def test_C_and_M_unchanged(self):
        p=fixture();_,fits,_=fit_bases(p,p.index[-1])
        for k in ['options','delayed']:
            for h in ['C','M']:
                self.assertEqual(wire(fits[k]['fit']['heads'][h]),wire(fits['baseline']['fit']['heads'][h]))
                self.assertEqual(fits[k]['raw'][h],fits['baseline']['raw'][h])
    def test_single_class_and_zero_magnitude(self):
        p=fixture()
        for value,want in [(.04,1.),(0.,.5)]:
            p['D']=value;models,fits,_=fit_bases(p,p.index[-1])
            for k in VARIANTS:
                self.assertNotIn('q',models[k]['heads']);self.assertEqual(fits[k]['raw']['q'],want)
    def test_q_coefficient_tamper_detected(self):
        p=fixture();at=p.index[-1];models,fits,_=fit_bases(p,at);k='options';info=fits[k]['fit']
        models[k]['heads']['q'].coef_[0,0]+=.01
        with self.assertRaises(AssertionError):weighted_equations(p,list(pd.to_datetime(info['members'],utc=True)),at,models[k],info,columns(CLOCKS[0]),np.asarray(info['parent_weights']),k)
    def test_current_unready_abstains(self):
        p=fixture();p.loc[p.index[-1],'feature_ready']=False;_,_,raws=fit_bases(p,p.index[-1]);self.assertTrue(all(x is None for x in raws.values()))
    def test_feature_future_rows_cannot_change_earlier_inputs(self):
        raw,cal,f=raw_fixture();a,_,_=build_features(f,raw,cal['sessions']);day=f.iloc[40].date
        for s in SYMBOLS:
            for fam in raw[s]:
                frame=raw[s][fam];cols=frame.columns.difference(['source_day']);frame.loc[frame.source_day>str(day.date()),cols]*=50
        b,_,_=build_features(f,raw,cal['sessions']);pd.testing.assert_frame_equal(a.loc[f.date<=day],b.loc[f.date<=day])
    def test_lags_use_official_holiday_sessions(self):
        raw,cal,f=raw_fixture();a,_,r=build_features(f,raw,cal['sessions'])
        rec=next(x for x in r['days'] if x['date']=='2025-12-01')
        self.assertEqual(rec['lags']['2']['latest_source_day'],'2025-11-26')
        self.assertEqual(rec['lags']['5']['latest_source_day'],'2025-11-21')
    def test_activity_denominator_excludes_current_volume(self):
        raw,cal,f=raw_fixture();d,by,_=daily_measurements(raw,cal['sessions']);o=raw['NVDA']['options'];j=30
        want=np.log(o.option_volume.iloc[j]/o.option_volume.iloc[j-20:j].mean())
        self.assertAlmostEqual(by['NVDA'].activity.iloc[j],want,places=12)
    def test_missing_member_does_not_become_seven_stock_mean(self):
        raw,cal,f=raw_fixture();raw['NVDA']['options']=raw['NVDA']['options'].drop(index=35)
        a,_,r=build_features(f,raw,cal['sessions']);self.assertFalse(a.iloc[37].option_ready)
        self.assertTrue(np.isnan(a.iloc[37][option_columns(2)[0]]))
    def test_no_profitable_score_abstains(self):
        e=decide({'ready':True,'point':[-.01,-.08],'lower':[-.02,-.09]}, {'expired_unresolved_prefix':False},True)
        self.assertEqual(e['action'],'CASH')
    def test_invalid_counts_remain_unknown(self):
        raw,cal,f=raw_fixture();raw['NVDA']['options'].loc[35,'option_volume']+=1
        a,_,_=build_features(f,raw,cal['sessions']);self.assertFalse(a.iloc[37].option_ready)
    def test_uniform_payoff_scaling_preserves_q(self):
        p=fixture();at=p.index[-1];a=fit_bases(p,at);p[['C','D']]*=2;b=fit_bases(p,at)
        for k in VARIANTS:self.assertAlmostEqual(a[1][k]['raw']['q'],b[1][k]['raw']['q'],places=12)
    def test_failed_primary_cannot_open_case_replay(self):
        from run_options import authorize_cases
        p=read(SOURCE/'protocol.json')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            stats={'primary_average':0.,'completed_trades':0,'all_months_nonnegative':True,
                   'monthly':[{'month':m,'return':0.,'mark_complete':True} for m in p['development']['months']]}
            dump(root/'registration.json',{'phase':'development','protocol_sha256':sha(SOURCE/'protocol.json')})
            dump(root/'results.json',{'primary_name':p['primary'],'primary':stats,'routes':{p['primary']:stats},'coverage':{'complete':True}})
            dump(root/'verification.json',{'passed':True,'results_sha256':sha(root/'results.json')})
            with self.assertRaisesRegex(ValueError,'Development primary failed'):
                authorize_cases(root,p)


if __name__=='__main__':unittest.main()
