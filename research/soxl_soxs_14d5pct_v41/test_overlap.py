"""Meaningful geometry, causality, model and calibration boundary tests."""
import unittest,copy,json
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
from overlap_learning import *
from audit_overlap import reference_uniqueness,base_check
from run_planning import json_ready


def fixture(n=145):
    rng=np.random.default_rng(417);dates=pd.bdate_range('2024-01-02',periods=n);at=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC') for d in dates])
    p=pd.DataFrame(rng.normal(size=(n,len(columns(CLOCKS[0])))),columns=columns(CLOCKS[0]),index=at)
    p['date']=dates;p['clock']=CLOCKS[0];p['feature_ready']=True;p['release']=at+pd.Timedelta(minutes=5);p['terminal_release']=p.release
    p['C']=rng.normal(-.01,.005,n);p['D']=np.where(p.iloc[:,0]>0,.05,-.06);p['span_start']=np.arange(n)*2;p['span_end']=p.span_start+rng.integers(1,7,n)
    return p


def wire(x):return json.dumps(json_ready(list(x) if isinstance(x,tuple) else x),sort_keys=True,default=str)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.limits=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def test_hand_computed_overlap(self):
        a=[0,2,4];b=[2,3,5];u,r=uniqueness(a,b);np.testing.assert_allclose(u,[5/6,.75,1]);np.testing.assert_allclose(u,reference_uniqueness(a,b,r))
    def test_duplicates_are_fractional_not_independent(self):
        u,_=uniqueness([0,0,0],[4,4,4]);np.testing.assert_allclose(u,[1/3]*3)
    def test_nonoverlap_gaps_and_one_bar(self):
        u,r=uniqueness([0,100,200],[0,102,201]);np.testing.assert_array_equal(u,[1,1,1]);np.testing.assert_allclose(u,reference_uniqueness([0,100,200],[0,102,201],r))
    def test_permutation_equivariance(self):
        u,_=uniqueness([0,2,4],[2,3,5]);v,_=uniqueness([4,0,2],[5,2,3]);np.testing.assert_allclose(v,u[[2,0,1]])
    def test_translation_invariance(self):
        a,_=uniqueness([0,2],[3,5]);b,_=uniqueness([100,102],[103,105]);np.testing.assert_array_equal(a,b)
    def test_empty_and_invalid_intervals(self):
        u,r=uniqueness([],[]);self.assertEqual(len(u),0);reference_uniqueness([],[],r)
        with self.assertRaises(AssertionError):uniqueness([3],[2])
    def test_independent_model_equations(self):
        p=fixture();at=p.index[-1];m,f,raw=fit_bases(p,at)
        for k in VARIANTS:
            ids=list(pd.to_datetime(f[k]['fit']['members'],utc=True));v,n=base_check(p,ids,at,m[k],f[k],columns(CLOCKS[0]),k);self.assertEqual(n,3);np.testing.assert_allclose(raw[k],[v[s] for s in core.ETFS],atol=1e-12)
    def test_current_future_outcomes_and_intervals_cannot_change_fit(self):
        p=fixture(150);at=p.index[140];a=fit_bases(p,at)
        p.loc[p.index>=at,['C','D','span_start','span_end']]=999999;p.loc[p.index>at,columns(CLOCKS[0])]=999
        b=fit_bases(p,at);self.assertEqual(wire(a[1:]),wire(b[1:]))
    def test_unknown_prefix_cannot_be_skipped(self):
        p=fixture();p.loc[p.index[50],'terminal_release']=pd.NaT;a=fit_bases(p,p.index[-1]);p.loc[p.index[50]:,['span_start','span_end']]=10000000;b=fit_bases(p,p.index[-1])
        self.assertEqual(wire(a[1:]),wire(b[1:]));self.assertTrue(all(m is None for m in a[0].values()))
    def test_119_dates_not_ready(self):
        p=fixture(120);self.assertTrue(all(m is None for m in fit_bases(p,p.index[-1])[0].values()))
    def test_baseline_exact_and_factorial_raw_pairs(self):
        p=fixture();at=p.index[-1];m,f,r=fit_bases(p,at);dates,_=training_dates(p,day_cutoff(at));old,info=fit_clock(p[p.clock==CLOCKS[0]],dates['early'],columns(CLOCKS[0]))
        self.assertEqual(wire(f['baseline']['fit']),wire(info));self.assertEqual(wire(f['baseline']['raw']),wire(predict(old,p.loc[[at]])))
        self.assertEqual(wire(f['unique']),wire(f['base_only']));self.assertEqual(wire(f['baseline']),wire(f['cal_only']))
    def test_weights_keep_regularization_scale(self):
        for n in [1,120,250]:
            w=weights(n,126.,np.linspace(.01,1,n));self.assertAlmostEqual(float(w.sum()),n,places=10);self.assertTrue((w>0).all())
    def test_unsupported_current_features_abstain(self):
        p=fixture();p.loc[p.index[-1],'feature_ready']=False;self.assertTrue(all(v is None for v in fit_bases(p,p.index[-1])[2].values()))
    def test_zero_direction_and_single_class(self):
        for d,q in [(0.,.5),(.04,1.)]:
            p=fixture();p['D']=d;m,f,r=fit_bases(p,p.index[-1]);self.assertTrue(all(x['raw']['q']==q for x in f.values()))
    def test_pair_interval_excludes_closed_time_and_earlier_leg_does_not_choose_span(self):
        dates=pd.DatetimeIndex(['2024-01-05','2024-01-08']);calendar={d:{'open_at':pd.Timestamp(str(d.date())+' 14:30',tz='UTC'),'close_at':pd.Timestamp(str(d.date())+' 14:40',tz='UTC')} for d in dates}
        market=SimpleNamespace(sessions=dates,calendar=calendar);at=pd.Timestamp('2024-01-05 14:25:30Z');entry=pd.Timestamp('2024-01-05 14:30Z');ends=[pd.Timestamp('2024-01-05 14:35Z'),pd.Timestamp('2024-01-08 14:40Z')]
        p=pd.DataFrame({'terminal_release':[ends[1]+pd.Timedelta(seconds=1)]},index=[at]);ls={key(at,s):{'entry_at':entry,'exit_at':end} for s,end in zip(core.ETFS,ends)}
        out,r=attach_spans(p,ls,market);self.assertEqual(len(r['grid']),4);self.assertEqual(out.loc[at,'span_start'],0);self.assertEqual(out.loc[at,'span_end'],3)
    def test_calibration_dates_not_queries_and_independent_equations(self):
        from audit_overlap import daily_check
        rows=[];hist={k:{} for k in VARIANTS};calendar={};rng=np.random.default_rng(18);dates=pd.bdate_range('2024-01-02',periods=65)
        for j,d in enumerate(dates):
            calendar[d]={'close_at':pd.Timestamp(str(d.date())+' 16:00',tz=core.ET).tz_convert('UTC')}
            for i,c in enumerate(CLOCKS):
                t=pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC')
                rows.append({'at':t,'date':d,'clock':c,'feature_ready':True,'terminal_release':t+pd.Timedelta(minutes=5),'C':rng.normal(-.01,.02),'D':rng.normal(0,.05),'span_start':j*10+i,'span_end':j*10+i+15})
                raw=rng.normal(0,.05,2)
                for k in VARIANTS:hist[k][t]=raw
        p=pd.DataFrame(rows).set_index('at');market=SimpleNamespace(calendar=calendar);at=p.index[-4];reg=registry(p,market)
        result=fit_calibration(p,hist,reg,at);saved=json.loads(wire(result));models,n=daily_check(p,market,hist,at,saved);self.assertEqual(n,1600)
        self.assertEqual(len(result['parents']),64)
        for k in VARIANTS:self.assertAlmostEqual(sum(result['variants'][k]['row_weights']),64)
        p.loc[p.index>=at,['C','D','span_start','span_end']]=9999
        for k in VARIANTS:
            for t in list(hist[k]):
                if t>=at:hist[k][t]=np.array([9999,9999])
        self.assertEqual(wire(result),wire(fit_calibration(p,hist,registry(p,market),at)))
        before=p.index[59*4];self.assertFalse(fit_calibration(p,hist,registry(p,market),before)['ready'])
    def test_weight_tamper_detected(self):
        p=fixture();at=p.index[-1];m,f,_=fit_bases(p,at);rec=f['unique'];rec['fit']['parent_weights'][0]*=2
        with self.assertRaises(AssertionError):base_check(p,list(pd.to_datetime(rec['fit']['members'],utc=True)),at,m['unique'],rec,columns(CLOCKS[0]),'unique')

if __name__=='__main__':unittest.main()
