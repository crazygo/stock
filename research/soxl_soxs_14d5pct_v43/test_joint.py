"""Train-only basis boundaries, independent equations and interaction capability."""
import unittest,copy,json
from threadpoolctl import threadpool_limits
from joint_learning import *
from audit_joint_models import weighted_equations
from macro_learning import fit_bases as fit_additive_reference
from run_planning import json_ready


def wire(x):return json.dumps(json_ready(list(x) if isinstance(x,tuple) else x),sort_keys=True,default=str)

def fixture(n=150,clock=CLOCKS[0]):
    rng=np.random.default_rng(431);days=pd.bdate_range('2024-01-02',periods=n)
    at=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' '+clock,tz=core.ET).tz_convert('UTC') for d in days])
    cols=columns(clock,'additive');p=pd.DataFrame(rng.normal(size=(n,len(cols))),columns=cols,index=at)
    p['date']=days;p['clock']=clock;p['feature_ready']=True;p['release']=at+pd.Timedelta(minutes=5);p['terminal_release']=p.release
    p['C']=rng.normal(-.01,.003,n);p['D']=np.where(p.macro_spy_5*p.sector_5>0,.05,-.06);return p

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.limits=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def test_registered_pairs_and_dimensions(self):
        self.assertEqual([len(pairs(k)) for k in VARIANTS],[24,8,16,0])
        self.assertEqual([len(columns(CLOCKS[0],k)) for k in VARIANTS],[59,43,51,35])
        self.assertEqual([len(columns(CLOCKS[1],k)) for k in VARIANTS],[71,55,63,47])
        self.assertEqual(set(pairs('joint')),set(pairs('trend'))|set(pairs('risk')))
    def test_manual_product_and_negative_values(self):
        b={'pairs':[('a','b')],'center':{'a':2.,'b':3.},'scale':{'a':2.,'b':4.}}
        p=pd.DataFrame({'a':[4.,0.,2.],'b':[7.,7.,7.]});x=expanded(p,b)
        np.testing.assert_allclose(x[cross_name('a','b')],[1.,-1.,0.]);self.assertEqual(p.columns.tolist(),['a','b'])
    def test_nan_is_not_replaced(self):
        b={'pairs':[('a','b')],'center':{'a':0.,'b':0.},'scale':{'a':1.,'b':1.}}
        self.assertTrue(np.isnan(expanded(pd.DataFrame({'a':[np.nan],'b':[2.]}),b).iloc[0,-1]))
    def test_independent_equations_all_variants_and_clocks(self):
        for clock in [CLOCKS[0],CLOCKS[1]]:
            p=fixture(clock=clock);at=p.index[-1];models,fits,raw=fit_bases(p,at)
            for k in VARIANTS:
                f=fits[k]['fit'];ids=pd.to_datetime(f['members'],utc=True);w=np.array(f['parent_weights']);v,n,_=weighted_equations(p,ids,at,models[k],f,columns(clock,k),w,k)
                self.assertEqual(n,3);np.testing.assert_allclose(raw[k],[v[s] for s in core.ETFS],atol=1e-12)
    def test_current_future_labels_and_future_inputs_cannot_change_fit(self):
        p=fixture();at=p.index[140];a=fit_bases(p,at);p.loc[p.index>=at,['C','D']]=999;p.loc[p.index>at,columns(CLOCKS[0],'additive')]=999
        self.assertEqual(wire(a[1:]),wire(fit_bases(p,at)[1:]))
    def test_current_inputs_cannot_change_basis_or_coefficients(self):
        p=fixture();at=p.index[-1];a=fit_bases(p,at)[1];p.loc[at,columns(CLOCKS[0],'additive')]=500;b=fit_bases(p,at)[1]
        for k in VARIANTS:self.assertEqual(wire(a[k]['fit']),wire(b[k]['fit']))
    def test_appended_future_does_not_change_old_fit(self):
        p=fixture();at=p.index[140];self.assertEqual(wire(fit_bases(p.iloc[:141],at)[1:]),wire(fit_bases(p,at)[1:]))
    def test_unresolved_prefix_is_not_skipped(self):
        p=fixture();p.loc[p.index[50],'terminal_release']=pd.NaT;m,f,r=fit_bases(p,p.index[-1]);self.assertTrue(all(x is None for x in m.values()));self.assertTrue(f['joint']['feedback']['expired_unresolved_prefix'])
    def test_at_least_120_mature_dates(self):
        p=fixture(120);self.assertTrue(all(x is None for x in fit_bases(p,p.index[-1])[0].values()))
        p=fixture(121);self.assertTrue(all(x is not None for x in fit_bases(p,p.index[-1])[0].values()))
    def test_equal_release_cutoff_excluded(self):
        p=fixture();at=p.index[-1];p.loc[p.index[100],'terminal_release']=day_cutoff(at);self.assertTrue(all(x is None for x in fit_bases(p,at)[0].values()))
    def test_current_unready_abstains(self):
        p=fixture();at=p.index[-1];p.loc[at,'feature_ready']=False;self.assertTrue(all(x is None for x in fit_bases(p,at)[2].values()))
    def test_additive_exact_v42(self):
        p=fixture();at=p.index[-1];a=fit_bases(p,at);b=fit_additive_reference(p,at)
        self.assertEqual(wire(a[1]['additive']),wire(b[1]['macro']));np.testing.assert_array_equal(a[2]['additive'],b[2]['macro'])
    def test_single_class_and_zero_magnitude_fallbacks(self):
        for d,q in [(0.,.5),(.05,1.),(-.05,0.)]:
            p=fixture();p['D']=d;_,f,_=fit_bases(p,p.index[-1])
            for k in VARIANTS:self.assertEqual(f[k]['raw']['q'],q)
    def test_constant_input_and_interaction_scale_one(self):
        p=fixture();p['macro_spy_5']=3.;m,f,r=fit_bases(p,p.index[-1]);self.assertEqual(f['joint']['fit']['basis']['scale']['macro_spy_5'],1.)
        self.assertTrue(np.isfinite(r['joint']).all())
    def test_basis_tamper_detected(self):
        p=fixture();at=p.index[-1];m,f,_=fit_bases(p,at);info=f['joint']['fit'];m['joint']['basis']['center']['sector_5']+=.1
        with self.assertRaises(AssertionError):weighted_equations(p,pd.to_datetime(info['members'],utc=True),at,m['joint'],info,columns(CLOCKS[0],'joint'),np.array(info['parent_weights']),'joint')
    def test_saved_predictor_reconstructs_products(self):
        p=fixture();at=p.index[-1];m,_,raw=fit_bases(p,at)
        for k in VARIANTS:
            v=predict_joint(m[k],p.loc[[at]]);np.testing.assert_array_equal(raw[k],[v[s] for s in core.ETFS])
    def test_conditional_direction_can_reverse_without_month_ids(self):
        p=fixture(201);p.loc[:,columns(CLOCKS[0],'additive')]=0.
        for i,at in enumerate(p.index):
            a,b=[(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)][i%4];p.loc[at,['macro_spy_5','sector_5']]=[a,b];p.loc[at,'D']=.05*a*b
        m,_,_=fit_bases(p,p.index[-1]);model=m['joint'];qs=[]
        for state in [-1.,1.]:
            current=p.iloc[[-1]].copy();current['macro_spy_5']=1.;current['sector_5']=state;qs.append(predict_joint(model,current)['q'])
        self.assertLess(qs[0],.2);self.assertGreater(qs[1],.8)
    def test_clock_rows_not_pooled_in_base_fit(self):
        p=fixture(clock=CLOCKS[0]);q=fixture(clock=CLOCKS[1]);at=p.index[-1]
        self.assertEqual(wire(fit_bases(p,at)[1:]),wire(fit_bases(pd.concat([p,q]).sort_index(),at)[1:]))

if __name__=='__main__':unittest.main()
