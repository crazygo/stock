"""Time boundaries, payoff weighting, independent tree equations and missingness."""
import unittest,copy
from threadpoolctl import threadpool_limits
from nonlinear_direction import *
from audit_nonlinear_models import model_equations
from audit_trees import fit_check,probability
from run_planning import json_ready
import json


def fixture(n=145):
    rng=np.random.default_rng(407);dates=pd.bdate_range('2024-01-02',periods=n)
    at=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC') for d in dates])
    p=pd.DataFrame(rng.normal(size=(n,len(columns(CLOCKS[0])))),columns=columns(CLOCKS[0]),index=at)
    p['date']=dates;p['clock']=CLOCKS[0];p['feature_ready']=True;p['release']=at+pd.Timedelta(minutes=5);p['terminal_release']=p.release
    p['C']=rng.normal(-.01,.005,n);p['D']=np.where(p.iloc[:,0]*p.iloc[:,1]>0,.05,-.06)
    return p


def wire(x):return json.dumps(json_ready(list(x) if isinstance(x,tuple) else x),sort_keys=True,default=str)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.limits=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def test_manual_tree_and_ridge_equations(self):
        p=fixture();at=p.index[-1];models,fits,raws=fit_bases(p,at)
        for kind in VARIANTS:
            info=fits[kind]['fit'];ids=list(pd.to_datetime(info['members'],utc=True));values,n=model_equations(p,ids,at,models[kind],info,columns(CLOCKS[0]),kind)
            self.assertGreater(n,2);np.testing.assert_allclose(raws[kind],[values[s] for s in core.ETFS],atol=1e-12)
    def test_future_and_current_outcomes_cannot_change_past_fit(self):
        p=fixture(150);at=p.index[140];a=fit_bases(p,at)
        p.loc[p.index>=at,['C','D']]=999;p.loc[p.index>at,columns(CLOCKS[0])]=999
        b=fit_bases(p,at);self.assertEqual(wire(a[1:]),wire(b[1:]))
    def test_unresolved_earlier_date_cannot_be_skipped(self):
        p=fixture();p.loc[p.index[50],'terminal_release']=pd.NaT
        models,fits,raws=fit_bases(p,p.index[-1]);self.assertTrue(all(m is None for m in models.values()))
        self.assertTrue(all(len(fits[k]['fit']['members'])==50 for k in VARIANTS))
    def test_119_dates_not_ready(self):
        models,fits,raws=fit_bases(fixture(120),fixture(120).index[-1]);self.assertTrue(all(x is None for x in models.values()))
    def test_uniform_payoff_scaling_preserves_direction(self):
        p=fixture();at=p.index[-1];a=fit_bases(p,at);p[['C','D']]*=2;b=fit_bases(p,at)
        for k in VARIANTS:self.assertAlmostEqual(a[1][k]['raw']['q'],b[1][k]['raw']['q'],places=12)
    def test_single_weighted_class_keeps_constant(self):
        p=fixture();p['D']=.04;models,fits,raws=fit_bases(p,p.index[-1])
        for k in VARIANTS:self.assertNotIn('q',models[k]['heads']);self.assertEqual(fits[k]['raw']['q'],1)
    def test_zero_magnitude_keeps_half(self):
        p=fixture();p['D']=0.;models,fits,raws=fit_bases(p,p.index[-1])
        for k in VARIANTS:self.assertEqual(fits[k]['raw']['q'],.5);self.assertEqual(raws[k][0],raws[k][1])
    def test_same_C_and_M_across_learners(self):
        p=fixture();models,fits,raws=fit_bases(p,p.index[-1])
        for k in ['tree','stump']:
            for h in ['C','M']:self.assertEqual(wire(fits[k]['fit']['heads'][h]),wire(fits['linear']['fit']['heads'][h]))
    def test_tree_leaf_tamper_detected(self):
        p=fixture();at=p.index[-1];models,fits,_=fit_bases(p,at);m=models['tree'];info=fits['tree']['fit'];ids=pd.to_datetime(info['members'],utc=True)
        nodes=m['heads']['q']._predictors[0][0].nodes;j=np.flatnonzero(nodes['is_leaf'])[0];nodes['value'][j]+=.01
        with self.assertRaises(AssertionError):model_equations(p,list(ids),at,m,info,columns(CLOCKS[0]),'tree')
    def test_unready_current_features_do_not_emit_scores(self):
        p=fixture();p.loc[p.index[-1],'feature_ready']=False;models,fits,raws=fit_bases(p,p.index[-1]);self.assertTrue(all(x is None for x in raws.values()))
    def test_positive_direction_is_not_profitable_action(self):
        info={'ready':True,'point':[-.01,-.08],'lower':[-.02,-.09]};e=decide(info,{'expired_unresolved_prefix':False},True);self.assertEqual(e['action'],'CASH')

if __name__=='__main__':unittest.main()
