import copy
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from calibration91 import *
import reference91 as ref
import tilt_learning


class StateTests(unittest.TestCase):
    def fixture(self,n=40):
        rng=np.random.default_rng(71);P=rng.dirichlet([5,3,1,1],n);y=rng.integers(0,4,n);w=rng.uniform(.1,1,n)
        X=rng.normal(size=(n,19));sides=rng.integers(0,2,n);clocks=rng.integers(0,8,n)
        return P,y,w,X,sides,clocks

    def test_derivatives_against_finite_differences(self):
        P,y,w,X,s,c=self.fixture();Z,_,_=design(X,s,c,w,'fundamental');B=np.random.default_rng(5).normal(0,.05,(28,4))
        f,g,H=objective(B,P,y,w,Z,True);eps=1e-5
        for i in [0,3,17,28,55,73,93,111]:
            D=np.zeros_like(B);D.ravel()[i]=eps
            plus,gp=objective(B+D,P,y,w,Z);minus,gm=objective(B-D,P,y,w,Z)
            self.assertAlmostEqual((plus-minus)/(2*eps),g.ravel()[i],places=7)
            np.testing.assert_allclose((gp-gm).ravel()/(2*eps),H[:,i],atol=1e-7,rtol=1e-7)
        np.testing.assert_allclose(H,H.T,atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(H).min(),10-1e-10)

    def test_independent_optimizer(self):
        P,y,w,X,s,c=self.fixture()
        for mode in ['fundamental','reaction','event','state']:
            model=fit(P,y,w,X,s,c,mode);Z,mu,sd=ref.matrix(X,s,c,w,mode);B=ref.solve(P,y,w,Z)
            np.testing.assert_allclose(model['beta'],B,atol=1e-7,rtol=1e-6)
            np.testing.assert_allclose(model['center'],mu,atol=1e-12);np.testing.assert_allclose(model['scale'],sd,atol=1e-12)

    def test_price_state_matches_original_v83(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('prior_calibration',ROOT/'research/soxl_soxs_14d5pct_v83/state_calibration.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        P,y,w,X,s,c=self.fixture();new=fit(P,y,w,X,s,c,'state');prior=old.fit(P,y,w,X[:,:5],s,c,'state')
        np.testing.assert_allclose(new['beta'],prior['beta'],atol=1e-10,rtol=1e-10)

    def test_calendar_state_matches_original_v87(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('prior_event_calibration',ROOT/'research/soxl_soxs_14d5pct_v87/state_calibration.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        P,y,w,X,s,c=self.fixture();new=fit(P,y,w,X,s,c,'event');prior=old.fit(P,y,w,X[:,:9],s,c,'event')
        np.testing.assert_allclose(new['beta'],prior['beta'],atol=1e-10,rtol=1e-10)

    def test_reaction_matches_original_v88(self):
        P,y,w,X,s,c=self.fixture();new=fit(P,y,w,X,s,c,'reaction');prior=upstream.fit(P,y,w,X[:,:13],s,c,'reaction')
        np.testing.assert_allclose(new['beta'],prior['beta'],atol=1e-10,rtol=1e-10)

    def test_constant_fundamentals_reduce_to_reaction(self):
        P,y,w,X,s,c=self.fixture();X[:,13:]=0
        full=fit(P,y,w,X,s,c,'fundamental');baseline=fit(P,y,w,X,s,c,'reaction')
        np.testing.assert_allclose(full['beta'][:22],baseline['beta'],atol=1e-11)
        np.testing.assert_allclose(full['beta'][22:],0,atol=1e-11)

    def test_no_varying_reaction_reduces_to_calendar_baseline(self):
        P,y,w,X,s,c=self.fixture();X[:,9:]=0
        full=fit(P,y,w,X,s,c,'reaction');baseline=fit(P,y,w,X,s,c,'event')
        np.testing.assert_allclose(full['beta'][:18],baseline['beta'],atol=1e-11)
        np.testing.assert_allclose(full['beta'][18:],0,atol=1e-11)

    def test_zero_parameters_preserve_probabilities_and_zero_support(self):
        P=np.array([[.7,.3,0,0],[1.,0,0,0]])
        q=probabilities(P,np.ones((2,9)),np.zeros((9,4)))
        np.testing.assert_allclose(P,q,atol=1e-14)
        self.assertEqual(q[0,2],0)

    def test_true_zero_support_rejects_entire_fit(self):
        P,y,w,X,s,c=self.fixture();P[0]=[.7,.3,0,0];y[0]=2
        m=fit(P,y,w,X,s,c,'state');self.assertFalse(m['ready']);self.assertEqual(m['unsupported_rows'],1)

    def test_constant_calendar_matches_noevent_control(self):
        P,y,w,X,s,c=self.fixture();X[:,5:]=0
        a=fit(P,y,w,X,s,c,'event');b=fit(P,y,w,X,s,c,'state')
        np.testing.assert_allclose(a['beta'][:14],b['beta'],atol=1e-11)
        np.testing.assert_allclose(a['beta'][14:],0,atol=1e-11)

    def test_scaling_invariance(self):
        P,y,w,X,s,c=self.fixture();a=fit(P,y,w,X,s,c,'fundamental');Y=X*np.arange(1,20)+8
        b=fit(P,y,w,Y,s,c,'fundamental')
        np.testing.assert_allclose(a['beta'],b['beta'],atol=1e-10)
        np.testing.assert_allclose(query(a,P[0],X[0],s[0],c[0]),query(b,P[0],Y[0],s[0],c[0]),atol=1e-11)

    def test_sign_and_interaction(self):
        r=SimpleNamespace(trend5=2,current=-3,relative5=4,log_vol20=-5,event_future_count=.25,event_future_proximity=.5,event_recent_count=3,event_recent_proximity=.8,reaction_count=1,issuer_reaction=.02,peer_reaction=-.01,volume_reaction=.3,fund_matched_count=7,fund_realization_log=.1,fund_above_upper_centered=.5,fund_outlook_count=6,fund_outlook_log_growth=.2,fund_result_mean_age=30)
        np.testing.assert_equal(state(r,0),[2,-3,4,-5,-6,.25,.5,3,.8,1,.02,-.01,.3,7,.1,.5,6,.2,30]);np.testing.assert_equal(state(r,1),[-2,3,-4,-5,-6,.25,.5,3,.8,1,-.02,.01,.3,7,-.1,-.5,6,-.2,30])

    def data_fixture(self,n=65):
        rows=[];parents=[];history={}
        for i,d in enumerate(pd.date_range('2024-01-01',periods=n)):
            at=pd.Timestamp(str(d.date())+' 09:25:30',tz=core.ET).tz_convert('UTC')
            rows.append(dict(at=at,date=d,clock=CLOCKS[0],trend5=i/10,current=i%5-2,relative5=i%3,log_vol20=-3.,calendar_known=True,fundamental_known=True,fund_matched_count=7,fund_realization_log=i/100,fund_above_upper_centered=i%3-1,fund_outlook_count=8,fund_outlook_log_growth=i/200,fund_result_mean_age=i%90,event_future_count=i%7/7,event_future_proximity=i%3/3,event_recent_count=i%8,event_recent_proximity=i%4/4,reaction_count=i%3,issuer_reaction=i/100,peer_reaction=-i/200,volume_reaction=i%4,L_gap=i%4,S_gap=(i+1)%4))
            parents.append(dict(date=d,members=[at],release=at+pd.Timedelta(days=14),feature_ready=True))
            history[at]={'events':{'recent_ratio':dict(ready=True,top_symbol='SOXL',action='SOXL',probabilities=[[.5,.3,.1,.1],[.4,.4,.1,.1]])}}
        return pd.DataFrame(rows).set_index('at'),parents,history

    def test_eligibility_weights_keep_original_age_and_date_mass(self):
        panel,parents,h=self.data_fixture(3);h[parents[1]['members'][0]]['events']['recent_ratio']['action']='CASH'
        P,y,w,X,s,c,meta=samples(panel,parents,h,'eligible')
        self.assertEqual(meta['real_dates'],2);self.assertAlmostEqual(w[0]/w[1],2**(-2/63));self.assertAlmostEqual(sum(w),meta['date_ess'])
        r=ref.data(panel,parents,h,'eligible',CLOCKS)
        for a,b in zip((P,y,w,X,s,c),r[:6]):np.testing.assert_allclose(a,b,atol=1e-12)

    def test_future_append_cannot_change_daily_fit(self):
        panel,parents,h=self.data_fixture();at=pd.Timestamp('2025-01-01 14:25:30Z');fb={'expired_unresolved_prefix':False}
        a=daily_models(panel,parents,fb,h,at)
        future=at+pd.Timedelta(days=1);changed=panel.copy();changed.loc[future]=changed.iloc[0];changed.loc[future,'L_gap']=3;changed.loc[future,'trend5']=1e8
        hh=copy.deepcopy(h);hh[future]={'events':{'recent_ratio':{'ready':False}}}
        b=daily_models(changed,parents,fb,hh,at)
        self.assertTrue(a['ready'])
        for name in MODELS:np.testing.assert_equal(a['models'][name]['beta'],b['models'][name]['beta'])

    def test_unknown_prefix_blocks_later_dates_and_expired_blocks_action(self):
        panel,parents,h=self.data_fixture(67);parents[65]['release']=None
        at=pd.Timestamp('2025-01-01 14:25:30Z');chosen,fb=mature_prefix(parents,at)
        self.assertEqual(len(chosen),65);self.assertTrue(fb['expired_unresolved_prefix'])
        d=daily_models(panel,chosen,fb,h,at);self.assertTrue(d['population_ready']);self.assertFalse(d['ready'])

    def test_issue_never_promotes_source_cash_or_switches_side(self):
        panel,parents,h=self.data_fixture();at=parents[-1]['members'][0];r=panel.loc[at];d=daily_models(panel,parents,{'expired_unresolved_prefix':False},h,at)
        old=copy.deepcopy(h[at]);old['events']['recent_ratio']['values']=[.01,-.01]
        old['events']['uniform_top']=old['events']['recent_ratio'];old['events']['source_blend']=old['events']['recent_ratio']
        base={'distributions':{'gap':{'payoff_means':[[1,1,1,1],[1,1,1,1]]}}};wait={'events':{'paired_state':{'action':'CASH'}}}
        first=issue(at,r,d,old,base,wait)
        for name in MODELS:self.assertEqual(first[name]['action'],'SOXL')
        old['events']['recent_ratio']['action']='CASH';second=issue(at,r,d,old,base,wait)
        for name in MODELS:self.assertEqual(second[name]['action'],'CASH')
        old['events']['recent_ratio']['action']='SOXL';r=r.copy();r.calendar_known=False
        missing=issue(at,r,d,old,base,wait)
        for name in MODELS:
            self.assertEqual(missing[name]['action'],'CASH');self.assertFalse(missing[name]['ready'])
        r.calendar_known=True;r.fundamental_known=False
        missing=issue(at,r,d,old,base,wait)
        for name in MODELS:self.assertFalse(missing[name]['ready'])


if __name__=='__main__':unittest.main()
