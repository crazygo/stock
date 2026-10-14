import copy,unittest
import numpy as np
import pandas as pd
from projection92 import project,centered
from reference92 import certificate,solve
from bounded92 import *


class ProjectionTests(unittest.TestCase):
    def check(self,raw,observed):
        raw=np.asarray(raw,float);observed=np.asarray(observed,float).reshape(-1,4)
        a,b=raw.copy(),observed.copy();record=project(raw,observed);certificate(raw,observed,record)
        np.testing.assert_equal(raw,a);np.testing.assert_equal(observed,b)
        return record

    def test_no_context_is_exactly_identity(self):
        r=self.check([8,-5,2,-1],[]);np.testing.assert_equal(r['correction'],[0]*4)
        np.testing.assert_allclose(apply_correction([.5,.4,.1,0],r['correction']),[.5,.4,.1,0],atol=1e-15)

    def test_zero_query_is_fixed_and_never_inflated(self):
        for points in [[[2,-2,0,0]],np.random.default_rng(2).normal(size=(20,4))]:
            r=self.check([0]*4,points);np.testing.assert_allclose(r['correction'],0,atol=1e-12)

    def test_line_projection_interior_endpoints_and_orthogonal(self):
        points=[[1,-1,0,0]]
        for raw,expected in [([.5,-.5,1,-1],[.5,-.5,0,0]),([3,-3,0,0],[1,-1,0,0]),([-2,2,0,0],[0]*4)]:
            r=self.check(raw,points);np.testing.assert_allclose(r['correction'],expected,atol=1e-12)

    def test_planar_and_full_dimensional_random_against_independent_qp(self):
        rng=np.random.default_rng(809)
        for rank in [1,2,3]:
            basis=np.linalg.qr(centered(rng.normal(size=(4,4))).T)[0][:,:rank]
            for _ in range(8):
                points=rng.normal(size=(12,rank))@basis.T;raw=rng.normal(size=4)*3
                r=self.check(raw,points);other=solve(raw,points)
                np.testing.assert_allclose(r['correction'],other,atol=2e-6,rtol=2e-5)

    def test_every_training_point_is_fixed(self):
        points=centered(np.random.default_rng(11).normal(size=(20,4)))
        for raw in points:
            r=self.check(raw,points);np.testing.assert_allclose(r['correction'],raw,atol=1e-11)

    def test_translation_permutation_duplicates_and_scale(self):
        rng=np.random.default_rng(102);points=rng.normal(size=(20,4));raw=rng.normal(size=4)*5;a=self.check(raw,points)
        b=self.check(raw+80,points+12);np.testing.assert_allclose(a['correction'],b['correction'],atol=1e-11)
        perm=[2,0,3,1];b=self.check(raw[perm],np.vstack([points[::-1],points])[:,perm]);np.testing.assert_allclose(a['correction'][perm],b['correction'],atol=1e-11)
        b=self.check(raw*8,points*8);np.testing.assert_allclose(a['correction']*8,b['correction'],atol=1e-10)

    def test_nearly_constant_and_extreme_queries(self):
        for points,raw in [(np.ones((10,4)),[1,2,3,4]),(np.ones((10,4))*1e-15,[1,2,3,4]),([[.01,-.01,0,0]],[1e6,-1e6,1e5,-1e5])]:self.check(raw,points)

    def test_zero_probability_support_preserved(self):
        r=self.check([50,-50,4,-4],[[.1,-.1,.3,-.3]])
        p=apply_correction([.8,.2,0,0],r['correction']);self.assertEqual(p[2],0);self.assertEqual(p[3],0);self.assertAlmostEqual(sum(p),1)

    def test_nonfinite_or_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):project([1,2,3],[])
        with self.assertRaises(ValueError):project([1,2,np.nan,4],[])
        with self.assertRaises(ValueError):project([1,2,3,4],[[1,2,np.inf,4]])

    def fixture(self):
        from test_calibration91 import StateTests
        return StateTests().data_fixture()

    def test_contexts_separate_side_and_clock(self):
        panel,parents,history=self.fixture();at=parents[-1]['members'][0]
        d=daily_models(panel,parents,{'expired_unresolved_prefix':False},history,at)
        for name in BMODELS:
            self.assertEqual(len(d['supports'][name]['0:0']['members']),65)
            for key,g in d['supports'][name].items():
                if key!='0:0':self.assertEqual(len(g['members']),0)

    def test_future_append_preserves_support_and_models(self):
        panel,parents,history=self.fixture();at=parents[-1]['members'][0];a=daily_models(panel,parents,{'expired_unresolved_prefix':False},history,at)
        changed=panel.copy();t=at+pd.Timedelta(days=90);changed.loc[t]=changed.iloc[0];changed.loc[t,'fund_realization_log']=1e9
        hh=copy.deepcopy(history);hh[t]={'events':{'recent_ratio':{'ready':False}}};b=daily_models(changed,parents,{'expired_unresolved_prefix':False},hh,at)
        for name in BMODELS:
            for key in a['supports'][name]:np.testing.assert_equal(a['supports'][name][key]['corrections'],b['supports'][name][key]['corrections'])

    def test_serialized_daily_record_preserves_contexts(self):
        from verify_transfer import canonical
        panel,parents,h=self.fixture();at=parents[-1]['members'][0];d=daily_models(panel,parents,{'expired_unresolved_prefix':False},h,at)
        saved=canonical(d['base']);rebuilt=supports(saved,panel,h)
        for name in BMODELS:
            for key in d['supports'][name]:np.testing.assert_equal(d['supports'][name][key]['corrections'],rebuilt[name][key]['corrections'])

    def test_old_events_exact_and_source_cash_never_promoted(self):
        panel,parents,h=self.fixture();at=parents[-1]['members'][0];r=panel.loc[at];d=daily_models(panel,parents,{'expired_unresolved_prefix':False},h,at)
        old=copy.deepcopy(h[at]);old['events']['recent_ratio']['values']=[.01,-.01]
        for name in ['uniform_top','source_blend']:old['events'][name]=old['events']['recent_ratio']
        b={'distributions':{'gap':{'payoff_means':[[1]*4,[1]*4]}}};w={'events':{'paired_state':{'action':'CASH'}}}
        for action in ['SOXL','CASH']:
            old['events']['recent_ratio']['action']=action
            before=previous.issue(at,r,d['base'],old,b,w);after=issue(at,r,d,old,b,w)
            for name in previous.ROUTES:
                for field in before[name]:
                    if isinstance(before[name][field],np.ndarray):np.testing.assert_equal(before[name][field],after[name][field])
                    else:self.assertEqual(before[name][field],after[name][field])
            for name in BMODELS:self.assertEqual(after[name]['action'],action)


if __name__=='__main__':unittest.main()
