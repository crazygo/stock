import copy,unittest
from paired93 import *
from verify_transfer import canonical
import reference93 as ref


class PairedTests(unittest.TestCase):
    def fixture(self,n=65):
        from test_calibration91 import StateTests
        return StateTests().data_fixture(n)

    def test_both_candidates_keep_each_date_mass(self):
        panel,parents,h=self.fixture(3);h[parents[1]['members'][0]]['events']['recent_ratio']['action']='CASH'
        a=paired_samples(panel,parents,h);b=ref.data(panel,parents,h,CLOCKS)
        for x,y in zip(a[:6],b[:6]):np.testing.assert_allclose(x,y,atol=1e-12)
        self.assertEqual(a[-1]['real_dates'],3);self.assertEqual(a[-1]['members'],b[-1]['members'])
        np.testing.assert_equal(a[4],[0,1,0,1,0,1]);self.assertEqual(len(a[1]),6)
        w=a[2];self.assertAlmostEqual(w.sum(),a[-1]['date_ess']);self.assertAlmostEqual(w[0]/w[4],2**(-2/63))
        self.assertAlmostEqual(w[::2].sum(),w[1::2].sum())
        top=calibration.samples(panel,parents,h,'top');np.testing.assert_allclose(w.reshape(-1,2).sum(1),top[2],atol=1e-12)

    def test_training_contains_original_unchosen_side_probability_and_label(self):
        panel,parents,h=self.fixture(1);a=paired_samples(panel,parents,h)
        np.testing.assert_equal(a[0][1],h[parents[0]['members'][0]]['events']['recent_ratio']['probabilities'][1]);self.assertEqual(a[1][1],1)

    def test_contexts_cover_both_sides_without_cross_clock_mixing(self):
        panel,parents,h=self.fixture();at=pd.Timestamp('2025-01-01 14:25:30Z')
        d=daily_models(panel,parents,dict(expired_unresolved_prefix=False),h,at,True);self.assertTrue(d['ready'])
        for mode in PAIRED_MODELS:
            for side in range(2):self.assertEqual(len(d['supports'][mode][f'{side}:0']['members']),65)
            for key,g in d['supports'][mode].items():
                if key.split(':')[1]!='0':self.assertFalse(g['members'])

    def test_future_append_cannot_change_fit_or_support(self):
        panel,parents,h=self.fixture();at=pd.Timestamp('2025-01-01 14:25:30Z');fb=dict(expired_unresolved_prefix=False)
        a=daily_models(panel,parents,fb,h,at,True);later=at+pd.Timedelta(days=1);p=panel.copy();p.loc[later]=p.iloc[0];p.loc[later,'L_gap']=3;p.loc[later,'trend5']=1e8
        hh=copy.deepcopy(h);hh[later]={'events':{'recent_ratio':{'ready':False}}}
        b=daily_models(p,parents,fb,hh,at,True);self.assertEqual(canonical(a),canonical(b))

    def test_unknown_maturity_prefix_and_original_readiness_still_block(self):
        panel,parents,h=self.fixture(67);parents[65]['release']=None;at=pd.Timestamp('2025-01-01 14:25:30Z')
        chosen,fb=mature_prefix(parents,at);self.assertEqual(len(chosen),65)
        self.assertFalse(daily_models(panel,chosen,fb,h,at,True)['ready'])
        self.assertFalse(daily_models(panel,chosen,dict(expired_unresolved_prefix=False),h,at,False)['ready'])

    def test_original_cash_dates_are_not_discarded_or_used_to_relax_gate(self):
        panel,parents,h=self.fixture()
        for v in h.values():v['events']['recent_ratio']['action']='CASH'
        d=daily_models(panel,parents,dict(expired_unresolved_prefix=False),h,pd.Timestamp('2025-01-01 14:25:30Z'),True)
        self.assertFalse(d['ready']);self.assertEqual(d['eligible_dates'],0);self.assertEqual(d['models']['fundamental']['real_dates'],65)

    def example(self,values=(.01,.02),action='CASH'):
        return dict(probabilities=np.array([[.7,.2,.1,0],[.6,.2,.2,0]]),values=values,outside=[[],[]]),dict(top_symbol='SOXL',action=action,values=[.01,.02])

    def test_reselect_can_buy_other_side_and_promote_source_cash(self):
        e,s=self.example();self.assertEqual(decision(e,s,'reselect')['action'],'SOXS')
        self.assertEqual(decision(e,s,'fixed_top')['action'],'SOXL');self.assertEqual(decision(e,s,'fixed_source')['action'],'CASH')
        s['action']='SOXL';self.assertEqual(decision(e,s,'fixed_source')['action'],'SOXL')

    def test_cash_threshold_and_deterministic_equal_value_tie(self):
        for values,action in [((-.01,-.02),'CASH'),((0,0),'CASH'),((1e-12,1e-12),'CASH'),((.01,.01),'SOXL')]:
            e,s=self.example(values);self.assertEqual(decision(e,s,'reselect')['action'],action)

    def test_serialization_and_missing_feature_gate_preserve_old_events(self):
        panel,parents,h=self.fixture();at=parents[-1]['members'][0];r=panel.loc[at]
        d=canonical(daily_models(panel,parents,dict(expired_unresolved_prefix=False),h,pd.Timestamp('2025-01-01 14:25:30Z'),True))
        old=copy.deepcopy(h[at]);old['events']['recent_ratio']['values']=[.01,-.01]
        b={'distributions':{'gap':{'payoff_means':[[1]*4,[2]*4]}}};prior={k:dict(action='CASH',allocation=0) for k in previous.ROUTES}
        out,pairs=issue(at,r,d,old,b,prior);self.assertEqual(out['paired_fundamental']['action'],'SOXS')
        for k in prior:self.assertEqual(out[k],prior[k])
        r=r.copy();r.fundamental_known=False;out,pairs=issue(at,r,d,old,b,prior)
        self.assertFalse(pairs)
        for name in POLICIES:self.assertFalse(out[name]['ready']);self.assertEqual(out[name]['action'],'CASH')

    def test_zero_support_in_other_side_blocks_fit(self):
        panel,parents,h=self.fixture();t=parents[0]['members'][0];h[t]['events']['recent_ratio']['probabilities'][1]=[1,0,0,0]
        d=daily_models(panel,parents,dict(expired_unresolved_prefix=False),h,pd.Timestamp('2025-01-01 14:25:30Z'),True)
        self.assertFalse(d['ready']);self.assertEqual(d['models']['fundamental']['unsupported_rows'],1)

    def test_missing_initial_payoff_distribution_keeps_new_policies_unready(self):
        panel,parents,h=self.fixture(1);at=parents[0]['members'][0];r=panel.loc[at]
        prior={k:dict(action='CASH',allocation=0) for k in previous.ROUTES}
        for ready in [False,True]:
            out,pairs=issue(at,r,dict(ready=ready),h[at],{'distributions':{'gap':None}},prior)
            self.assertEqual(pairs,{})
            for name in POLICIES:self.assertFalse(out[name]['ready']);self.assertEqual(out[name]['action'],'CASH')
            for name in prior:self.assertEqual(out[name],prior[name])


if __name__=='__main__':unittest.main()
