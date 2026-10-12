import unittest
import numpy as np
import pandas as pd
from anchored_allocation import mean_anchor,choose,COLUMNS,FRACTIONS,utility


class AnchorTests(unittest.TestCase):
    def test_two_scenario_exact_solution_and_log_utility(self):
        r=np.array([-.20,.05]);prior=np.array([.4,.6]);x=np.log1p(r);target=.01
        q,info=mean_anchor(r,prior,target)
        expected=(target-x[0])/(x[1]-x[0])
        self.assertAlmostEqual(q[1],expected,places=12)
        expected_lambda=(np.log(expected/(1-expected))-np.log(prior[1]/prior[0]))/(x[1]-x[0])
        self.assertAlmostEqual(info['lambda'],expected_lambda,places=10)
        self.assertAlmostEqual(float((utility(r,[1.],1)@q)[0]),target,places=12)
        np.testing.assert_array_equal(r,[-.20,.05])

    def test_unchanged_mean_preserves_prior_and_zero_probability(self):
        r=np.array([-.3,-.1,.05]);p=np.array([0.,.4,.6]);target=p@np.log1p(r)
        q,info=mean_anchor(r,p,target);np.testing.assert_allclose(q,p,atol=1e-12,rtol=1e-12)
        self.assertEqual(q[0],0.);self.assertAlmostEqual(info['kl_divergence'],0.,places=12)

    def test_no_extrapolation_or_silent_clipping(self):
        for target in [.2,-.9,np.log1p(.05),np.log1p(-.1)]:
            q,info=mean_anchor([-.1,.05],[.5,.5],target)
            self.assertIsNone(q);self.assertFalse(info['valid'])
        q,info=mean_anchor([.02,.02],[.3,.7],np.log1p(.02))
        np.testing.assert_array_equal(q,[.3,.7]);self.assertTrue(info['valid'])
        self.assertIsNone(mean_anchor([.02,.02],[.3,.7],.01)[0])

    def test_relative_entropy_is_minimized_under_mean_constraint(self):
        r=np.array([-.3,-.06,.05]);p=np.array([.1,.3,.6]);x=np.log1p(r)
        q,info=mean_anchor(r,p,.005)
        null=np.array([x[1]-x[2],x[2]-x[0],x[0]-x[1]])
        self.assertAlmostEqual(null.sum(),0.);self.assertAlmostEqual(null@x,0.)
        for scale in [-.05,.05]:
            alt=q+scale*null;self.assertTrue((alt>0).all())
            self.assertGreater(np.sum(alt*np.log(alt/p)),info['kl_divergence'])

    def test_mean_anchor_uses_no_future_outcome_and_retains_fraction_limits(self):
        idx=pd.date_range('2025-01-01',periods=60,tz='UTC');pool=pd.DataFrame(index=idx)
        for j,c in enumerate(COLUMNS):pool[c]=np.arange(60)/60+j
        pool['date']=idx.tz_localize(None);pool['net_return']=np.tile([-.06,.05,.05],20)
        current=pool.iloc[-1].copy();current['score']=.01
        a=choose(pool,current,'local',5.,FRACTIONS)
        self.assertTrue(a['ready']);self.assertIn(a['allocation'],FRACTIONS)
        self.assertAlmostEqual(a['distribution']['mean_log_return'],.01,places=12)
        current['net_return']=-.99
        self.assertEqual(a,choose(pool,current,'local',5.,FRACTIONS))
        current['score']=.2
        b=choose(pool,current,'local',5.,FRACTIONS)
        self.assertFalse(b['ready']);self.assertEqual(b['allocation'],0.)
        self.assertEqual(b['reason'],'mean_outside_empirical_support')


if __name__=='__main__':unittest.main()
