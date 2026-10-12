import unittest
from types import SimpleNamespace
from censored_learning import *


def obs(t,e,r=None):
    return {'duration':t,'event':e,'price_log_return':r}


class CensoredTests(unittest.TestCase):
    def test_competing_events_are_not_censors(self):
        c,s,v=aj([obs(.1,1,.04),obs(.2,2,-.06),obs(.3,3,.01)],[1,1,1])
        np.testing.assert_allclose(c,[[1/3]*3]);np.testing.assert_allclose(s,0)
        np.testing.assert_allclose(v,(-.01/3))

    def test_censor_is_not_a_failure(self):
        c,s,v=aj([obs(.1,0),obs(.2,1,.04)],[1,1])
        np.testing.assert_allclose(c,[[1,0,0]]);np.testing.assert_allclose(v,.04)

    def test_tied_events_share_risk_set_before_censor(self):
        c,s,v=aj([obs(.2,1,.04),obs(.2,2,-.06),obs(.2,0),obs(.3,3,.01)],[1,1,1,1])
        np.testing.assert_allclose(c,[[.25,.25,.5]]);np.testing.assert_allclose(s,0)
        np.testing.assert_allclose(v,0,atol=1e-12)

    def test_unidentified_tail_is_retained(self):
        c,s,v=aj([obs(.1,1,.04),obs(.5,0)],[1,1])
        np.testing.assert_allclose(c,[[.5,0,0]]);np.testing.assert_allclose(s,.5)

    def test_resampling_weights_equals_explicit_rows(self):
        o=[obs(.1,1,.04),obs(.2,0),obs(.3,2,-.08),obs(.4,3,.02)]
        a=aj(o,[2,1,0,3]);b=aj([o[i] for i in [0,0,1,3,3,3]],[1]*6)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y)

    def sample_tape(self):
        ends=pd.date_range('2026-01-05 14:35Z',periods=4,freq='5min')
        return {'ends':ends,'available':ends+pd.Timedelta(seconds=30),'valid':np.ones(4,dtype=bool),
            'events':np.array([0,0,1,3]),'logs':np.array([0,0,.04,.01]),'expiry':pd.Timestamp('2026-01-05')}

    def test_future_path_changes_do_not_change_observation(self):
        t=self.sample_tape();cut=pd.Timestamp('2026-01-05 14:41Z');a=observe(t,cut)
        t['events'][2:]=2;t['logs'][2:]=-.99
        self.assertEqual(a,observe(t,cut));self.assertEqual(a['event'],0);self.assertEqual(a['observed_bars'],2)

    def test_late_bar_prevents_using_later_known_event(self):
        t=self.sample_tape();t['available']=pd.DatetimeIndex([t['available'][0],pd.Timestamp('2026-01-07 01:00Z'),*t['available'][2:]])
        o=observe(t,pd.Timestamp('2026-01-06 05:00Z'));self.assertEqual(o['event'],0)
        self.assertEqual(o['observed_bars'],1);self.assertTrue(o['expired_missing'])

    def test_event_before_gap_remains_known(self):
        t=self.sample_tape();t['valid'][3]=False;o=observe(t,pd.Timestamp('2026-01-06 05:00Z'))
        self.assertEqual(o['event'],1);self.assertFalse(o['expired_missing'])

    def test_cutoff_equality_is_not_yet_available(self):
        t=self.sample_tape();o=observe(t,t['available'][2]);self.assertEqual(o['event'],0)
        self.assertEqual(o['observed_bars'],2)

    def test_block_counts_and_parent_size(self):
        at=pd.Timestamp('2026-01-05 14:25:30Z');a=occurrence_counts(63,at)
        self.assertEqual(a.shape,(199,63));np.testing.assert_array_equal(a.sum(axis=1),63)
        np.testing.assert_array_equal(a,occurrence_counts(63,at))

    def test_pending_values_never_used_as_payoff(self):
        a=aj([obs(.1,0,999),obs(.2,1,.04)],[1,1]);b=aj([obs(.1,0,-999),obs(.2,1,.04)],[1,1])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_cash_and_premarket_masks(self):
        info={'ready':True,'point':[.04,-.02],'lower':[.01,-.05]}
        at=pd.Timestamp('2026-01-05 15:05:30Z');out=events(at,pd.Timestamp('2026-01-05'),CLOCKS[1],{'censored':info,'terminal':info},decide(info))
        self.assertEqual(out['premarket_only']['action'],'CASH');self.assertEqual(out['censored_band']['action'],'SOXL')
        self.assertEqual(out['double_cost'],out['censored_band'])

    def test_independent_risk_set_sums_match_batched_product(self):
        from verify_censored import independent_aj
        rng=np.random.default_rng(28);o=[obs(float(i%5)/5,int(i%4),float(rng.normal(0,.04))) for i in range(71)]
        W=rng.uniform(0,2,(11,71));a=aj(o,W);b=independent_aj(o,W)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-12)

    def test_same_day_and_future_coordinates_excluded(self):
        at=pd.Timestamp('2026-01-05 14:25:30Z');prior=at-pd.Timedelta(days=3)
        h={prior:[.01,-.01],at:[999,999],at+pd.Timedelta(days=1):[999,999]}
        e=estimate(h,{},at,None);self.assertEqual(e['members'],[prior])


if __name__=='__main__':unittest.main()
