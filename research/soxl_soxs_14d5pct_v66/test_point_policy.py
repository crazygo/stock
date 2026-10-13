"""Decision boundaries, missing feedback, and matching stress controls."""
import unittest
from path_learning import *


class PointPolicyTests(unittest.TestCase):
    def output(self,point,lower,clock=CLOCKS[0],ready=True,expired=False):
        at=pd.Timestamp('2026-01-05 '+clock,tz=core.ET)
        info={k:{'ready':ready,'point':point,'lower':lower} for k in VARIANTS}
        daily={k:{'feedback':{'expired_unresolved_prefix':expired}} for k in VARIANTS}
        cash={'ready':True,'action':'CASH','allocation':0.}
        return events(at,pd.Timestamp('2026-01-05'),clock,info,daily,cash,cash)

    def test_positive_point_can_act_when_band_abstains(self):
        out=self.output([.003,-.001],[-.002,-.006])
        self.assertEqual(out['conditional_point']['action'],'SOXL')
        self.assertEqual(out['conditional_band']['action'],'CASH')
        for name in ['premarket_only','double_cost']:
            self.assertEqual(out[name],out['conditional_point'])

    def test_nonpositive_or_roundoff_point_abstains(self):
        for point in [[0.,-.01],[-.01,-.02],[1e-12,0.]]:
            self.assertEqual(self.output(point,[-.1,-.2])['conditional_point']['action'],'CASH')

    def test_larger_point_and_stable_tie(self):
        self.assertEqual(self.output([.01,.02],[.001,.011])['conditional_point']['action'],'SOXS')
        self.assertEqual(self.output([.01,.01],[0.,0.])['conditional_point']['action'],'SOXL')

    def test_missing_estimate_and_expired_feedback_disable_action(self):
        for ready,expired in [(False,False),(True,True)]:
            out=self.output([.1,.2],[.01,.02],ready=ready,expired=expired)
            for name in ['conditional_point','conditional_band','double_cost','premarket_only']:
                self.assertEqual(out[name]['action'],'CASH')
                self.assertEqual(out[name]['allocation'],0.)

    def test_premarket_only_does_not_trade_intraday(self):
        out=self.output([.01,.02],[-.01,0.],clock=CLOCKS[2])
        self.assertEqual(out['conditional_point']['action'],'SOXS')
        self.assertEqual(out['double_cost'],out['conditional_point'])
        self.assertEqual(out['premarket_only']['action'],'CASH')

    def test_positive_policy_does_not_change_band_values(self):
        out=self.output([.02,.03],[.01,.02])
        self.assertEqual(out['conditional_band']['values'],[.01,.02])
        self.assertEqual(out['conditional_point']['values'],[.02,.03])

    def test_entry_diagnostic_accepts_point_only_trade(self):
        from audit_entry_direction import check_primary_entry
        out=self.output([.003,-.001],[-.002,-.006])
        chosen=out['conditional_point']
        check_primary_entry(chosen['action'],chosen['values'])
        for symbol,values in [('SOXL',[.01,.02]),('SOXL',[0.,-.01]),('SOXL',[1e-12,0.])]:
            with self.assertRaises(AssertionError):check_primary_entry(symbol,values)

if __name__=='__main__':unittest.main()
