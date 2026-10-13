"""Tests for temporal membership and an actual immutable null predictor."""
import unittest
from types import SimpleNamespace
from mean_learning import *


class MeanTests(unittest.TestCase):
    def setUp(self):
        self.ids=pd.date_range('2025-01-01 14:25:30',periods=121,freq='D',tz='UTC')
        self.panel=pd.DataFrame({'clock':CLOCKS[0],'C':np.linspace(-.02,.01,121),'D':np.linspace(-.04,.04,121)},index=self.ids)
        self.parents=[{'members':[t]} for t in self.ids[:120]]
        self.panels={'semiconductor':self.panel}

    def test_weighted_mean_and_direction_reconstruction(self):
        values=self.panel.iloc[:120][['C','D']].to_numpy();m=fit_mean(values);w=np.array([2**((i-119)/126) for i in range(120)])
        expected=sum(w[i]*values[i] for i in range(120))/sum(w)
        np.testing.assert_allclose(m['mean_cd'],expected,atol=1e-14);self.assertAlmostEqual(sum(m['weights']),120)
        raw=raw_prediction(m,SimpleNamespace(feature_ready=True));np.testing.assert_allclose(raw,[expected[0]+expected[1],expected[0]-expected[1]])

    def test_current_or_future_values_do_not_enter_fit(self):
        a,ids=fit_clock(self.panels,self.parents,CLOCKS[0]);self.panel.loc[self.ids[-1],['C','D']]=1e9
        b,ids2=fit_clock(self.panels,self.parents,CLOCKS[0]);np.testing.assert_array_equal(a['mean_cd'],b['mean_cd']);self.assertEqual(ids,ids2)

    def test_current_state_values_ignored_but_readiness_retained(self):
        m,_=fit_clock(self.panels,self.parents,CLOCKS[0]);a=SimpleNamespace(feature_ready=True,trend1=-1e6);b=SimpleNamespace(feature_ready=True,trend1=1e6)
        np.testing.assert_array_equal(raw_prediction(m,a),raw_prediction(m,b));self.assertIsNone(raw_prediction(m,SimpleNamespace(feature_ready=False)))

    def test_minimum_actual_dates_and_halfday_clock_absence(self):
        m,ids=fit_clock(self.panels,self.parents[:119],CLOCKS[0]);self.assertIsNone(m);self.assertEqual(ids,[])
        self.panel.loc[self.ids[0],'clock']=CLOCKS[1];m,ids=fit_clock(self.panels,self.parents,CLOCKS[0]);self.assertIsNone(m);self.assertEqual(len(ids),119)

    def test_auxiliary_label_values_not_averaged_into_mean(self):
        a,_=fit_clock(self.panels,self.parents,CLOCKS[0]);other=self.panel.copy();other[['C','D']]=99
        self.panels['nasdaq']=other;b,_=fit_clock(self.panels,self.parents,CLOCKS[0]);np.testing.assert_array_equal(a['mean_cd'],b['mean_cd'])

    def test_raw_route_respects_shared_ready_and_expired_gate(self):
        raw=np.array([.03,-.01]);info={'ready':False};fb={'expired_unresolved_prefix':False}
        self.assertEqual(raw_decision(raw,info,fb)['action'],'CASH')
        info['ready']=True;self.assertEqual(raw_decision(raw,info,fb)['action'],'SOXL')
        fb['expired_unresolved_prefix']=True;self.assertEqual(raw_decision(raw,info,fb)['action'],'CASH')

    def test_invalid_values_rejected(self):
        for x in [[],[1,2],[[1,np.nan]],[[1,2,3]]]:
            with self.assertRaises(ValueError):fit_mean(x)


if __name__=='__main__':unittest.main()
