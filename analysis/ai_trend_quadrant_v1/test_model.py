import unittest
import numpy as np
from model import describe,resample,measurements,smooth

class ModelTests(unittest.TestCase):
    def fixture(self):
        r=np.random.default_rng(17).normal(.002,.015,120)
        gap=.3*r;return np.column_stack([r,gap,r-gap,np.abs(r)+.01])
    def test_probabilities_and_reproducibility(self):
        a=self.fixture();x=describe(a,'US.TEST','2026-10-06',120,True);y=describe(a,'US.TEST','2026-10-06',120,True)
        self.assertEqual(x,y);self.assertAlmostEqual(sum(x['p']),1);self.assertTrue(all(0<v<1 for v in x['p']))
    def test_exact_gap_decomposition(self):
        a=self.fixture();np.testing.assert_allclose(a[:,0],a[:,1]+a[:,2],atol=1e-16)
    def test_direction_not_hardcoded_growth(self):
        a=self.fixture();a[:,:3]=-np.abs(a[:,:3]);r=describe(a,'US.TEST','2026-10-06',120)
        self.assertEqual(r['direction'],'下行');self.assertLess(r['price_return'],0)
    def test_constant_returns_finite(self):
        a=np.tile([0.,0.,0.,.01],(60,1));r=describe(a,'US.TEST','2026-10-06',60)
        self.assertTrue(np.isfinite(r['measurements']).all());self.assertEqual(r['direction'],'震荡')
    def test_batch_equals_single(self):
        a=self.fixture();np.testing.assert_allclose(measurements(a)[0],measurements(np.stack([a,a]))[1])
    def test_known_monotonic_path_has_no_drawdown(self):
        a=np.tile([.002,.001,.001,.004],(60,1));r=describe(a,'US.TEST','2026-10-06',60)
        self.assertEqual(r['measurements'][1],0);self.assertEqual(r['direction'],'上行');self.assertAlmostEqual(r['signed_efficiency'],1)
    def test_prefix_causality_and_price_scale_invariance(self):
        import pandas as pd
        from build import model_row
        days=pd.bdate_range('2026-01-01',periods=80).strftime('%Y-%m-%d').tolist()
        c=100*np.exp(np.r_[0,np.cumsum(self.fixture()[:79,0])]);o=c*.999
        f=pd.DataFrame({'day':days,'close':c,'open':o,'high':c*1.02,'low':o*.98,'volume':1000})
        before=model_row(f.iloc[:61],'US.TEST',days[60],60,days,True)
        after=model_row(f,'US.TEST',days[60],60,days,True)
        self.assertEqual(before,after)
        g=f.copy();g[['open','high','low','close']]*=1000
        scaled=model_row(g,'US.TEST',days[60],60,days,True)
        np.testing.assert_allclose(before['measurements'],scaled['measurements'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(before['p'],scaled['p'],rtol=0,atol=0)
    def test_missing_official_session_abstains(self):
        import pandas as pd
        from build import model_row
        days=pd.bdate_range('2026-01-01',periods=62).strftime('%Y-%m-%d').tolist()
        actual=days[:30]+days[31:]
        f=pd.DataFrame({'day':actual,'close':100.,'open':99.,'high':101.,'low':98.,'volume':1000})
        r=model_row(f,'US.TEST',days[-1],60,days)
        self.assertEqual(r['status'],'missing_sessions');self.assertIn(days[30],r['missing_days'])
    def test_smoothing_cannot_carry_across_missing(self):
        rows=[{'status':'scored','as_of':'2026-09-18','raw_p':[.9,.05,.025,.025]}, {'status':'missing_sessions','as_of':'2026-09-25'},
            {'status':'scored','as_of':'2026-10-02','raw_p':[.8,.1,.05,.05]}]
        smooth(rows);self.assertTrue(rows[-1]['smoothing_reset']);self.assertEqual(rows[-1]['p'],rows[-1]['raw_p'])
    def test_large_shift_resets_immediately(self):
        rows=[{'status':'scored','as_of':'2026-09-25','raw_p':[.9,.05,.025,.025]},
            {'status':'scored','as_of':'2026-10-02','raw_p':[.02,.03,.05,.9]}]
        smooth(rows);self.assertTrue(rows[-1]['smoothing_reset']);self.assertEqual(rows[-1]['dominant'],3)

if __name__=='__main__':unittest.main()
