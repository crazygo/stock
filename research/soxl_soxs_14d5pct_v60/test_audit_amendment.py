import unittest
from decimal import Decimal,getcontext
from path_learning import *
from audit_paths_canonical import reference_values
from audit_case_prefix import compare_panel

class AuditFixTests(unittest.TestCase):
    def test_exact_cent_barrier_and_original_price_order(self):
        b={'ohlc':np.empty((2,78,2,4)),'quotes':np.ones((2,8,2)),'previous_close':np.full((2,2),10.23),'split_ratio':np.ones((2,2)),'dividend_ratio':np.zeros((2,2)),'blocks':np.array([[0,1]])}
        b['ohlc'][0]=[25.061666666666664,25.07,24.13583333333333,24.13583333333333];b['ohlc'][1]=[11.15,11.16,10.87,11.10]
        plan={'dates':[0,1],'bar_counts':[78,78],'first_offset':0};sel=np.array([[0]]);v=reference_values(b,sel,CLOCKS[0],np.ones(2),plan);r=simulate(b,sel,CLOCKS[0],np.ones(2),plan)
        self.assertEqual(float(11.16*(24.13583333333333/10.23)),26.33);self.assertLess(float((11.16/10.23)*24.13583333333333),26.33)
        for f in r:np.testing.assert_allclose(r[f],v[f],atol=1e-12)
        self.assertTrue((r['reason']==1).all());np.testing.assert_allclose(r['fill'],26.33)
    def fixture(self):
        at=pd.DatetimeIndex(['2026-04-30 13:25:30Z']);old=pd.DataFrame({'x':[1.],'C':[np.nan],'D':[np.nan],'M':[np.nan],'release':pd.Series([pd.NaT],index=at,dtype='datetime64[ns, UTC]'),'terminal_release':pd.Series([pd.NaT],index=at,dtype='datetime64[ns, UTC]')},index=at);new=old.copy();new[['C','D','M']]=[[.01,.03,.03]];new['release']=pd.Timestamp('2026-05-13 20:00Z');new['terminal_release']=pd.Timestamp('2026-05-04 20:00Z');return old,new,pd.Timestamp('2026-05-01 04:00Z')
    def test_later_maturation_allowed(self):
        a,b,c=self.fixture();self.assertEqual(compare_panel(a,b,c),1)
    def test_causal_input_rewrite_rejected(self):
        a,b,c=self.fixture();b['x']=2.
        with self.assertRaises(AssertionError):compare_panel(a,b,c)
    def test_retroactive_release_rejected(self):
        a,b,c=self.fixture();b['terminal_release']=pd.Timestamp('2026-04-30 20:00Z')
        with self.assertRaises(AssertionError):compare_panel(a,b,c)
    def test_known_result_rewrite_rejected(self):
        a,b,c=self.fixture();a['C']=.02
        with self.assertRaises(AssertionError):compare_panel(a,b,c)

if __name__=='__main__':unittest.main()
