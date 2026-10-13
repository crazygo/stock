"""Economic accounting, invariance, causal support and independent scaled valuation."""
import copy,json,unittest
from unittest.mock import patch
from path_learning import *
from test_paths import fixture as prices_fixture
from test_activity_state import fixture as forecast_fixture
from audit_paths import reference_values,check_forecast
from audit_volatility import reference_support
from run_planning import json_ready


def fixture(days=4):
    bank,plan,selected=prices_fixture(days);bank['states']=np.zeros((6,8,4));return bank,plan,selected


def check(bank,plan,selected,logvol):
    actual=simulate(bank,selected,CLOCKS[0],[100.,100.],plan,logvol);reference=reference_values(bank,selected,CLOCKS[0],[100.,100.],plan,logvol)
    for k in actual:np.testing.assert_allclose(actual[k],reference[k],rtol=1e-11,atol=1e-11)
    return actual


class VolatilityTests(unittest.TestCase):
    def test_identity_exact_contract_with_cash_split_and_tick(self):
        bank,plan,selected=fixture();bank['split_ratio'][1]=10.;bank['dividend_ratio'][1]=.02;bank['ohlc'][1,4,:,1]=106
        a=check(bank,plan,selected,0.);b=simulate(bank,selected,CLOCKS[0],[100.,100.],plan)
        for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_larger_volatility_changes_target_touch(self):
        bank,plan,selected=fixture();bank['ohlc'][0,4,:,1]=103.
        low=check(bank,plan,selected,0.);high=check(bank,plan,selected,np.log(2.))
        self.assertTrue((low['reason']==3).all());self.assertTrue((high['reason']==1).all());self.assertTrue((high['exit_day']==0).all())

    def test_cash_ex_gap_not_manufactured_wealth(self):
        bank,plan,selected=fixture(2);bank['ohlc'][1]=98.;bank['dividend_ratio'][1]=.02
        for exponent in [.5,2.]:
            result=check(bank,plan,selected,np.log(exponent));np.testing.assert_allclose(result['fill'],98*.9995)
            np.testing.assert_allclose(result['cash'],2.);np.testing.assert_allclose(result['net'],(98*.9995**2+2)/(100*1.0005**2)-1)

    def test_first_day_dividend_does_not_belong_to_buyer(self):
        bank,plan,selected=fixture(1);bank['dividend_ratio'][0]=.4;a=check(bank,plan,selected,np.log(2.));bank['dividend_ratio'][0]=0.;b=check(bank,plan,selected,np.log(2.))
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertTrue((a['cash']==0).all())

    def test_split_rounding_and_adverse_gap(self):
        bank,plan,selected=fixture(2);bank['split_ratio'][1]=10.;bank['ohlc'][1,4,:,1]=103.
        result=check(bank,plan,selected,np.log(2.));np.testing.assert_allclose(result['fill'],105.1)
        bank['ohlc'][1,0]=[[80,85,75,81]]*2;result=check(bank,plan,selected,np.log(2.));np.testing.assert_allclose(result['fill'],64*.9995)

    def test_same_bar_both_barriers_stop_first(self):
        bank,plan,selected=fixture();bank['ohlc'][0,4,:,1]=103.;bank['ohlc'][0,4,:,2]=96.
        result=check(bank,plan,selected,np.log(2.));self.assertTrue((result['reason']==2).all());self.assertTrue((result['exit_bar']==4).all())

    def test_prefix_and_halfday_ignore_unavailable_path(self):
        bank,plan,selected=fixture(1);plan['first_offset']=12;plan['bar_counts']=[42];bank['ohlc'][0,5,:,1]=110.;bank['ohlc'][0,45,:,1]=110.
        result=check(bank,plan,selected,np.log(2.));self.assertTrue((result['reason']==3).all())

    def test_full_scaled_forecast_independent_partition_and_tail(self):
        bank,market,panel,at=forecast_fixture();bank['states'][:,:,2]=0.;panel.loc[at,'log_sector_volatility']=np.log(2.)
        bank['ohlc'][:30,4,0,1]=103.;bank['ohlc'][:30,4,1,2]=96.;bank['ohlc'][90:,10,1,1]=103.
        model,info,raw=forecast(bank,market,panel,at,'scaled');self.assertTrue(info['ready']);self.assertTrue(model['resolved'].any());self.assertTrue((~model['resolved']).any())
        canonical=json.loads(json.dumps(json_ready(info),default=str));value,n=check_forecast(bank,market,panel,at,'scaled',canonical,model);np.testing.assert_allclose(raw,value,rtol=1e-11,atol=1e-11);self.assertGreater(n,4096)

    def test_forecast_identity_exact_original_fresh(self):
        bank,market,panel,at=forecast_fixture();a,ia,ra=forecast(bank,market,panel,at,'scaled');b,ib,rb=forecast(bank,market,panel,at,'fresh')
        np.testing.assert_array_equal(ra,rb);np.testing.assert_array_equal(a['selected_blocks'],b['selected_blocks']);self.assertEqual(ia['members'],ib['members']);self.assertEqual(ia['raw'],ib['raw'])

    def test_invalid_support_abstains_entire_query_not_bad_path_deletion(self):
        bank,market,panel,at=forecast_fixture();bank['states'][:,:,2]=0.;panel.loc[at,'log_sector_volatility']=np.log(2.);bank['ohlc'][7]=1.;bank['dividend_ratio'][7]=.9
        _,base,_=forecast(bank,market,panel,at,'fresh');model,info,raw=forecast(bank,market,panel,at,'scaled')
        self.assertTrue(base['ready']);self.assertFalse(info['ready']);self.assertEqual(info['members'],base['members']);self.assertIsNone(model);self.assertIsNone(raw)
        value,n=check_forecast(bank,market,panel,at,'scaled',info,model);self.assertIsNone(value);self.assertEqual(n,0)

    def test_missing_second_day_volatility_disables_whole_scaled_query(self):
        bank,market,panel,at=forecast_fixture();bank['states'][-1,:,2]=np.nan
        _,fresh,_=forecast(bank,market,panel,at,'fresh');model,info,raw=forecast(bank,market,panel,at,'scaled')
        self.assertTrue(fresh['ready']);self.assertFalse(info['ready']);self.assertEqual(info['members'],fresh['members']);self.assertEqual(info['volatility']['reason'],'missing_or_nonfinite_prior_volatility')
        check_forecast(bank,market,panel,at,'scaled',info,model)

    def test_future_prices_labels_and_successors_cannot_change_forecast(self):
        bank,market,panel,at=forecast_fixture();bank['release_ns'][-1]=at.value;a,_,x=forecast(bank,market,panel,at,'scaled')
        bank['ohlc'][-1]*=999.;bank['states'][-1,:,2]=np.nan;del bank['successor_states'];del bank['successor_at_ns'];panel['C']=999.;panel['terminal_release']=at+pd.Timedelta(days=99)
        with patch('path_learning.transition_matrix',side_effect=AssertionError('unused future state')):b,_,y=forecast(bank,market,panel,at,'scaled')
        np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(a['selected_blocks'],b['selected_blocks'])

    def test_nonfinite_multiplier_rejected_without_clipping(self):
        bank,plan,_=fixture();members=np.arange(5)
        for logvol in [1000.,-1000.]:
            a=volatility_support(bank,members,CLOCKS[0],logvol,plan);b=reference_support(bank,members,CLOCKS[0],logvol,plan)
            self.assertEqual(a,b);self.assertFalse(a['ready']);self.assertEqual(a['reason'],'invalid_volatility_multiplier')

if __name__=='__main__':unittest.main()
