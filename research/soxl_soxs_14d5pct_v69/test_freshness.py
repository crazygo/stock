"""Latest known prices are usable without reading any unused future state."""
import copy,json,unittest
from unittest.mock import patch
from path_learning import *
from test_activity_state import fixture
from audit_paths import check_forecast
from run_planning import json_ready


class FreshnessTests(unittest.TestCase):
    def test_future_successor_does_not_delay_complete_prices(self):
        bank,market,panel,at=fixture();bank['successor_at_ns'][-1]=at.value;bank['successor_states'][-1]=np.nan
        self.assertNotIn(158,source_pool(bank,at,CLOCKS[0]));self.assertIn(158,price_known_pool(bank,at,CLOCKS[0]))
        model,info,raw=forecast(bank,market,panel,at,'fresh')
        canonical=json.loads(json.dumps(json_ready(info),default=str));expected,n=check_forecast(bank,market,panel,at,'fresh',canonical,model)
        np.testing.assert_array_equal(raw,expected);self.assertGreater(n,4096)

    def test_successor_arrays_are_not_read(self):
        bank,market,panel,at=fixture();model,info,raw=forecast(bank,market,panel,at,'fresh')
        del bank['successor_states'];del bank['successor_at_ns']
        other,changed,value=forecast(bank,market,panel,at,'fresh')
        np.testing.assert_array_equal(raw,value);np.testing.assert_array_equal(model['selected_blocks'],other['selected_blocks'])
        self.assertEqual(info['members'],changed['members'])

    def test_unused_transition_matrix_is_not_called(self):
        bank,market,panel,at=fixture()
        with patch('path_learning.transition_matrix',side_effect=AssertionError('must not read unused transition')):
            _,info,raw=forecast(bank,market,panel,at,'fresh');self.assertTrue(info['ready']);self.assertIsNotNone(raw)

    def test_missing_equal_cutoff_and_future_price_releases_rejected(self):
        bank,_,_,at=fixture()
        for value in [pd.NaT.value,day_cutoff(at).value,(at+pd.Timedelta(days=1)).value]:
            changed=copy.deepcopy(bank);changed['release_ns'][7]=value;members=price_known_pool(changed,at,CLOCKS[0])
            self.assertNotIn(6,members);self.assertNotIn(7,members)

    def test_missing_price_reference_and_state_still_rejected(self):
        bank,_,_,at=fixture()
        for key,index in [('quotes',(7,0,0)),('states',(7,0,0)),('absolute_states',(7,0)),('activity_states',(7,0))]:
            changed=copy.deepcopy(bank);changed[key][index]=np.nan
            self.assertNotIn(7,price_known_pool(changed,at,CLOCKS[0]))

    def test_same_eligible_pool_preserves_all_price_draws(self):
        bank,market,panel,at=fixture();a,ai,x=forecast(bank,market,panel,at,'fresh');b,bi,y=forecast(bank,market,panel,at,'conditional')
        self.assertEqual(ai['members'],bi['members']);np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(a['selected_blocks'],b['selected_blocks'])

    def test_future_price_values_cannot_change_prediction(self):
        bank,market,panel,at=fixture();bank['release_ns'][-1]=at.value;a,_,x=forecast(bank,market,panel,at,'fresh')
        bank['ohlc'][-1]*=9;bank['quotes'][-1]*=4;b,_,y=forecast(bank,market,panel,at,'fresh')
        np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(a['selected_blocks'],b['selected_blocks'])

if __name__=='__main__':unittest.main()
