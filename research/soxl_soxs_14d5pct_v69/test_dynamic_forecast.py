"""Whole-query eligibility, full dynamic forecast and future-input invariance."""
import json,unittest,copy
from test_activity_state import fixture as market_fixture
from test_state_paths import fixture as references_fixture
from state_paths import roll_state
from path_learning import *
from audit_paths import check_forecast
from run_planning import json_ready


def fixture():
    bank,market,panel,at=market_fixture();_,ref,h,q,_,_,_,_=references_fixture();n=len(bank['date_ns']);ref={k:a[:n].copy() for k,a in ref.items()};ref['release_ns']=bank['release_ns'].copy();bank['states'][:,:,:3]=np.random.default_rng(69).normal(size=(n,8,3));panel.loc[at,PRIOR]=roll_state(h[None,:])[0]
    query={'at_ns':np.array([at.value]),'history':h[None,:],'quotes':q[None,:],'ready':np.ones(1,bool),'history_release_ns':np.array([at.value-pd.Timedelta(days=3).value]),'quote_release_ns':np.array([at.value-pd.Timedelta(seconds=29).value])}
    context={'reference':ref,'queries':query,'positions':{at.value:0}}
    return bank,market,panel,at,context


def checked(args):
    b,m,p,at,c=args;model,info,raw=forecast(b,m,p,at,'dynamic',context=c);canonical=json.loads(json.dumps(json_ready(info),default=str));expected,_=check_forecast(b,m,p,at,'dynamic',canonical,model,context=c)
    if raw is None:assert expected is None
    else:np.testing.assert_allclose(raw,expected,rtol=1e-10,atol=1e-11)
    return model,info,raw


class DynamicForecastTests(unittest.TestCase):
    def test_full_query_mixed_absorption_and_state_equations(self):
        args=fixture();args[0]['ohlc'][:30,4,0,1]=106.;args[0]['ohlc'][:30,4,1,2]=93.;model,info,raw=checked(args);self.assertTrue(info['ready']);self.assertTrue(model['resolved'].any() and (~model['resolved']).any());self.assertEqual(info['state_transition']['paths'],2048)

    def test_missing_reference_disables_whole_query_without_changing_pool(self):
        args=fixture();b,m,p,at,c=args;_,control,_=forecast(b,m,p,at,'fresh');c['reference']['regular_close'][7,40,9]=np.nan;model,info,raw=checked(args);self.assertIsNone(model);self.assertIsNone(raw);self.assertEqual(info['members'],control['members']);self.assertEqual(info['state_inputs']['reason'],'reference_support_missing')

    def test_current_history_must_precede_day_not_decision(self):
        args=fixture();b,m,p,at,c=args;c['queries']['history_release_ns'][0]=day_cutoff(at).value;model,info,raw=checked(args);self.assertFalse(info['ready']);self.assertEqual(info['state_inputs']['reason'],'incomplete_current_history_or_quote')

    def test_future_prices_labels_successors_and_context_do_not_change_query(self):
        args=fixture();b,m,p,at,c=args;b['release_ns'][-1]=at.value;a,info,raw=checked(args);b['ohlc'][-1]*=999.;c['reference']['regular_close'][-1]=np.nan;c['reference']['release_ns'][-1]=at.value;del b['successor_states'];del b['successor_at_ns'];p['C']=999.;p['terminal_release']=at+pd.Timedelta(days=99)
        other,_,changed=checked(args);np.testing.assert_array_equal(raw,changed);np.testing.assert_array_equal(a['selected_blocks'],other['selected_blocks']);np.testing.assert_array_equal(a['state_path']['states_before_blocks'],other['state_path']['states_before_blocks'])

if __name__=='__main__':unittest.main()
