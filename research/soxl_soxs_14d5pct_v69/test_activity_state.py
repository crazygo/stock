import unittest,copy
from types import SimpleNamespace
from path_learning import *
from audit_paths import check_forecast
from run_planning import json_ready
import json


def fixture():
    n=160;dates=pd.bdate_range('2025-01-02',periods=n);at=pd.Timestamp('2026-01-05 14:25:30Z');end=at-pd.Timedelta(seconds=30)
    bank={'ohlc':np.full((n,78,2,4),100.),'quotes':np.full((n,8,2),100.),'previous_close':np.full((n,2),100.),'split_ratio':np.ones((n,2)),'dividend_ratio':np.zeros((n,2)),'states':np.ones((n,8,4)),
          'blocks':np.array([[i,i+1] for i in range(n-1)]),'date_ns':dates.asi8,'release_ns':np.full(n,pd.Timestamp('2025-09-01 20:00Z').value,dtype='int64'),
          'absolute_states':np.ones((n,8)),'activity_states':np.tile(np.linspace(-2,2,n)[:,None],(1,8)),'successor_states':np.ones((n-1,3)),'successor_at_ns':np.full(n-1,pd.Timestamp('2025-09-02 13:25:30Z').value,dtype='int64')}
    days=pd.bdate_range('2026-01-05','2026-01-30');cal={d:{'open_at':pd.Timestamp(str(d.date())+' 09:30',tz=core.ET).tz_convert('UTC'),'close_at':pd.Timestamp(str(d.date())+' 16:00',tz=core.ET).tz_convert('UTC')} for d in days}
    frame=pd.DataFrame([{'start_at':end-pd.Timedelta(minutes=5),'available_at':end+pd.Timedelta(seconds=1),'close':100.,'econ_close':100.}],index=[end])
    market=SimpleNamespace(calendar=cal,bars={s:frame for s in core.ETFS});panel=pd.DataFrame([{'clock':CLOCKS[0],'feature_ready':True,ABSOLUTE:1.,ACTIVITY:1.5,**{f:1. for f in FIELDS}}],index=[at])
    return bank,market,panel,at


class ActivityStateTests(unittest.TestCase):
    def test_activity_distinguishes_identical_price_states(self):
        bank,_,panel,at=fixture();row=panel.loc[at].copy();_,old,pos=first_kernels(bank,np.arange(159),CLOCKS[0],row)
        row[ACTIVITY]=-1.5;_,other,neg=first_kernels(bank,np.arange(159),CLOCKS[0],row)
        np.testing.assert_array_equal(old['probability'],other['probability'])
        self.assertGreater(np.array(pos['probability'])@bank['activity_states'][:159,0],np.array(neg['probability'])@bank['activity_states'][:159,0])

    def test_constant_activity_preserves_five_field_kernel(self):
        bank,_,panel,at=fixture();bank['states'][:]=np.random.default_rng(65).normal(size=(160,8,4));bank['activity_states'][:]=1.5
        _,old,new=first_kernels(bank,np.arange(159),CLOCKS[0],panel.loc[at])
        np.testing.assert_allclose(old['probability'],new['probability'],atol=1e-15)

    def test_only_first_selection_changes_and_independent_valuation(self):
        bank,market,panel,at=fixture();a,ai,_=forecast(bank,market,panel,at,'conditional');b,bi,_=forecast(bank,market,panel,at,'fixed')
        self.assertFalse(np.array_equal(a['selected_blocks'][:,0],b['selected_blocks'][:,0]));np.testing.assert_array_equal(a['selected_blocks'][:,1:],b['selected_blocks'][:,1:])
        np.testing.assert_array_equal(ai['kernels']['later']['probability'],bi['kernels']['later']['probability'])
        for kind,model,info in [('conditional',a,ai),('fixed',b,bi)]:check_forecast(bank,market,panel,at,kind,json.loads(json.dumps(json_ready(info),default=str)),model)

    def test_missing_query_activity_abstains_all_variants(self):
        bank,market,panel,at=fixture();panel.loc[at,ACTIVITY]=np.nan
        for k in VARIANTS:
            model,info,raw=forecast(bank,market,panel,at,k);self.assertFalse(info['ready']);self.assertIsNone(model);self.assertIsNone(raw)

    def test_missing_source_activity_excluded_from_common_pool(self):
        bank,_,_,at=fixture();bank['activity_states'][7,0]=np.nan
        self.assertNotIn(7,source_pool(bank,at,CLOCKS[0]));self.assertEqual(len(source_pool(bank,at,CLOCKS[0])),158)

    def test_future_contract_labels_do_not_change_query(self):
        bank,market,panel,at=fixture();model,_,raw=forecast(bank,market,panel,at,'conditional');panel['C']=999.;panel['D']=-999.;panel['terminal_release']=at+pd.Timedelta(days=99)
        changed,_,other=forecast(bank,market,panel,at,'conditional');np.testing.assert_array_equal(raw,other);np.testing.assert_array_equal(model['selected_blocks'],changed['selected_blocks'])

if __name__=='__main__':unittest.main()
