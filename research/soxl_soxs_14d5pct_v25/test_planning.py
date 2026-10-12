import json,unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from planning import *
from run_planning import json_ready
from cycle_ledger import replay


class PlanningTests(unittest.TestCase):
    def fixture(self,n=170):
        rng=np.random.default_rng(25);days=pd.bdate_range('2025-01-01',periods=n)
        ats=(days.tz_localize(core.ET)+pd.Timedelta(hours=9,minutes=25,seconds=30)).tz_convert('UTC')
        panel=pd.DataFrame(rng.normal(size=(n,19)),index=ats,columns=COLUMNS);panel['date']=days;panel['feature_ready']=True
        bank=[]
        for i,t in enumerate(ats[:-4]):
            nexts=[ats[i+1],ats[i+2],ats[i+3]]
            bank.append({'at':t,'date':days[i],'next_at':nexts,'next_date':list(days[[i+1,i+2,i+3]]),
                'ready_at':max(nexts),'expired_after':days[min(i+12,n-1)],'duration':[1,2,3],
                'reward':[0.,float(rng.normal(.01,.04)),float(rng.normal(.01,.04))]})
        return panel,bank

    def test_next_cycle_waits_for_T1_and_paid_dividend(self):
        days=pd.DatetimeIndex(['2026-01-02','2026-01-05','2026-01-06','2026-01-07'])
        market=SimpleNamespace(sessions=days);lab={'exit_at':pd.Timestamp('2026-01-02 16:00',tz=core.ET),'net_return':.03,'dividends':[]}
        self.assertEqual(next_cycle(market,lab),days[1])
        lab['dividends']=[{'pay':days[2]}];self.assertEqual(next_cycle(market,lab),days[2])
        lab['net_return']=None;self.assertIsNone(next_cycle(market,lab))

    def test_incomplete_earlier_transition_cannot_be_skipped(self):
        p,b=self.fixture(20);b[2]['ready_at']=None
        ids,m=admitted(b,p.index[-1]);self.assertEqual(ids,[b[0]['at'],b[1]['at']]);self.assertTrue(m['expired_unresolved'])
        self.assertGreater(m['known_behind_pending'],0)

    def test_exact_cutoff_and_future_features_not_admitted(self):
        p,b=self.fixture(20);cut=b[2]['ready_at'];ids,_=admitted(b,cut)
        self.assertNotIn(b[2]['at'],ids)
        after,_=admitted(b,cut+pd.Timedelta(seconds=1));self.assertIn(b[2]['at'],after)

    def test_future_rewards_and_states_do_not_change_current_fit(self):
        p,b=self.fixture();at=p.index[150];ids,_=admitted(b,day_cutoff(at));a,ia=fit(p,b,ids)
        q=p.copy();q.loc[q.index>=at,COLUMNS]=1e9
        changed=[{**r,'reward':[0.,99.,-99.]} if r['at'] not in ids else r for r in b]
        z,iz=fit(q,changed,ids);self.assertEqual(ia,iz)
        self.assertEqual(raw_prediction(a,p.loc[at]),raw_prediction(z,p.loc[at]))

    def test_actual_duration_changes_discounted_first_stage_target(self):
        p,b=self.fixture();ids=[r['at'] for r in b[:130]]
        for r in b:r['reward']=[0.,.1,.1];r['duration']=[1,1,8]
        model,info=fit(p,b,ids);v=raw_prediction(model,p.iloc[-1])['myopic']
        self.assertAlmostEqual(v[1],GAMMA*.1);self.assertAlmostEqual(v[2],GAMMA**8*.1)

    def test_calibration_uses_saved_next_values_not_future_truth(self):
        p,b=self.fixture();ids=[r['at'] for r in b[:140]]
        history={t:{'planning':[.01,.02,.03],'myopic':[0.,.01,.02]} for t in p.index[:145]}
        a=calibrate(b,ids,history,'planning');history[p.index[-1]]={'planning':[999.]*3,'myopic':[999.]*3}
        self.assertEqual(a,calibrate(b,ids,history,'planning'))
        self.assertTrue(a['ready']);self.assertEqual(len(a['members']),120)

    def test_waiting_value_can_beat_positive_buy_and_ties_wait(self):
        raw={'planning':[.04,.03,.02]};cal={'ready':True,'correction':[0.,0.,0.]};meta={'expired_unresolved':False}
        self.assertEqual(decide(raw,cal,'planning',meta)['action'],'CASH')
        raw['planning']=[.03,.03,.02];self.assertEqual(decide(raw,cal,'planning',meta)['action'],'CASH')
        raw['planning']=[-.01,-.02,-.03];self.assertEqual(decide(raw,cal,'planning',meta)['action'],'CASH')

    def test_record_roundtrip_and_unknown_null(self):
        a=np.array([.1234567890123456,.8765432109876544]);obj={'raw':a,'unknown':np.nan}
        result=json.loads(json.dumps(json_ready(obj),allow_nan=False));np.testing.assert_array_equal(result['raw'],a);self.assertIsNone(result['unknown'])

    def test_replay_retains_receivable_and_waits_for_full_cash(self):
        days=pd.bdate_range('2026-01-05',periods=5);ats=(days.tz_localize(core.ET)+pd.Timedelta(hours=9,minutes=25,seconds=30)).tz_convert('UTC')
        features=pd.DataFrame({'date':days[:4],'clock':['09:25:30']*4},index=ats[:4]);labs=[]
        for i,at in enumerate(ats[:4]):
            labs.append({'decision_at':at,'symbol':'SOXL','entry_at':at+pd.Timedelta(minutes=5),'entry_fill_economic':100.,
                'exit_at':pd.Timestamp(str(days[1 if i==0 else i].date())+' 16:00',tz=core.ET).tz_convert('UTC'),
                'exit_fill_economic':105.,'entry_share_factor':1.,'dividends':[{'ex':days[1],'pay':days[3],'economic_amount':1.}] if i==0 else [],
                'net_return':.05,'reason':'target','attached_order':{}})
        market=SimpleNamespace(sessions=days,daily={'SOXL':pd.DataFrame({'close':[100.]*5,'complete':[True]*5},index=days)})
        book=replay(market,features,labs,{at:'SOXL' for at in ats[:4]},str(days[0].date()),str(days[3].date()))
        self.assertEqual(list(book[2].decision),['buy_SOXL','position_held','waiting_for_full_cash','buy_SOXL'])
        self.assertGreater(book[0].receivables.iloc[2],0.);self.assertEqual(book[0].receivables.iloc[3],0.)


if __name__=='__main__':unittest.main()
