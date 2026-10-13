import copy
import unittest
import math
import numpy as np
import pandas as pd
from reaction_inputs import normalize,response,PANEL,ET
from reference_reaction import reference


class ReactionTests(unittest.TestCase):
    def fixture(self):
        dates=pd.bdate_range('2024-01-02','2024-03-29').strftime('%Y-%m-%d').tolist()
        sessions=[dict(session_date=d) for d in dates]
        raw={s:pd.DataFrame(dict(time_key=[d+' 00:00:00' for d in dates],open=100.,high=100.,low=100.,close=100.,volume=100.)) for s in PANEL+['SPY']}
        actions={s:pd.DataFrame(columns=['ex_div_date','split_ratio','per_cash_div']) for s in raw}
        return raw,actions,sessions

    def tables(self,raw,actions,sessions):return {s:normalize(q,actions[s],sessions) for s,q in raw.items()}

    def compare_reference(self,events,at,raw,actions,sessions):
        actual=response(events,at,self.tables(raw,actions,sessions),sessions)
        native={s:{x['time_key'][:10]:x for x in d.to_dict('records')} for s,d in raw.items()}
        expected=reference(events,at,native,{s:a.to_dict('records') for s,a in actions.items()},sessions)
        np.testing.assert_allclose(actual['values'],expected['values'],atol=1e-13)
        self.assertEqual(actual['pending'],expected['pending'])
        return actual

    def test_first_following_session_and_next_midnight_with_dst(self):
        raw,actions,sessions=self.fixture();events=[['NVDA','2024-03-08']]
        for at in ['2024-03-11T15:05:30-04:00','2024-03-11T23:59:59-04:00']:
            value=self.compare_reference(events,at,raw,actions,sessions)
            self.assertEqual(value['values'],[0,0,0,0]);self.assertEqual(value['pending'],[['NVDA','2024-03-08','2024-03-11']])
        value=self.compare_reference(events,'2024-03-12T00:00:00-04:00',raw,actions,sessions)
        self.assertEqual(value['values'],[1,0,0,0]);self.assertEqual(value['pending'],[])
        self.assertEqual(value['observed'][0]['available_at'],'2024-03-12T00:00:00-04:00')

    def test_non_session_planned_day_uses_previous_close(self):
        raw,actions,sessions=self.fixture()
        value=self.compare_reference([['NVDA','2024-03-09']],'2024-03-12T09:25:30-04:00',raw,actions,sessions)
        self.assertEqual(value['observed'][0]['base_date'],'2024-03-08')

    def test_issuer_is_excluded_from_peer_response_and_volume_excludes_current(self):
        raw,actions,sessions=self.fixture();r=raw['NVDA'];mask=r.time_key.str[:10]=='2024-03-11'
        r.loc[mask,['open','high','low','close']]=102.;r.loc[mask,'volume']=200.
        value=self.compare_reference([['NVDA','2024-03-08']],'2024-03-12T09:25:30-04:00',raw,actions,sessions)
        np.testing.assert_allclose(value['values'],[1,math.log(1.02),0,math.log(2)],atol=1e-13)
        self.assertEqual(len(value['observed'][0]['volume_history']),20)
        self.assertNotIn('2024-03-11',value['observed'][0]['volume_history'])

    def test_equal_market_move_cancels_excess_returns(self):
        raw,actions,sessions=self.fixture()
        for r in raw.values():r.loc[r.time_key.str[:10]=='2024-03-11',['open','high','low','close']]=102.
        value=self.compare_reference([['NVDA','2024-03-08']],'2024-03-12T09:25:30-04:00',raw,actions,sessions)
        np.testing.assert_allclose(value['values'],[1,0,0,0],atol=1e-13)

    def test_split_changes_share_units_without_fake_price_or_volume_jump(self):
        raw,actions,sessions=self.fixture();r=raw['NVDA'];mask=r.time_key.str[:10]>='2024-03-11'
        r.loc[mask,['open','high','low','close']]=10.;r.loc[mask,'volume']=1000.
        actions['NVDA']=pd.DataFrame([dict(ex_div_date='2024-03-11',split_ratio=.1,per_cash_div=np.nan)])
        value=self.compare_reference([['NVDA','2024-03-08']],'2024-03-12T09:25:30-04:00',raw,actions,sessions)
        np.testing.assert_allclose(value['values'],[1,0,0,0],atol=1e-13)

    def test_cash_entitlement_is_in_economic_return(self):
        raw,actions,sessions=self.fixture();r=raw['NVDA'];mask=r.time_key.str[:10]=='2024-03-11'
        r.loc[mask,['open','high','low','close']]=99.
        actions['NVDA']=pd.DataFrame([dict(ex_div_date='2024-03-11',split_ratio=np.nan,per_cash_div=1.)])
        value=self.compare_reference([['NVDA','2024-03-08']],'2024-03-12T09:25:30-04:00',raw,actions,sessions)
        np.testing.assert_allclose(value['values'],[1,0,0,0],atol=1e-13)

    def test_future_prices_and_actions_cannot_change_current_response(self):
        raw,actions,sessions=self.fixture();events=[['NVDA','2024-03-08']];at='2024-03-12T09:25:30-04:00'
        first=self.compare_reference(events,at,raw,actions,sessions)
        for r in raw.values():r.loc[r.time_key.str[:10]>='2024-03-12',['open','high','low','close','volume']]=10000.
        actions['NVDA']=pd.DataFrame([dict(ex_div_date='2024-03-15',split_ratio=.01,per_cash_div=np.nan)])
        second=self.compare_reference(events,at,raw,actions,sessions)
        self.assertEqual(first,second)

    def test_missing_mature_peer_or_volume_is_unknown_not_zero(self):
        raw,actions,sessions=self.fixture();tables=self.tables(raw,actions,sessions)
        tables['AMD']=tables['AMD'].drop('2024-03-11')
        value=response([['NVDA','2024-03-08']],'2024-03-12T09:25:30-04:00',tables,sessions)
        self.assertFalse(value['known']);self.assertIsNone(value['values'])
        tables=self.tables(raw,actions,sessions);tables['NVDA']=tables['NVDA'].drop('2024-03-07')
        value=response([['NVDA','2024-03-08']],'2024-03-12T09:25:30-04:00',tables,sessions)
        self.assertFalse(value['known']);self.assertIsNone(value['values'])

    def test_unsupported_or_simultaneous_action_remains_unknown(self):
        raw,actions,sessions=self.fixture()
        actions['NVDA']=pd.DataFrame([dict(ex_div_date='2024-03-11',split_ratio=.1,per_cash_div=1.)])
        value=response([['NVDA','2024-03-08']],'2024-03-12T09:25:30-04:00',self.tables(raw,actions,sessions),sessions)
        self.assertFalse(value['known'])

    def test_empty_pending_and_complete_neutral_have_distinct_states(self):
        raw,actions,sessions=self.fixture();tables=self.tables(raw,actions,sessions)
        empty=response([],'2024-03-11T09:25:30-04:00',tables,sessions)
        pending=response([['NVDA','2024-03-08']],'2024-03-11T09:25:30-04:00',tables,sessions)
        neutral=response([['NVDA','2024-03-08']],'2024-03-12T09:25:30-04:00',tables,sessions)
        self.assertEqual(empty['values'],pending['values']);self.assertNotEqual(empty['pending'],pending['pending'])
        self.assertEqual(neutral['values'],[1,0,0,0])

    def test_duplicates_future_and_stale_plan_are_rejected(self):
        raw,actions,sessions=self.fixture();tables=self.tables(raw,actions,sessions)
        for events in [[['NVDA','2024-03-08']]*2,[['NVDA','2024-03-12']],[['NVDA','2024-02-01']]]:
            with self.assertRaises(ValueError):response(events,'2024-03-12T09:25:30-04:00',tables,sessions)


if __name__=='__main__':unittest.main()
