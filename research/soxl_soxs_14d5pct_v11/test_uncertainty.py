"""Maturity, parent weighting, observation cutoff and case-gate boundaries."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from uncertainty import (core,build_bank,pool_for_day,decision,development_gate,restrict_market)
from run_uncertainty import authorize_cases,SOURCE

P = json.loads((SOURCE/'protocol.json').read_text())


def fixture(n=45):
    records=[];releases={}
    for i,day in enumerate(pd.date_range('2025-01-01',periods=n)):
        releases[day]=day.tz_localize(core.ET)+pd.Timedelta(days=14)
        for j in range(2):
            state={c:float(i+j) for c in P['local_columns'] if c!='raw'}
            records.append({'id':f'{i}-{j}','entry_date':day,'at':day.tz_localize(core.ET)+pd.Timedelta(hours=10+j),
                'state':state,'raw':.01,'target':.01+(i+j)/1000})
    return records,releases


class UncertaintyTests(unittest.TestCase):
    def test_parent_maximum_and_finite_sample_rank(self):
        records,releases=fixture();bank=build_bank(records,releases,P['local_columns'])
        pool,_=pool_for_day(bank,pd.Timestamp('2025-06-01',tz=core.ET),P)
        self.assertEqual(pool['source_rows'],90);self.assertEqual(len(pool['quantile_parents']),45)
        self.assertEqual(pool['quantile_rank_one_based'],42)
        self.assertAlmostEqual(pool['residual_upper'],.042)
        for i,value in enumerate(pool['parent_maxima']):self.assertAlmostEqual(value,(i+1)/1000)

    def test_current_future_late_release_and_missing_raw_excluded(self):
        records,releases=fixture();cut=pd.Timestamp('2025-06-01',tz=core.ET)
        records[0]['raw']=None
        releases[records[2]['entry_date']]=cut
        records[4]['at']=cut+pd.Timedelta(days=1)
        fake=deepcopy(records[-1]);fake.update(id='future',entry_date=pd.Timestamp('2025-06-01'),at=cut+pd.Timedelta(hours=1),raw=-100.,target=100.)
        records.append(fake);releases[fake['entry_date']]=cut-pd.Timedelta(days=1)
        pool,_=pool_for_day(build_bank(records,releases,P['local_columns']),cut,P)
        self.assertEqual(len(pool['eligible_parents']),42)
        self.assertNotIn(fake['entry_date'],pool['eligible_parents'])

    def test_unknown_state_does_not_silently_shrink_parent(self):
        records,releases=fixture(40);records[0]['state']=None
        pool,_=pool_for_day(build_bank(records,releases,P['local_columns']),pd.Timestamp('2025-06-01',tz=core.ET),P)
        self.assertEqual(len(pool['eligible_parents']),39);self.assertFalse(pool['ready'])

    def test_primary_requires_strict_negative_upper(self):
        record=fixture()[0][0];state=record['state'];pool={'ready':True,'residual_upper':.04}
        self.assertEqual(decision(-.01,state,pool,None,P,P['primary'])['action'],'HOLD')
        self.assertEqual(decision(-.04,state,pool,None,P,P['primary'])['action'],'HOLD')
        self.assertEqual(decision(-.041,state,pool,None,P,P['primary'])['action'],'EXIT')
        self.assertFalse(decision(None,state,pool,None,P,P['primary'])['ready'])

    def test_local_neighbors_are_distinct_parent_dates(self):
        records,releases=fixture();bank=build_bank(records,releases,P['local_columns'])
        pool,local=pool_for_day(bank,pd.Timestamp('2025-06-01',tz=core.ET),P)
        event=decision(.01,records[-1]['state'],pool,local,P,P['diagnostic'])
        self.assertEqual(len(event['neighbors']),40)
        self.assertEqual(bank.loc[event['neighbors'],'entry_date'].nunique(),40)
        values=bank.loc[event['neighbors'],'target'].to_numpy();self.assertAlmostEqual(event['bound'],values.mean()+1.645*values.std(ddof=1)/40**.5)

    def test_parent_replication_does_not_multiply_preprocessing_weight(self):
        records,releases=fixture();bank=build_bank(records,releases,P['local_columns'])
        _,a=pool_for_day(bank,pd.Timestamp('2025-06-01',tz=core.ET),P)
        repeated=[{**deepcopy(r),'id':r['id']+'copy'+str(i)} for i in range(10) for r in records[:2]]
        _,b=pool_for_day(build_bank(records+repeated,releases,P['local_columns']),pd.Timestamp('2025-06-01',tz=core.ET),P)
        np.testing.assert_allclose(a['center'],b['center']);np.testing.assert_allclose(a['scale'],b['scale'])

    def test_market_view_removes_future_prices_and_ex_dates(self):
        cut=pd.Timestamp('2026-05-01',tz=core.ET);days=pd.to_datetime(['2026-04-30','2026-05-01'])
        ends=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' 16:00',tz=core.ET) for d in days])
        bars=pd.DataFrame({'end_at':ends,'available_at':ends+pd.Timedelta(seconds=1),'close':[100.,9999.]})
        daily=pd.DataFrame({'available_at':ends+pd.Timedelta(seconds=1),'complete':True,'close':[100.,9999.]},index=days)
        actions=[{'date':'2026-04-30','payment_date':'2026-05-20'},{'date':'2026-05-01'}]
        market=core.Market({'SOXL':bars},{'SOXL':daily},{'future':'calendar'},days,{'SOXL':actions},{}, {})
        view=restrict_market(market,cut)
        self.assertEqual(view.bars['SOXL'].close.tolist(),[100.]);self.assertEqual(view.daily['SOXL'].close.tolist(),[100.])
        self.assertEqual(view.actions['SOXL'],actions[:1]);self.assertEqual(view.sessions.tolist(),days.tolist())
        market.bars['SOXL'].loc[1,'close']=-9999.;self.assertEqual(restrict_market(market,cut).bars['SOXL'].close.tolist(),[100.])

    def test_gate_cannot_be_rescued_by_aggregate_profit_or_boolean(self):
        stats={'monthly':[{'month':m,'return':.12,'mark_complete':True} for m in P['development']['months']],
               'completed_trades':10,'primary_pass':True}
        self.assertTrue(development_gate(stats,True,P)['passed'])
        stats['monthly'][1]['return']=-.01;stats['monthly'][0]['return']=.5
        self.assertFalse(development_gate(stats,True,P)['passed'])
        stats['monthly'][1]['return']=.12;stats['completed_trades']=9
        self.assertFalse(development_gate(stats,True,P)['passed'])

    def test_failed_verified_development_cannot_authorize_cases(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);result={'primary':{'monthly':[{'month':m,'return':0.,'mark_complete':True} for m in P['development']['months']],
                                         'completed_trades':20},'coverage':{'complete':True}}
            (p/'results.json').write_text(json.dumps(result))
            (p/'registration.json').write_text(json.dumps({'phase':'development','protocol_sha256':hashlib.sha256((SOURCE/'protocol.json').read_bytes()).hexdigest()}))
            (p/'verification.json').write_text(json.dumps({'passed':True,'results_sha256':hashlib.sha256((p/'results.json').read_bytes()).hexdigest()}))
            with self.assertRaisesRegex(ValueError,'gate failed'):authorize_cases(p,P)


if __name__=='__main__':unittest.main()
