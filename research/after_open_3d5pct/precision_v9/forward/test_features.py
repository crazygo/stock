import json
from pathlib import Path
import unittest
import zipfile

import numpy as np
import pandas as pd

from .features import (CAUSAL_GROUP_PEERS, LEGACY_GROUP_PEERS, SHAPES,
                       build_features_asof)
from .. import data as d
from ...focus_v8.core import TARGETS, mature_prior, path_summary
from ...train_multiscale_v6 import tabular


V8=Path(__file__).resolve().parents[2]/'runs/focus_v8_20260927'


def schema(route='B_no_group', *, group_peer_semantics=LEGACY_GROUP_PEERS):
    grouped=route in {'B_group','C_group','C_no_daily'}
    result={'route_id':route,'targets':list(TARGETS),'feature_columns':list(d.FEATURE_COLUMNS),
            'input_shapes':{k:list(v) for k,v in SHAPES.items()},'price_basis':'NONE',
            'tensor_order':['x5','x60','xday','group_seq','prior'],
            'tabular_width':765 if grouped else 585,'path_width':84,'curve_width':8,
            'relative_width':6 if route=='B_group' else 0,
            'prior_rule':'own_complete_nine_beta_1_1_last_63',
            'group_mode':'causal_weekly' if grouped else 'excluded',
            'normalizer':'fixed_no_fitted_scaler',
            'support_rule':'checkpoint_buffers' if route in ('C_group','C_no_daily') else 'none',
            'tabular_columns':[f'f_{i:03d}' for i in range(765 if grouped else 585)],
            'path_columns':[f'path_{i:03d}' for i in range(84)],
            'curve_columns':['open40_ret','mid40_ret','late40_ret','prefix_drawdown',
                             'prefix_runup','prefix_efficiency','late_early_vol_delta',
                             'late_early_logdollar_delta'],
            'relative_columns':([f'own_minus_{name}_cutoff_return' for name in
                                ('trend15','trend63','trend126','volatility','liquidity','qqq')]
                                if route=='B_group' else [])}
    if grouped:
        result['group_peer_semantics']=group_peer_semantics
    return result


def row_from_npz(path,name,index):
    """Read one compressed NPY row without materializing the 487 MB archive."""
    with zipfile.ZipFile(path) as archive, archive.open(name+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        reader=(np.lib.format.read_array_header_1_0 if version==(1,0)
                else np.lib.format.read_array_header_2_0)
        shape,fortran,dtype=reader(stream)
        assert not fortran
        step=int(np.prod(shape[1:]))*dtype.itemsize
        remaining=index*step
        while remaining:
            data=stream.read(min(remaining,1<<20))
            if not data:raise EOFError(name)
            remaining-=len(data)
        data=stream.read(step)
        if len(data)!=step:raise EOFError(name)
        return np.frombuffer(data,dtype).reshape(shape[1:]).copy()


def real_fixture(symbol,date):
    calendar=json.loads((V8/'inputs/market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())['sessions']
    # Frozen v8 v6_data._calendar starts at 2026-01-01 despite the source
    # calendar file also carrying December 2025 sessions.
    sessions=[x for x in calendar if '2026-01-01'<=x['session_date']<=date]
    bar=pd.read_parquet(V8/'inputs/market_data/us_5m'/symbol/'2026.parquet')
    session=sessions[-1]
    cutoff=pd.Timestamp(session['open_at'])+pd.Timedelta(hours=2)
    decision={'symbol':symbol,'session_date':date,'cutoff_at':cutoff.isoformat(),
              'decision_at':cutoff.isoformat(),
              'information_deadline_at':(cutoff+pd.Timedelta(seconds=30)).isoformat()}
    rows=pd.read_parquet(V8/'dataset/rows.parquet')
    ix=int(rows.index[(rows.symbol==symbol)&(rows.session_date==date)][0])
    return {'bars':{symbol:bar},'sessions':sessions,'mode':'historical_fixture'},decision,ix,rows


class FeatureTests(unittest.TestCase):
    def test_real_frozen_rows_match_feature_only_encoder(self):
        for symbol,date in [('AMD','2026-06-01'),('COHR','2026-06-02')]:
            with self.subTest(symbol=symbol,date=date):
                snapshot,decision,ix,rows=real_fixture(symbol,date)
                built=build_features_asof(snapshot,decision,schema(),
                    {'split_days':{symbol:[]}},[])
                self.assertEqual(built['rejections'],[])
                self.assertFalse(built['lineage']['g2_eligible'])
                X=built['X']
                reference={name:row_from_npz(V8/'dataset/features.npz',name,ix)
                           for name in ('x5','x60','xday')}
                for name in reference:
                    np.testing.assert_array_equal(X[name],reference[name])
                data={name:reference[name][None] for name in reference}
                data['group_seq']=np.zeros((1,6,6,10),np.float32)
                np.testing.assert_array_equal(X['xbase'],tabular(data,group=False).to_numpy(np.float32)[0])
                np.testing.assert_array_equal(X['paths'],path_summary(data)[0])

    def test_future_and_late_revisions_or_other_stock_cannot_change_own_input(self):
        snapshot,decision,_,_=real_fixture('AMD','2026-06-01')
        baseline=build_features_asof(snapshot,decision,schema(),{'split_days':{'AMD':[]}},[])
        altered={**snapshot,'bars':dict(snapshot['bars'])}
        bars=snapshot['bars']['AMD'].copy()
        future=bars.session_date=='2026-06-02'
        bars.loc[future,'high']*=2
        revision=bars.loc[bars.session_date=='2026-06-01'].iloc[[0]].copy()
        revision['received_at']='2026-06-01T17:00:00+00:00'
        revision['source_id']='late_revision'
        bars=pd.concat([bars,revision],ignore_index=True)
        altered['bars']['AMD']=bars
        altered['bars']['COHR']=bars.copy()
        future_label={'sample_id':'AMD:future','symbol':'AMD','decision_at':'2026-06-02T15:30:30+00:00',
                      'label_end_at':'2026-06-05T20:00:00+00:00',
                      'label_available_at':'2026-06-05T20:00:01+00:00','y':[1]*9}
        changed=build_features_asof(altered,decision,schema(),
                                    {'split_days':{'AMD':['2026-06-02']}},[future_label])
        for name in baseline['X']:
            np.testing.assert_array_equal(baseline['X'][name],changed['X'][name])
        self.assertEqual(baseline['lineage']['source_ids'],changed['lineage']['source_ids'])
        rejected=build_features_asof(snapshot,decision,schema(),
                                     {'split_days':{'AMD':['2026-06-01']}},[])
        self.assertEqual(rejected['rejections'],['split_in_input_window'])

    def test_reject_outcome_entry_and_missing_receipt_in_live(self):
        snapshot,decision,_,_=real_fixture('AMD','2026-06-01')
        snapshot['bars']['AMD']['entry_price']=100
        with self.assertRaisesRegex(ValueError,'outcome or entry'):
            build_features_asof(snapshot,decision,schema(),{'split_days':{'AMD':[]}},[])
        snapshot['bars']['AMD']=snapshot['bars']['AMD'].drop(columns='entry_price')
        snapshot['mode']='live'
        with self.assertRaisesRegex(ValueError,'received_at'):
            build_features_asof(snapshot,decision,schema(),{'split_days':{'AMD':[]}},[])

    def test_prior_uses_all_mature_candidate_rows_not_buy_only(self):
        _,decision,_,rows=real_fixture('AMD','2026-06-01')
        # A non-buy candidate is still eligible if all nine labels matured.
        past={'sample_id':'AMD:prior_non_buy','symbol':'AMD','decision_at':'2026-05-01T15:30:30+00:00',
              'label_end_at':'2026-05-08T20:00:00+00:00',
              'label_available_at':'2026-05-08T20:00:01+00:00','y':[1]*9}
        future={**past,'sample_id':'AMD:future','label_available_at':'2026-06-03T20:00:01+00:00'}
        from .features import _prior
        p,ids=_prior('AMD',pd.Timestamp(decision['information_deadline_at']),[past,future])
        self.assertEqual(ids,['AMD:prior_non_buy'])
        np.testing.assert_array_equal(p,np.full(9,2/3,np.float32))

    def test_future_group_member_version_cannot_change_prior_state(self):
        date='2026-06-01'
        opening=pd.Timestamp('2026-06-01 09:30',tz='America/New_York')
        session={'session_date':date,'open_at':opening.isoformat(),
                 'close_at':pd.Timestamp('2026-06-01 16:00',tz='America/New_York').isoformat(),
                 'duration_minutes':390}
        bars={}
        for symbol,base in [('AMD',100.),('COHR',120.),('QQQ',500.)]:
            rows=[]
            for j in range(24):
                start=opening+pd.Timedelta(minutes=j*5)
                end=start+pd.Timedelta(minutes=5)
                price=base+j*.1
                rows.append({'session_date':date,'start_at_et':start.isoformat(),
                             'end_at':end.isoformat(),'available_at':(end+pd.Timedelta(seconds=1)).isoformat(),
                             'session_type':'regular','price_basis':'NONE',
                             'open':price,'high':price+.1,'low':price-.1,'close':price+.05,
                             'volume':100.,'turnover':10000.})
            bars[symbol]=pd.DataFrame(rows)
        snapshot={'bars':bars,'sessions':[session],'mode':'historical_fixture'}
        decision={'symbol':'AMD','session_date':date,
                  'cutoff_at':'2026-06-01T11:30:00-04:00',
                  'decision_at':'2026-06-01T11:30:00-04:00',
                  'information_deadline_at':'2026-06-01T11:30:30-04:00'}
        version={'week_id':'2026-W23','strategy_id':'trend_15','version_id':'known',
                 'membership_basis':'causal_weekly_reconstruction',
                 'feature_cutoff_at':'2026-05-29T20:00:00+00:00',
                 'effective_from':'2026-05-29T20:00:00+00:00',
                 'effective_to':'2026-06-08T14:30:00+00:00'}
        member=lambda symbol,version_id:{'symbol':symbol,'week_id':'2026-W23',
                 'strategy_id':'trend_15','version_id':version_id,'group_ids':['trend_15:up'],
                 'facts':{'history_end':'2026-05-29'}}
        meta={'split_days':{'AMD':[]},'versions':[version],
              'memberships':[member('AMD','known'),member('COHR','known')],
              'universe_symbols':['AMD','COHR'],
              'group_peer_semantics':CAUSAL_GROUP_PEERS}
        with np.errstate(all='ignore'):
            before=build_features_asof(snapshot,decision,
                                       schema('B_group',group_peer_semantics=CAUSAL_GROUP_PEERS),meta,[])
        self.assertEqual(before['X']['group_seq'][-1,0,9],1.)
        future={**version,'version_id':'future',
                'feature_cutoff_at':'2026-06-02T20:00:00+00:00',
                'effective_from':'2026-06-02T20:00:00+00:00'}
        changed={**meta,'versions':[version,future],
                 'memberships':meta['memberships']+[member('XYZ','future')]}
        after=build_features_asof(snapshot,decision,
                                  schema('B_group',group_peer_semantics=CAUSAL_GROUP_PEERS),changed,[])
        for name in before['X']:
            np.testing.assert_array_equal(before['X'][name],after['X'][name])
        self.assertEqual(before['lineage']['group_version_ids'],after['lineage']['group_version_ids'])


if __name__=='__main__':unittest.main()
