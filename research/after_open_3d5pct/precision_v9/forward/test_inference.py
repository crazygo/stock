import hashlib
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from .features import CAUSAL_GROUP_PEERS, build_features_asof
from .inference import _canonical, _sha, load_route, predict_route
from .test_features import V8, real_fixture, row_from_npz, schema
from .ledger import Ledger, _at, issue_once, register_cohort, route_contract
from ...focus_v8.core import TARGETS, mature_prior, path_summary
from ...train_multiscale_v6 import tabular
from ...iterations_v7.b_group.experiment import curve_features, group_relative_features


SUMMARY=json.loads((V8/'round_10/summary.json').read_text())
CODE=Path(__file__).resolve().parents[2]
SCHEMA_TMP=tempfile.TemporaryDirectory()


def manifest(route,model_dir):
    recipe=SUMMARY[route]['incumbent_recipe']
    models=([model_dir/f'target_{j}.txt' for j in range(9)] if route.startswith('B')
            else [model_dir/'model.pt'])
    model_records=[{'path':str(p),'sha256':_sha(p)} for p in models]
    calibration=model_dir/'calibration.json'
    code=[CODE/'focus_v8/run.py',CODE/'focus_v8/core.py',CODE/'focus_v8/support.py',
          CODE/'train_multiscale_v6.py',CODE/'iterations_v7/b_group/experiment.py',
          CODE/'iterations_v7/c_no_daily/experiment.py',
          CODE/'precision_v9/forward/features.py',CODE/'precision_v9/forward/inference.py']
    source_records=[{'path':str(p),'sha256':_sha(p)} for p in code]
    s=schema(route)
    schema_file=Path(SCHEMA_TMP.name)/f'{route}.json'
    schema_file.write_bytes(_canonical(s))
    return {'route_id':route,'model_version':f'focus_v8_dev1_{route}',
            'recipe':recipe,'schema':s,
            'schema_file':{'path':str(schema_file),'sha256':_sha(schema_file)},
            'schema_sha256':hashlib.sha256(_canonical(s)).hexdigest(),
            'model_files':model_records,
            'model_sha256':hashlib.sha256(_canonical([m['sha256'] for m in model_records])).hexdigest(),
            'calibration_file':{'path':str(calibration),'sha256':_sha(calibration)},
            'calibration_sha256':_sha(calibration),'source_files':source_records,
            'source_code_sha256':hashlib.sha256(_canonical([m['sha256'] for m in source_records])).hexdigest(),
            'score_column':'raw_3d_5pct','thresholds':{'chips':.9,'optics':.9,'storage':.9},
            'action_frozen':False}


def frozen_row(symbol='AMD',date='2026-06-01'):
    rows=pd.read_parquet(V8/'dataset/rows.parquet')
    idx=int(rows.index[(rows.symbol==symbol)&(rows.session_date==date)][0])
    source=V8/'dataset/features.npz'
    data={name:row_from_npz(source,name,idx)[None] for name in ('x5','x60','xday','group_seq')}
    with np.load(source) as archive:
        y=archive['y'].reshape(-1,9)
    prior,_=mature_prior(rows,y)
    return rows.iloc[idx],idx,data,prior[idx],rows,y


def route_x(route,data,prior):
    grouped=route in ('B_group','C_group','C_no_daily')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        x={'x5':data['x5'][0], 'x60':data['x60'][0], 'prior':prior,
           'xbase':tabular(data,group=grouped).to_numpy(np.float32)[0],
           'paths':path_summary(data)[0],
           'curve':curve_features(data).to_numpy(np.float32)[0],
           'relative':(group_relative_features(data).to_numpy(np.float32)[0] if route=='B_group'
                       else np.empty(0,np.float32))}
    if route!='C_no_daily':x['xday']=data['xday'][0]
    if grouped:x['group_seq']=data['group_seq'][0]
    return x


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row,cls.idx,cls.data,cls.prior,cls.rows,cls.y=frozen_row()

    def test_all_five_real_checkpoints_replay_nine_raw_and_processed(self):
        for route in ('B_group','B_no_group','C_group','C_no_group','C_no_daily'):
            with self.subTest(route=route):
                path=Path(SUMMARY[route]['incumbent_paths'][0])
                artifact=load_route(manifest(route,path))
                X=route_x(route,self.data,self.prior)
                result=predict_route(artifact,{'X':X,'lineage':{'symbol':'AMD',
                    'session_date':'2026-06-01','sample_id':self.row.sample_id,
                    'mode':'historical_fixture','g2_eligible':False,
                    'group_peer_semantics':'legacy_static_only' if route in
                    ('B_group','C_group','C_no_daily') else 'excluded'},'rejections':[]})
                reference=pd.read_parquet(path/'eval.parquet').set_index('sample_id').loc[self.row.sample_id]
                for target in TARGETS:
                    self.assertLessEqual(abs(result['raw'][target]-reference['raw_'+target]),1e-7)
                    self.assertLessEqual(abs(result['processed'][target]-reference['p_'+target]),1e-7)
                self.assertFalse(result['quality']['publication_eligible'])
                self.assertFalse(result['quality']['g2_eligible'])
                if route in ('B_group','C_group','C_no_daily'):
                    self.assertFalse(result['quality']['g1_ready'])
                    self.assertIn('legacy_static_diagnostic_only',result['quality']['rejections'])
                self.assertEqual(len(result['raw']),9)

    def test_feature_only_neural_prior_route_needs_no_current_or_future_label(self):
        snapshot,decision,idx,rows=real_fixture('AMD','2026-06-01')
        own=np.flatnonzero((rows.symbol=='AMD').to_numpy())
        store=[{'sample_id':rows.iloc[i].sample_id,'symbol':'AMD',
                'decision_at':rows.iloc[i].decision_at,
                'label_end_at':rows.iloc[i].label_end_at,
                'label_available_at':rows.iloc[i].label_available_at,'y':self.y[i].tolist()}
               for i in own if i!=idx]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            built=build_features_asof(snapshot,decision,schema('C_no_group'),
                                     {'split_days':{'AMD':[]}},store)
        np.testing.assert_array_equal(built['X']['prior'],self.prior)
        self.assertEqual(len(built['lineage']['mature_prior_history_ids']),
                         sum(1 for x in store if pd.Timestamp(x['label_available_at'])<pd.Timestamp(decision['information_deadline_at'])))
        path=Path(SUMMARY['C_no_group']['incumbent_paths'][0])
        result=predict_route(load_route(manifest('C_no_group',path)),built)
        reference=pd.read_parquet(path/'eval.parquet').set_index('sample_id').loc[self.row.sample_id]
        self.assertLessEqual(max(abs(result['raw'][t]-reference['raw_'+t]) for t in TARGETS),1e-7)
        self.assertFalse(result['quality']['g2_eligible'])
        own=build_features_asof(snapshot,decision,schema('B_no_group'),
                                {'split_days':{'AMD':[]}},store)
        tree_path=Path(SUMMARY['B_no_group']['incumbent_paths'][0])
        tree=predict_route(load_route(manifest('B_no_group',tree_path)),own)
        tree_reference=pd.read_parquet(tree_path/'eval.parquet').set_index('sample_id').loc[self.row.sample_id]
        self.assertLessEqual(max(abs(tree['raw'][t]-tree_reference['raw_'+t]) for t in TARGETS),1e-7)

    def test_missing_manifest_hash_and_removed_route_inputs_fail_closed(self):
        path=Path(SUMMARY['B_no_group']['incumbent_paths'][0])
        original=manifest('B_no_group',path)
        broken=dict(original);broken.pop('schema_sha256')
        with self.assertRaisesRegex(ValueError,'missing'):
            load_route(broken)
        broken=dict(original);broken['model_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'aggregate hash'):
            load_route(broken)
        loaded=load_route(original)
        X=route_x('B_no_group',self.data,self.prior)
        X['group_seq']=self.data['group_seq'][0]
        with self.assertRaisesRegex(ValueError,'forbidden'):
            predict_route(loaded,X)
        neural=load_route(manifest('C_no_daily',Path(SUMMARY['C_no_daily']['incumbent_paths'][0])))
        X=route_x('C_no_daily',self.data,self.prior)
        X['xday']=self.data['xday'][0]
        with self.assertRaisesRegex(ValueError,'forbidden'):
            predict_route(neural,{'X':X,'lineage':{'mode':'historical_fixture',
                'g2_eligible':False,'group_peer_semantics':'legacy_static_only'},
                'rejections':[]})

    def test_legacy_group_checkpoint_cannot_claim_causal_feature_parity(self):
        route='C_group'
        old=manifest(route,Path(SUMMARY[route]['incumbent_paths'][0]))
        loaded=load_route(old)
        causal={'X':route_x(route,self.data,self.prior),
                'lineage':{'mode':'historical_fixture','g2_eligible':False,
                           'group_peer_semantics':CAUSAL_GROUP_PEERS,
                           'group_metadata_sha256':'a'*64},
                'rejections':[]}
        with self.assertRaisesRegex(ValueError,'peer semantics mismatch'):
            predict_route(loaded,causal)
        # Merely changing schema text and its digest cannot give an old
        # checkpoint a causal training provenance.
        forged=deepcopy(old)
        forged['schema']['group_peer_semantics']=CAUSAL_GROUP_PEERS
        forged['schema_sha256']=hashlib.sha256(_canonical(forged['schema'])).hexdigest()
        with self.assertRaisesRegex(ValueError,'legacy group checkpoint'):
            load_route(forged)
        forged['model_version']='r03_unverified_C_group'
        with self.assertRaisesRegex(ValueError,'training provenance'):
            load_route(forged)

    def test_real_feature_only_prediction_uses_ledger_decision_anchor(self):
        snapshot,decision,_,_=real_fixture('AMD','2026-06-01')
        built=build_features_asof(snapshot,decision,schema('B_no_group'),
                                 {'split_days':{'AMD':[]}},[])
        source=Path(SUMMARY['B_no_group']['incumbent_paths'][0])
        prediction=predict_route(load_route(manifest('B_no_group',source)),built)
        self.assertEqual(prediction['decision_at'],
                         pd.Timestamp(decision['cutoff_at']).tz_convert('UTC').isoformat())
        self.assertEqual(prediction['decision_at'],_at('2026-06-01',11,30).isoformat())
        self.assertEqual(prediction['information_deadline_at'],
                         pd.Timestamp(decision['information_deadline_at']).tz_convert('UTC').isoformat())
        calendar=json.loads((V8/'inputs/market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())['sessions']
        calendar=[x for x in calendar if '2026-06-01'<=x['session_date']]
        dates=[x['session_date'] for x in calendar[:60]]
        spec={'calendar_sessions':calendar,'groups':{'chips':['AMD'],
              'optics':['AMD'],'storage':['AMD']}}
        route={'route_id':'B_no_group','model_version':prediction['model_version'],
               'score_column':'raw_3d_5pct','thresholds':{'chips':.9,'optics':.9,'storage':.9},
               'artifacts':prediction['quality']['artifacts']}
        cohort={'cohort_id':'feature-contract-fixture','session_dates':dates,
                'calendar_sessions':calendar,
                'routes':['B_group','B_no_group','C_group','C_no_group','C_no_daily'],
                'groups':['chips','optics','storage'],
                'last_session_close_at':calendar[59]['close_at'],'evidence_mode':'fixture',
                'bootstrap_repeats':40}
        cohort['route_contracts']={name:route_contract(route) for name in cohort['routes']}
        with tempfile.TemporaryDirectory() as tmp:
            now=[pd.Timestamp('2026-05-31T12:00:00-04:00').to_pydatetime()]
            ledger=Ledger(Path(tmp)/'forward.sqlite',clock=lambda:now[0])
            try:
                register_cohort(cohort,ledger)
                now[0]=(pd.Timestamp(decision['cutoff_at'])+pd.Timedelta(seconds=10)).to_pydatetime()
                result=issue_once(spec,route,cohort,'real-feature-fixture',[prediction],
                                  {'g2_eligible':False},ledger)[0]
                self.assertEqual(result['payload']['reason'],'data_unverified')
                self.assertNotIn('reason_detail',result['payload'])
            finally:
                ledger.close()


if __name__=='__main__':unittest.main()
