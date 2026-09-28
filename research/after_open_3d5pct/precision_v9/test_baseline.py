import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from . import baseline as b
from . import evaluate as ev
from .acquire_batch import make_calendar
from ..focus_v8.core import mature_prior
from ..train_multiscale_v6 import tabular


def _row(day, *, label_end=None, symbol='AMD'):
    et = pd.Timestamp(day+' 11:30', tz='America/New_York')
    entry = et + pd.Timedelta(minutes=5)
    end = pd.Timestamp((label_end or day)+' 16:00', tz='America/New_York')
    return {'sample_id':f'{symbol}:{day}', 'symbol':symbol, 'session_date':day,
            'cutoff_at':et.isoformat(),
            'decision_at':(et+pd.Timedelta(seconds=30)).isoformat(),
            'entry_at':entry.isoformat(), 'entry_price':100.,
            'label_end_at':end.isoformat(),
            'label_available_at':(end+pd.Timedelta(seconds=1)).isoformat(),
            'terminal_1d':0., 'terminal_3d':0., 'terminal_5d':0.,
            'full_126_prior_days':0, 'available_daily_days':50, 'group_valid_count':0}


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = self.root/'old';self.new = self.root/'new'
        old_rows = pd.DataFrame([
            _row('2026-03-10',label_end='2026-04-21'),  # purged at tune boundary
            _row('2026-03-20',label_end='2026-03-25'),
            _row('2026-04-20',label_end='2026-04-24'),
            _row('2026-05-11',label_end='2026-05-15'),
            _row('2026-06-01',label_end='2026-06-05')])
        new_rows = pd.concat([pd.DataFrame([_row('2025-10-01',label_end='2025-10-08')]),
                              old_rows],ignore_index=True)
        for root, rows in ((self.old,old_rows),(self.new,new_rows)):
            ds = root/'dataset';ds.mkdir(parents=True)
            rows.to_parquet(ds/'rows.parquet',index=False)
            y=np.zeros((len(rows),3,3),np.float32)
            if root == self.new:y[0]=1
            np.savez_compressed(ds/'features.npz',y=y)
            (ds/'manifest.json').write_text('{}')
        self.allowed=self.root/'allowed.parquet'
        old_rows[['symbol','session_date']].to_parquet(self.allowed,index=False)
        self.calendar=self.root/'calendar.json'
        self.calendar.write_text(json.dumps(make_calendar()))
        self.summary=self.root/'summary.json'
        self.summary.write_text(json.dumps({'B_no_group':{'incumbent_recipe':{
            'representation':True,'regularization':True,'calibration':True}}}))
        for name in ('backlog.md','matrix.md','fit_adoption.md'):
            (self.root/name).write_text('{}')
        (self.root/'paired.json').write_text(json.dumps({
            'paired_rows':5,'arrays':{'y':{'changed_rows':0}},
            'fields':{name:{'changed_rows':0} for name in b.LABEL_FIELDS}}))
        (self.root/'actual.json').write_text(json.dumps({
            'status':'accepted_for_fit',
            'evidence_scope':'exposed_development',
            'limitations':['current_snapshot_retrospective','historical_availability_assumed',
                           'corporate_actions_historical_coverage_uncertified'],
            'dataset_sha256':{name:ev.sha(self.new/'dataset'/name)
                              for name in ('rows.parquet','features.npz','manifest.json')},
            'calendar_sha256':ev.sha(self.calendar),
            'paired_audit_sha256':ev.sha(self.root/'paired.json')}))
        self.plan={'old_dataset_root':str(self.old),'new_dataset_root':str(self.new),
                   'dataset_root':str(self.new),'calendar':str(self.calendar),
                   'allowed_keys':str(self.allowed),'incumbent_summary':str(self.summary),
                   'actual_manifest':str(self.root/'actual.json'),
                   'paired_audit':str(self.root/'paired.json'),
                   'backlog':str(self.root/'backlog.md'),'matrix':str(self.root/'matrix.md'),
                   'native_fit_adoption':str(self.root/'fit_adoption.md'),
                   'native_run_id':'fixture_r03_native_plan',
                   'route':'B_no_group','fold':'dev1','arm':'new_expanded_fit',
                   'fit_date_floor':'2025-10-01','seed':3566}
        self.plan_path=self.root/'plan.json'
        self.plan_path.write_text(json.dumps(self.plan))

    def test_common_keys_are_exact_and_labels_match(self):
        allowed, old_rows, new_rows = b.validate_common_keys(self.old,self.new,self.allowed)
        self.assertEqual(len(allowed),5)
        new_rows.loc[2,'terminal_3d']=.2
        new_rows.to_parquet(self.new/'dataset/rows.parquet',index=False)
        with self.assertRaisesRegex(ValueError,'terminal mismatch'):
            b.validate_common_keys(self.old,self.new,self.allowed)

    def test_reject_duplicate_key_and_incomplete_common_list(self):
        rows=pd.read_parquet(self.new/'dataset/rows.parquet')
        pd.concat([rows,rows.iloc[[1]]]).to_parquet(self.new/'dataset/rows.parquet',index=False)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            b.validate_common_keys(self.old,self.new,self.allowed)
        rows.to_parquet(self.new/'dataset/rows.parquet',index=False)
        pd.read_parquet(self.allowed).iloc[:-1].to_parquet(self.allowed,index=False)
        with self.assertRaisesRegex(ValueError,'full common set'):
            b.validate_common_keys(self.old,self.new,self.allowed)

    def test_expanded_fit_keeps_full_prior_history_and_purges_crossing_label(self):
        rows=pd.read_parquet(self.new/'dataset/rows.parquet')
        allowed=b._allowed_keys(self.allowed)
        fold=b._fold_config('dev1')
        selected=b.split_indices(rows,fold,allowed,'new_expanded_fit','2025-10-01')
        self.assertEqual(rows.iloc[selected['fit']].session_date.tolist(),['2025-10-01','2026-03-20'])
        self.assertEqual(rows.iloc[selected['eval']].session_date.tolist(),['2026-06-01'])
        with np.load(self.new/'dataset/features.npz') as z:y=z['y'].reshape(-1,9)
        prior,counts=mature_prior(rows,y)
        march=rows.index[rows.session_date=='2026-03-20'][0]
        self.assertEqual(counts[march],1)
        self.assertAlmostEqual(float(prior[march,4]),2/3)
        bridge=b.split_indices(rows,fold,allowed,'new_common','2026-03-10')
        self.assertEqual(rows.iloc[bridge['fit']].session_date.tolist(),['2026-03-20'])
        # Eligibility differs; the new snapshot's as-of prior does not.
        self.assertAlmostEqual(float(prior[march,4]),2/3)

    def test_calendar_denominator_keeps_cross_year_no_signal_sessions(self):
        dates,_=b._calendar(self.calendar)
        window=b.official_dates(dates,'2025-12-23','2026-01-05')
        self.assertEqual(len(window),7)
        frame=pd.DataFrame([{'symbol':'AMD','session_date':'2025-12-23',
                             'y_3d_5pct':0.,'terminal_3d':0.}])
        signals=pd.DataFrame(columns=[*frame.columns,'trigger_groups'])
        result=ev.summarize(frame,signals,'chips',window)
        self.assertEqual(result['evaluation_sessions'],7)
        self.assertEqual(result['signals_per_5_sessions'],0)
        self.assertIsNone(result['precision'])
        _,sessions=b._calendar(self.calendar)
        entry=pd.Timestamp('2025-12-23 11:35',tz='America/New_York')
        source=pd.DataFrame([{'sample_id':'A:2025-12-23'}])
        rows=pd.DataFrame([{'sample_id':'A:2025-12-23','entry_at':entry.isoformat(),
                            'terminal_3d':0.}])
        enriched=ev.enrich(source,rows,sessions)
        self.assertEqual(pd.Timestamp(enriched.signal_end_at.iloc[0]).tz_convert(
            'America/New_York').strftime('%Y-%m-%d %H:%M'),'2025-12-29 14:35')

    def test_registration_precedes_fit_and_detects_source_mutation(self):
        output=self.root/'output'
        context=b.prepare_run(self.plan_path,output)
        self.assertTrue((output/'registration.json').exists())
        self.assertFalse((output/'result.json').exists())
        self.assertEqual(context['registration']['split_support']['effective_fit']['rows'],2)
        self.assertEqual(context['registration']['prior_scope'],
                         'all_rows_in_selected_immutable_dataset_before_allowed_key_filter')
        from . import native_artifact
        registration, contract, split = native_artifact.verify_prefit(
            output, context['registration']['native_contract_file'])
        self.assertEqual(contract['identity']['route'], registration['route'])
        self.assertEqual(set(split['splits']), {'fit', 'tune', 'cal', 'eval', 'eval_full'})
        b.verify_sources(context['registration'])
        (self.root/'matrix.md').write_text('mutated')
        with self.assertRaisesRegex(ValueError,'source changed'):
            b.verify_sources(context['registration'])
        with self.assertRaisesRegex(ValueError, 'pre-fit source changed'):
            native_artifact.verify_prefit(output, context['registration']['native_contract_file'])

    def test_bad_paired_audit_stops_before_registration(self):
        paired=self.root/'paired.json'
        data=json.loads(paired.read_text())
        data['arrays']['y']['changed_rows']=1
        paired.write_text(json.dumps(data))
        output=self.root/'blocked'
        with self.assertRaisesRegex(ValueError,'paired builder audit'):
            b.prepare_run(self.plan_path,output)
        self.assertFalse(output.exists())

    def test_actual_manifest_must_accept_exact_snapshot(self):
        path=self.root/'actual.json'
        manifest=json.loads(path.read_text())
        manifest['status']='draft'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'has not accepted'):
            b.prepare_run(self.plan_path,self.root/'draft')
        manifest['status']='accepted_for_fit'
        manifest['dataset_sha256']['rows.parquet']='0'*64
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'dataset hashes differ'):
            b.prepare_run(self.plan_path,self.root/'wrong_hash')

    def test_engineering_smoke_requires_self_paired_dev1_registration(self):
        plan={**self.plan,'purpose':'engineering_validation_only',
              'new_dataset_root':str(self.old),'dataset_root':str(self.old),
              'arm':'old_common','fit_date_floor':'2026-03-10'}
        actual=json.loads((self.root/'actual.json').read_text())
        actual['status']='engineering_validation_only'
        actual['dataset_sha256']={name:ev.sha(self.old/'dataset'/name)
                                  for name in ('rows.parquet','features.npz','manifest.json')}
        (self.root/'actual.json').write_text(json.dumps(actual))
        self.root.joinpath('engineering.json').write_text(json.dumps(plan))
        output=self.root/'engineering'
        context=b.prepare_run(self.root/'engineering.json',output)
        self.assertEqual(context['registration']['purpose'],'engineering_validation_only')
        self.assertIn(str(b.HERE.parent/'focus_v8/prepare.py'),
                      context['registration']['source_hashes'])
        plan['arm']='new_common'
        (self.root/'engineering.json').write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError,'engineering replay'):
            b.prepare_run(self.root/'engineering.json',self.root/'invalid_engineering')

    def test_zero_signal_eval_keeps_full_official_denominator(self):
        rows=pd.read_parquet(self.new/'dataset/rows.parquet')
        dates,sessions=b._calendar(self.calendar)
        output=self.root/'selection';output.mkdir()
        def frame(day):
            item=rows.loc[rows.session_date==day,
                          ['sample_id','symbol','session_date','decision_at']].copy()
            for target in b.TARGETS:
                item['y_'+target]=0.
                item['raw_'+target]=.1
                item['p_'+target]=.1
            return item
        context={'rows':rows,'dates':dates,'sessions':sessions,
                 'fold':b._fold_config('dev1'),'output':output}
        _, result=b._selection(context,{'cal':frame('2026-05-11'),
                                         'eval':frame('2026-06-01'),
                                         'eval_full':frame('2026-06-01')})
        score=result['eval:cal_selected']['chips']
        self.assertEqual(score['n'],0)
        self.assertIsNone(score['precision'])
        self.assertEqual(score['evaluation_sessions'],
                         len(b.official_dates(dates,'2026-06-01','2026-06-27')))

    def test_specialize_reports_positive_weight_fit_support(self):
        rows=pd.DataFrame([_row('2026-03-20',symbol='AMD'),
                           _row('2026-03-20',symbol='XYZ'),
                           _row('2026-03-21',symbol='AMD')])
        fold={'fit':np.array([0,1,2]),'tune':np.array([0]),
              'cal':np.array([0]),'eval':np.array([0])}
        support=b._support(rows,fold,{'specialize':True},3566)
        self.assertEqual(support['fit']['rows'],3)
        self.assertEqual(support['effective_fit']['rows'],2)
        self.assertEqual(support['effective_fit']['dates'],2)

    def test_prior_history_ids_are_separate_from_fit_supervision(self):
        rows=pd.read_parquet(self.new/'dataset/rows.parquet')
        with np.load(self.new/'dataset/features.npz') as z:y=z['y'].reshape(-1,9)
        prior,counts=mature_prior(rows,y)
        fold=b.split_indices(rows,b._fold_config('dev1'),b._allowed_keys(self.allowed),
                             'new_common','2026-03-10')
        output=self.root/'prior';output.mkdir()
        context={'fold_ids':fold,'registration':{'seed':3566},'output':output}
        b._write_supervision_and_prior(context,(rows,None,y,prior,counts),{'prior':True})
        fit=pd.read_parquet(output/'fit_supervision_ids.parquet')
        history=pd.read_parquet(output/'prior_history_ids.parquet')
        self.assertEqual(fit.session_date.tolist(),['2026-03-20'])
        march=history.loc[rows.session_date=='2026-03-20'].iloc[0]
        self.assertEqual(march.prior_history_ids.tolist(),['AMD:2025-10-01'])

    def test_b_no_group_schema_excludes_group_tensor(self):
        data={'x5':np.zeros((1,8,192,14),np.float32),
              'x60':np.zeros((1,31,17,14),np.float32),
              'xday':np.zeros((1,126,14),np.float32),
              'group_seq':np.zeros((1,6,6,10),np.float32)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            original=tabular(data,group=True).to_numpy(np.float32)
            data['group_seq'][:]=7
            changed=tabular(data,group=True).to_numpy(np.float32)
        np.testing.assert_array_equal(original[:,:585],changed[:,:585])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            schema=b._assert_b_no_group_schema((None,data,None,None,None,
                                                np.zeros((1,84)),changed,None,None,None,None),
                                               {'representation':True})
        self.assertEqual(schema['base']['count']+schema['path']['count'],669)

    def test_runner_writes_predictions_with_registered_old_fit_interface(self):
        rows=pd.read_parquet(self.new/'dataset/rows.parquet')
        with np.load(self.new/'dataset/features.npz') as z:y=z['y'].reshape(-1,9)
        prior,counts=mature_prior(rows,y)
        n=len(rows)
        data={'x5':np.zeros((n,8,192,14),np.float32),
              'x60':np.zeros((n,31,17,14),np.float32),
              'xday':np.zeros((n,126,14),np.float32),
              'group_seq':np.zeros((n,6,6,10),np.float32)}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            xbase=tabular(data,group=True).to_numpy(np.float32)
        loaded=(rows,data,y,prior,counts,np.zeros((n,84),np.float32),xbase,
                np.empty((n,0)),np.empty((n,0)),(),{})
        called=[]
        def fake_fit(route,recipe,rows,y,prior,xbase,paths,curve,relative,fold,output,seed):
            self.assertTrue((output/'registration.json').exists())
            called.append((route,seed))
            return ({name:np.full((len(ids),9),.1) for name,ids in fold.items()},
                    {'features':669,'trees':[1]*9,'replay_max_error':0.})
        output=self.root/'run'
        with warnings.catch_warnings(), patch.object(b.old,'load_data',return_value=loaded), \
             patch.object(b.old,'fit_b',side_effect=fake_fit), \
             patch.object(b.old,'calibrate',return_value=[[1.,0.]]*9):
            warnings.simplefilter('ignore',RuntimeWarning)
            result=b.run(self.plan_path,output)
        self.assertEqual(called,[('B_no_group',3566)])
        self.assertTrue(result['source_unchanged'])
        self.assertEqual(result['scores']['eval']['rows'],1)
        self.assertTrue((output/'selection.json').exists())
        self.assertTrue((output/'eval_cal_selected_signals.parquet').exists())


if __name__ == '__main__':unittest.main()
