"""Offline adversarial tests for R03 request-level local source reuse."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd

from . import acquire_batch as batch
from . import acquire_recovery as recovery
from . import acquire_local_reuse as acquisition
from . import local_reuse_source as local
from .prepare import validate_acquisition
from .test_data import _r03_original, _r03_recovery
from .evaluate import HERE, ROOT, sha


def _write(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def _raw(symbol='TEST', dates=('2025-08-04 09:35:00','2025-09-04 09:35:00')):
    return pd.DataFrame([{'code':'US.'+symbol,'time_key':date,
        'open':10.,'high':10.2,'low':9.9,'close':10.1,'volume':100.,
        'turnover':1000.,'pe_ratio':0.,'turnover_rate':0.,'change_rate':0.,
        'last_close':10.} for date in dates])


def _fixture(root: Path, raw=None):
    external=root/'external';cache=root/'cache';script=root/'backfill_model_history.py'
    calendar_script=root/'model_history_calendar.py'
    shutil.copy2(ROOT/'scripts/backfill_model_history.py',script)
    shutil.copy2(ROOT/'scripts/model_history_calendar.py',calendar_script)
    reg={'symbols':['TEST']+[f'X{i:03d}' for i in range(108)],'interval':'5m',
         'session':'ALL','price_basis':'NONE','no_automatic_cloud_writes':True,
         'script_sha256':sha(script)}
    _write(external/'registration.json',reg)
    _write(external/'calendar.json',batch.make_calendar())
    request=cache/'requests/TEST/2025-08-01_2025-09-30/attempt_1'
    request.mkdir(parents=True)
    raw=_raw() if raw is None else raw
    raw.to_parquet(request/'raw_001.parquet',index=False)
    received='2026-09-27T11:14:25+00:00'
    audit=[{'symbol':'TEST','start':'2025-08-01','end':'2025-09-30',
            'page':1,'attempt':1,'requested_at':'2026-09-27T11:14:24+00:00',
            'received_at':received,'ok':True,'rows':len(raw),
            'sha256':sha(request/'raw_001.parquet'),'has_more':False}]
    _write(request/'audit.json',audit)
    _write(request/'result.json',{'symbol':'TEST','start':'2025-08-01',
            'end':'2025-09-30','status':'fetched'})
    normalized=recovery.normalize(raw,'TEST')
    if len(normalized): normalized.to_parquet(request/'normalized.parquet',index=False)
    for month in ('2025-08','2025-09'):
        part=external/'parts/TEST'/month
        part.mkdir(parents=True)
        chunk=normalized.loc[normalized.session_date.str[:7]==month].copy()
        _write(part/'complete.json',{'source':str(request.resolve()),
             'pagination_complete':True,'status':'source_returned',
             'request_received_at':received})
        if len(chunk):
            chunk.to_parquet(part/'bars.parquet',index=False)
            _write(part/'imports.json',[{'source':str(request/'normalized.parquet'),
                                         'sha256':sha(request/'normalized.parquet')}])
    return external,cache,script,calendar_script,request


class LocalReuseTests(unittest.TestCase):
    def test_external_scope_requires_frozen_code_and_calendar(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);external,cache,script,calendar_script,_=_fixture(root)
            expected=json.loads((external/'registration.json').read_text())['symbols']
            identity=local.source_identity(external,cache,expected,script,calendar_script)
            self.assertEqual(identity['candidate_symbols'],expected)
            reg=json.loads((external/'registration.json').read_text());reg['price_basis']='QFQ'
            _write(external/'registration.json',reg)
            with self.assertRaises(local.SourceConflict):
                local.source_identity(external,cache,expected,script,calendar_script)
            reg['price_basis']='NONE';_write(external/'registration.json',reg)
            cal=json.loads((external/'calendar.json').read_text())
            day=next(s for s in cal['sessions'] if s['session_date']=='2025-11-28')
            day['duration_minutes']=390
            _write(external/'calendar.json',cal)
            with self.assertRaises(local.SourceConflict):
                local.source_identity(external,cache,expected,script,calendar_script)

    def test_multimonth_request_frozen_once_and_replayed_by_month(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);external,cache,script,calendar_script,request=_fixture(root)
            first=local.freeze_request(external,cache,script,calendar_script,'TEST','2025-08',root/'bundles')
            second=local.freeze_request(external,cache,script,calendar_script,'TEST','2025-09',root/'bundles')
            self.assertEqual(first[0],second[0])
            self.assertEqual(len(list((root/'bundles').glob('TEST/*/manifest.json'))),1)
            self.assertEqual(first[2]['request_pages'],1)
            self.assertEqual(first[2]['data_state'],'observed_partial_coverage')
            self.assertEqual(second[2]['quality']['rows'],1)
            self.assertFalse(first[2]['quality']['rth_complete'])
            self.assertEqual(local.verify_bundle(first[0],'TEST','2025-09')[1]['quality']['rows'],1)
            manifest=json.loads((first[0]/'manifest.json').read_text())
            self.assertEqual(manifest['months'],['2025-08','2025-09'])
            self.assertTrue(any(x['bundle_path']=='request/raw_001.parquet' for x in manifest['files']))
            self.assertFalse(any(x['bundle_path'].endswith('security_metadata.json') for x in manifest['files']))

    def test_complete_without_result_is_pending_and_metadata_is_not_empty(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);external,cache,script,calendar_script,request=_fixture(root)
            (request/'result.json').unlink()
            with self.assertRaises(local.PendingSource):
                local.freeze_request(external,cache,script,calendar_script,'TEST','2025-08',root/'bundles')
            marker=external/'parts/TEST/2025-09/complete.json'
            _write(marker,{'source':'security_metadata_and_official_history',
                           'status':'before_current_regular_trading','pagination_complete':True})
            self.assertEqual(local.locate(external,cache,'TEST','2025-09')[0],'metadata_only')
            self.assertIsNone(local.freeze_request(external,cache,script,calendar_script,
                                                    'TEST','2025-09',root/'bundles'))

    def test_empty_month_requires_actual_complete_covering_request(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);external,cache,script,calendar_script,request=_fixture(root,
                raw=_raw(dates=('2025-08-04 09:35:00',)))
            bundle,frame,evidence=local.freeze_request(external,cache,script,calendar_script,
                                    'TEST','2025-09',root/'bundles')
            self.assertTrue(frame.empty)
            self.assertTrue(evidence['empty_success'])
            self.assertEqual(evidence['data_state'],'provider_returned_no_bars_unknown_listing_state')
            self.assertFalse(evidence['quality']['rth_complete'])
            doc=json.loads((request/'result.json').read_text());doc['end']='2025-08-31'
            _write(request/'result.json',doc)
            with self.assertRaises(local.SourceConflict):
                local.freeze_request(external,cache,script,calendar_script,'TEST','2025-09',root/'other')

    def test_missing_page_wrong_hash_symbol_duplicate_and_bad_ohlc_reject(self):
        for defect in ('page','hash','symbol','duplicate','offgrid','ohlc','basis'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as t:
                root=Path(t);external,cache,script,calendar_script,request=_fixture(root)
                audit=json.loads((request/'audit.json').read_text())
                if defect=='page':audit[0]['has_more']=True
                elif defect=='hash':audit[0]['sha256']='0'*64
                elif defect in ('symbol','duplicate','offgrid','ohlc'):
                    raw=pd.read_parquet(request/'raw_001.parquet')
                    if defect=='symbol':raw.loc[0,'code']='US.OTHER'
                    elif defect=='duplicate':raw.loc[1,'time_key']=raw.loc[0,'time_key']
                    elif defect=='offgrid':raw.loc[0,'time_key']='2025-08-04 09:36:00'
                    else:raw.loc[0,'high']=9.
                    raw.to_parquet(request/'raw_001.parquet',index=False)
                    audit[0].update(sha256=sha(request/'raw_001.parquet'),rows=len(raw))
                elif defect=='basis':
                    bars=external/'parts/TEST/2025-08/bars.parquet'
                    frame=pd.read_parquet(bars);frame['price_basis']='QFQ'
                    frame.to_parquet(bars,index=False)
                _write(request/'audit.json',audit)
                with self.assertRaises(local.SourceConflict):
                    local.freeze_request(external,cache,script,calendar_script,
                                          'TEST','2025-08',root/'bundles')

    def test_source_change_during_copy_and_merged_revision_reject(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);external,cache,script,calendar_script,request=_fixture(root)
            changed=False
            def mutate(source,dest):
                nonlocal changed
                if not changed:
                    changed=True
                    source.write_bytes(source.read_bytes()+b' ')
            with self.assertRaises(local.SourceConflict):
                local.freeze_request(external,cache,script,calendar_script,
                    'TEST','2025-08',root/'bundles',copy_hook=mutate)
            self.assertFalse(list((root/'bundles').glob('TEST/*/manifest.json')))
            failures=list((root/'bundles').glob('TEST/*.staging/failure.json'))
            self.assertEqual(len(failures),1)
            self.assertEqual(json.loads(failures[0].read_text())['status'],'freeze_failed_uncommitted')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);external,cache,script,calendar_script,request=_fixture(root)
            imports=external/'parts/TEST/2025-08/imports.json'
            items=json.loads(imports.read_text());_write(imports,items+items)
            revision=cache/'source_revisions/TEST/2025-08'
            _write(revision/'resolution.json',{'differing_rows':1})
            with self.assertRaisesRegex(local.SourceConflict,'merged/revised'):
                local.freeze_request(external,cache,script,calendar_script,
                                     'TEST','2025-08',root/'bundles')

    def test_dst_and_half_day_normalize_to_et_calendar(self):
        aug=recovery.normalize(_raw(dates=('2025-08-04 09:35:00',)),'TEST')
        nov=recovery.normalize(_raw(dates=('2025-11-03 09:35:00',)),'TEST')
        self.assertEqual(pd.Timestamp(aug.iloc[0].start_at).hour,13)
        self.assertEqual(pd.Timestamp(nov.iloc[0].start_at).hour,14)
        early=batch.normalize(_raw(dates=('2025-11-28 13:05:00',)),'TEST')
        self.assertEqual(early.iloc[0].session_type,'post_market')

    def test_provider_audit_order_timezone_and_null_code_reject(self):
        for defect in ('after_terminal','wrong_attempt','naive_time','null_code'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as t:
                root=Path(t);external,cache,script,calendar_script,request=_fixture(root)
                audit=json.loads((request/'audit.json').read_text())
                if defect=='after_terminal':
                    audit.append({**audit[-1], 'page':2,'ok':False,'attempt':1})
                elif defect=='wrong_attempt':audit[0]['attempt']=2
                elif defect=='naive_time':audit[0]['received_at']='2026-09-27T11:14:25'
                else:
                    raw=pd.read_parquet(request/'raw_001.parquet')
                    raw.loc[0,'code']=None
                    raw.to_parquet(request/'raw_001.parquet',index=False)
                    audit[0]['sha256']=sha(request/'raw_001.parquet')
                _write(request/'audit.json',audit)
                with self.assertRaises(local.SourceConflict):
                    local.freeze_request(external,cache,script,calendar_script,
                                         'TEST','2025-08',root/'bundles')

    def test_failed_attempt_then_successful_retry_is_valid(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);external,cache,script,calendar_script,request=_fixture(root)
            audit=json.loads((request/'audit.json').read_text())
            failure={**audit[0], 'ok':False,'requested_at':'2026-09-27T11:14:21+00:00',
                     'received_at':'2026-09-27T11:14:22+00:00'}
            audit[0]['attempt']=2
            _write(request/'audit.json',[failure,audit[0]])
            bundle,frame,evidence=local.freeze_request(external,cache,script,calendar_script,
                                      'TEST','2025-08',root/'bundles')
            self.assertEqual(evidence['request_pages'],1)
            self.assertEqual(len(frame),1)

    def test_old_recovery_empty_never_accepts_new_local_tier(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=_r03_original(root);run=_r03_recovery(root,source)
            part=run/'parts/TEST/2025-12'
            meta=json.loads((part/'result.json').read_text())
            meta['tier']='local_verified_provider'
            _write(part/'result.json',meta)
            _write(part/'audit.json',[])
            (part/'raw_001.parquet').unlink()
            with self.assertRaisesRegex(ValueError,'empty provider reply not audited'):
                validate_acquisition(run,['TEST'])

    def test_prepare_explicit_version_rechecks_bundle_and_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=_r03_original(root)
            external,cache,script,calendar_script,request=_fixture(root)
            run=root/'new';run.mkdir()
            original=json.loads((source/'registration.json').read_text())
            code={name:sha(path) for name,path in acquisition.CODE_PATHS.items()}
            ident={'version':local.VERSION,'source_run':str(source.resolve()),
                   'source_registration_sha256':sha(source/'registration.json'),
                   'source_registration':original,'session':'ALL','price_basis':'NONE',
                   'no_upload':True,'source_scope':'original_305_first_candidate_only_108_plus_qqq_aug_dec_2025',
                   'code_sha256':code,
                   'external_identity':{'candidate_symbols':['TEST'],
                       'registration_sha256':sha(external/'registration.json'),
                       'calendar_sha256':sha(external/'calendar.json'),
                       'script_sha256':sha(script),
                       'calendar_script_sha256':sha(calendar_script)}}
            _write(run/'registration.json',{'identity':ident})
            _write(run/'calendar.json',batch.make_calendar())
            snapshot=run/'source_snapshot';snapshot.mkdir()
            for name,path in acquisition.CODE_PATHS.items():
                dest=snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest)
            for start,_ in batch.MONTHS[1:]:
                month=start[:7]
                recovery._inherit(source/'parts/TEST'/month,run/'parts/TEST'/month,
                                  'TEST',month,run)
            bundle,frame,evidence=local.freeze_request(external,cache,script,calendar_script,
                                          'TEST','2025-08',run/'source_bundles')
            acquisition._local_result(run/'parts/TEST/2025-08','TEST','2025-08',
                                      bundle,frame,evidence,batch.make_calendar())
            with self.assertRaisesRegex(ValueError,'original complete part must be inherited'):
                validate_acquisition(run,['TEST'])
            original_part=source/'parts/TEST/2025-08/result.json'
            original_meta=json.loads(original_part.read_text())
            original_meta['pagination_complete']=False
            original_meta['failure']='source_unresolved'
            _write(original_part,original_meta)
            provenance,coverage,empty,gaps=validate_acquisition(run,['TEST'])
            self.assertEqual(len(coverage),5)
            self.assertFalse(empty)
            self.assertEqual(len(gaps),5)
            self.assertIn('local_source_bundle_input',{p['role'] for p in provenance})
            dependency=snapshot/'scripts/fetch_research_data.py'
            original_dependency=dependency.read_bytes()
            dependency.write_bytes(original_dependency+b' ')
            with self.assertRaisesRegex(ValueError,'local-reuse source code changed'):
                validate_acquisition(run,['TEST'])
            dependency.write_bytes(original_dependency)
            receipt=run/'parts/TEST/2025-08/audit.json'
            saved=receipt.read_bytes();_write(receipt,[])
            with self.assertRaisesRegex(local.SourceConflict,'local import receipt changed'):
                validate_acquisition(run,['TEST'])
            receipt.write_bytes(saved)
            raw=bundle/'request/raw_001.parquet';raw.write_bytes(raw.read_bytes()+b' ')
            with self.assertRaises((local.SourceConflict,ValueError)):
                validate_acquisition(run,['TEST'])

    def test_prepare_accepts_verified_multimonth_empty_without_full_coverage_claim(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);source=_r03_original(root)
            external,cache,script,calendar_script,_=_fixture(root,
                raw=_raw(dates=('2025-08-04 09:35:00',)))
            source_result=source/'parts/TEST/2025-09/result.json'
            old=json.loads(source_result.read_text())
            old.update(pagination_complete=False,failure='source_unresolved')
            _write(source_result,old)
            run=root/'new';run.mkdir()
            original=json.loads((source/'registration.json').read_text())
            code={name:sha(path) for name,path in acquisition.CODE_PATHS.items()}
            ident={'version':local.VERSION,'source_run':str(source.resolve()),
                   'source_registration_sha256':sha(source/'registration.json'),
                   'source_registration':original,'session':'ALL','price_basis':'NONE',
                   'no_upload':True,'source_scope':'original_305_first_candidate_only_108_plus_qqq_aug_dec_2025',
                   'code_sha256':code,
                   'external_identity':{'candidate_symbols':['TEST'],
                       'registration_sha256':sha(external/'registration.json'),
                       'calendar_sha256':sha(external/'calendar.json'),
                       'script_sha256':sha(script),
                       'calendar_script_sha256':sha(calendar_script)}}
            _write(run/'registration.json',{'identity':ident})
            _write(run/'calendar.json',batch.make_calendar())
            for name,path in acquisition.CODE_PATHS.items():
                dest=run/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest)
            for start,_ in batch.MONTHS:
                month=start[:7]
                if month != '2025-09':
                    recovery._inherit(source/'parts/TEST'/month,run/'parts/TEST'/month,
                                      'TEST',month,run)
            bundle,frame,evidence=local.freeze_request(external,cache,script,calendar_script,
                                          'TEST','2025-09',run/'source_bundles')
            self.assertTrue(frame.empty)
            acquisition._local_result(run/'parts/TEST/2025-09','TEST','2025-09',
                                      bundle,frame,evidence,batch.make_calendar())
            _,coverage,empty,gaps=validate_acquisition(run,['TEST'])
            self.assertEqual(len(coverage),5)
            self.assertEqual(empty,[{'symbol':'TEST','month':'2025-09',
                'data_state':'provider_returned_no_bars_unknown_listing_state'}])
            self.assertFalse(coverage[1]['rth_complete'])
            self.assertEqual(len(gaps),5)


if __name__=='__main__':unittest.main()
