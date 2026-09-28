"""R03 local-raw reuse version; explicit execution only after owner handoff.

Importing or offline auditing this module never opens R2/OpenD or the shared
acquisition lock. The execute command is intentionally separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import futu as ft
import pandas as pd

from scripts.r2_client import R2Client
from . import acquire_batch as batch
from . import acquire_recovery as recovery
from . import local_reuse_source as local
from .evaluate import HERE, ROOT, sha


VERSION = local.VERSION
DEFAULT_SOURCE = HERE.parent/'runs/precision_v9_20260927/R03'
DEFAULT_OUTPUT = HERE.parent/'runs/precision_v9_20260927/R03_recovery_local_v1'
EXTERNAL = ROOT/local.EXTERNAL_ROOT
CACHE = ROOT/local.EXTERNAL_CACHE
EXTERNAL_SCRIPT = ROOT/'scripts/backfill_model_history.py'
CALENDAR_SCRIPT = ROOT/'scripts/model_history_calendar.py'
CODE_PATHS = {
    **{name:HERE/name for name in ('acquire_batch.py','acquire_pilot.py',
        'acquire_recovery.py','local_reuse_source.py','acquire_local_reuse.py',
        'prepare.py','evaluate.py','data.py')},
    **{f'scripts/{name}':ROOT/'scripts'/name for name in
        ('fetch_research_data.py','r2_client.py')},
}


def _identity(source: Path, external: Path, cache: Path, script: Path, calendar_script: Path) -> dict:
    source = source.resolve(strict=True)
    original = recovery._source_registration(source)
    config = original['config']
    local_id = local.source_identity(external,cache,config['symbols'],script,calendar_script)
    return {'version':VERSION,'source_run':str(source),
            'source_registration_sha256':sha(source/'registration.json'),
            'source_registration':original,
            'external_identity':local_id,
            'session':'ALL','price_basis':'NONE','no_upload':True,
            'code_sha256':{name:sha(path) for name,path in CODE_PATHS.items()},
            'source_scope':'original_305_first_candidate_only_108_plus_qqq_aug_dec_2025'}


def initialize(source: Path, output: Path, external: Path = EXTERNAL,
               cache: Path = CACHE, script: Path = EXTERNAL_SCRIPT,
               calendar_script: Path = CALENDAR_SCRIPT) -> dict:
    """Write a new versioned identity; never mutate original/recovery runs."""
    identity = _identity(source,external,cache,script,calendar_script)
    if output.exists():
        reg = output/'registration.json'
        if not reg.is_file() or json.loads(reg.read_text()).get('identity') != identity:
            raise FileExistsError(f'nonempty/changed local-reuse run: {output}')
    else:
        output.mkdir(parents=True)
        recovery._write(output/'registration.json',{'created_at':datetime.now(timezone.utc).isoformat(),
                                                     'identity':identity})
        recovery._write(output/'calendar.json',batch.make_calendar())
        snapshot = output/'source_snapshot';snapshot.mkdir()
        for name in identity['code_sha256']:
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(CODE_PATHS[name],target)
    if json.loads((output/'calendar.json').read_text()) != batch.make_calendar():
        raise ValueError('local-reuse calendar changed')
    for name,expected in identity['code_sha256'].items():
        if sha(output/'source_snapshot'/name) != expected or sha(CODE_PATHS[name]) != expected:
            raise ValueError(f'local-reuse source snapshot changed: {name}')
    return identity


def audit_inventory(source: Path = DEFAULT_SOURCE, external: Path = EXTERNAL,
                    cache: Path = CACHE, script: Path = EXTERNAL_SCRIPT,
                    calendar_script: Path = CALENDAR_SCRIPT) -> dict:
    """No locks, bundles, downloads, or writes; an inventory is not acceptance."""
    identity = _identity(source,external,cache,script,calendar_script)
    counts = Counter()
    for symbol in identity['source_registration']['config']['symbols']:
        for start,_ in batch.MONTHS:
            month = start[:7]
            original = source/'parts'/symbol/month
            if (original/'result.json').is_file():
                meta = json.loads((original/'result.json').read_text())
                if meta.get('pagination_complete') is True and meta.get('failure') is None:
                    recovery.verified_complete(original,symbol,month)
                    counts['original_complete'] += 1
                    continue
            status,_ = local.locate(external,cache,symbol,month)
            counts[status] += 1
    return {'version':VERSION,'scope_parts':109*len(batch.MONTHS),
            'counts':dict(counts),'source_registration_sha256':identity['source_registration_sha256'],
            'external_registration_sha256':identity['external_identity']['registration_sha256'],
            'no_provider_request':True}


def _same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right): return False
    try:
        pd.testing.assert_frame_equal(left.sort_values('start_at').reset_index(drop=True),
                                      right.sort_values('start_at').reset_index(drop=True),
                                      check_dtype=False,check_exact=True)
        return True
    except AssertionError:
        return False


def _legacy_local(symbol: str, start: str, end: str, cal: dict) -> tuple[pd.DataFrame, list[dict]] | None:
    paths = [ROOT/'market_data/us_5m'/symbol/'2025.parquet',
             HERE.parent/'runs/precision_v9_20260927/R02_normalized/market_data/us_5m'/symbol/'2025.parquet']
    accepted = []
    for path in paths:
        if path.is_file():
            frame = pd.read_parquet(path)
            if batch.quality(frame,start,end,cal)['rth_complete']:
                accepted.append((frame.loc[frame.session_date.between(start,end)].copy(),
                                 {'source':str(path),'sha256':sha(path)}))
    if not accepted: return None
    if any(not _same_frame(accepted[0][0],frame) for frame,_ in accepted[1:]):
        raise local.SourceConflict(f'contradictory legacy local sources: {symbol}/{start[:7]}')
    return accepted[0][0],[receipt for _,receipt in accepted]


def _local_result(part: Path, symbol: str, month: str, bundle: Path,
                  frame: pd.DataFrame, evidence: dict, cal: dict) -> dict:
    part.mkdir(parents=True)
    frame.to_parquet(part/'bars.parquet',index=False,compression='zstd',compression_level=7)
    audit = [{'tier':'local_verified_provider','bundle':str(bundle.resolve()),
              'manifest_sha256':sha(bundle/'manifest.json'),
              'provider_request_received_at':evidence['request_received_at'],
              'imported_at':datetime.now(timezone.utc).isoformat(),
              'parameter_evidence':evidence['parameter_evidence']}]
    result = {'symbol':symbol,'month':month,'tier':'local_verified_provider',
              'pagination_complete':True,'failure':None,'quality':batch.quality(
                  frame,*recovery._month_bounds(month),cal),
              'bars_sha256':sha(part/'bars.parquet'),
              'source_bundle':str(bundle.relative_to(part.parents[2])),
              'source_bundle_manifest_sha256':sha(bundle/'manifest.json'),
              'provider_request':{k:evidence[k] for k in ('request_start','request_end',
                  'request_pages','request_raw_rows','request_received_at',
                  'request_covers_month','parameter_evidence')},
              'data_state':evidence['data_state'],
              'empty_success':evidence['empty_success']}
    recovery._write(part/'audit.json',audit)
    recovery._write(part/'result.json',result)
    return result


def verify_committed_local(part: Path, acquisition: Path, symbol: str, month: str) -> tuple[dict,list[dict]]:
    """Read-only gate used again by prepare, including every bundle input."""
    meta = json.loads((part/'result.json').read_text())
    bundle_rel = Path(meta.get('source_bundle',''))
    if bundle_rel.is_absolute() or '..' in bundle_rel.parts:
        raise local.SourceConflict('source bundle path escapes acquisition')
    bundle = acquisition/bundle_rel
    if not bundle.resolve(strict=True).is_relative_to((acquisition/'source_bundles').resolve(strict=True)):
        raise local.SourceConflict('source bundle outside acquisition')
    if sha(bundle/'manifest.json') != meta.get('source_bundle_manifest_sha256'):
        raise local.SourceConflict('source bundle manifest changed')
    identity=json.loads((acquisition/'registration.json').read_text())['identity']
    external=identity.get('external_identity',{})
    for path_name,key in (
            ('external/registration.json','registration_sha256'),
            ('external/calendar.json','calendar_sha256'),
            ('source_code/backfill_model_history.py','script_sha256'),
            ('source_code/model_history_calendar.py','calendar_script_sha256')):
        if sha(bundle/path_name) != external.get(key):
            raise local.SourceConflict(f'bundle source identity changed: {path_name}')
    frame,evidence = local.verify_bundle(bundle,symbol,month)
    if (meta.get('symbol'),meta.get('month'),meta.get('tier')) != (symbol,month,'local_verified_provider'):
        raise local.SourceConflict('local result identity mismatch')
    request_keys = ('request_start','request_end','request_pages','request_raw_rows',
                    'request_received_at','request_covers_month','parameter_evidence')
    if (meta.get('pagination_complete') is not True or meta.get('failure') is not None or
            meta.get('empty_success') is not evidence['empty_success'] or
            meta.get('data_state') != evidence['data_state'] or
            meta.get('provider_request') != {k:evidence[k] for k in request_keys}):
        raise local.SourceConflict('local result provider state changed')
    audit=json.loads((part/'audit.json').read_text())
    if (not isinstance(audit,list) or len(audit) != 1 or
            audit[0].get('tier') != 'local_verified_provider' or
            audit[0].get('bundle') != str(bundle.resolve()) or
            audit[0].get('manifest_sha256') != sha(bundle/'manifest.json') or
            audit[0].get('provider_request_received_at') != evidence['request_received_at'] or
            audit[0].get('parameter_evidence') != evidence['parameter_evidence']):
        raise local.SourceConflict('local import receipt changed')
    try:
        imported=pd.Timestamp(audit[0]['imported_at'])
        received=pd.Timestamp(evidence['request_received_at'])
        if pd.isna(imported) or imported.tzinfo is None or imported < received:
            raise ValueError('missing timezone')
    except (KeyError,ValueError,TypeError) as exc:
        raise local.SourceConflict('local import receipt time invalid') from exc
    stored = pd.read_parquet(part/'bars.parquet')
    if not _same_frame(stored,frame) or sha(part/'bars.parquet') != meta.get('bars_sha256'):
        raise local.SourceConflict('local result bars differ from frozen raw replay')
    if meta.get('quality') != evidence['quality']:
        raise local.SourceConflict('local result quality changed')
    provenance = [{'role':'local_source_bundle_manifest','source':str(bundle/'manifest.json'),
                   'sha256':sha(bundle/'manifest.json')}]
    for item in json.loads((bundle/'manifest.json').read_text())['files']:
        path=bundle/item['bundle_path']
        provenance.append({'role':'local_source_bundle_input','source':str(path),
                           'sha256':item['sha256']})
    return meta,provenance


def _progress(output: Path, results: list[dict], total: int, *, checkpoint=False) -> dict:
    complete = sum(r.get('pagination_complete') is True and r.get('failure') is None for r in results)
    states = Counter(r.get('data_state',r.get('tier','unknown')) for r in results)
    unresolved = sum(r.get('pagination_complete') is not True or r.get('failure') is not None for r in results)
    pending = total-len(results)
    status = ('runtime_checkpoint' if checkpoint else
              'acquisition_complete' if pending == 0 and unresolved == 0 else 'incomplete_unresolved')
    report = {'status':status,'total_parts':total,'complete_parts':complete,
              'unresolved_parts':unresolved,'pending_parts':pending,
              'states':dict(states),'results':results,'no_training':True,'no_upload':True}
    recovery._write(output/'progress.json',report)
    if not checkpoint and pending == 0:
        recovery._write(output/'summary.json',report)
    return report


def run(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT,
        max_minutes: int = 180, *, external: Path = EXTERNAL, cache: Path = CACHE,
        script: Path = EXTERNAL_SCRIPT, calendar_script: Path = CALENDAR_SCRIPT) -> dict:
    """Single-worker acquisition. Do not invoke while the root waiter is active."""
    source=Path(source).resolve(strict=True);output=Path(output).resolve()
    identity=initialize(source,output,external,cache,script,calendar_script)
    cal=batch.make_calendar();symbols=identity['source_registration']['config']['symbols']
    total=len(symbols)*len(batch.MONTHS);results=[];network_started=None
    quote=None;checked=set();remote={}
    lock=open(recovery.LOCK,'a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def context():
        nonlocal quote
        if quote is None:
            ft.SysConfig.enable_proto_encrypt(False)
            quote=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
            ret,quota=quote.get_history_kl_quota(get_detail=False)
            if ret == ft.RET_OK: recovery._write(output/'quota_before.json',
                                                   {'used':int(quota[0]),'remaining':int(quota[1])})
        return quote
    try:
        for symbol in symbols:
            for start,end in batch.MONTHS:
                month=start[:7]
                part=output/'parts'/symbol/month
                source_part=source/'parts'/symbol/month
                original_complete=False
                original_result=source_part/'result.json'
                if original_result.is_file():
                    original_meta=json.loads(original_result.read_text())
                    if original_meta.get('pagination_complete') is True and original_meta.get('failure') is None:
                        recovery.verified_complete(source_part,symbol,month)
                        original_complete=True
                if original_complete:
                    if part.exists() and not part.is_symlink():
                        raise ValueError(f'original complete part conflicts with local result: {part}')
                    results.append(recovery._inherit(source_part,part,symbol,month,output))
                    _progress(output,results,total,checkpoint=True)
                    continue
                if part.is_symlink():
                    raise ValueError(f'unverified original part inherited: {part}')
                if (part/'result.json').is_file():
                    prior=json.loads((part/'result.json').read_text())
                    if prior.get('pagination_complete') is True and prior.get('failure') is None:
                        if prior.get('tier') == 'local_verified_provider':
                            verify_committed_local(part,output,symbol,month)
                        else: recovery.verified_complete(part,symbol,month)
                        results.append(prior);continue
                    raise ValueError(f'unresolved old local-reuse part needs new run: {part}')
                local_start=time.monotonic()
                try:
                    found=local.freeze_request(external,cache,script,calendar_script,
                                                symbol,month,output/'source_bundles')
                    legacy=_legacy_local(symbol,start,end,cal)
                    if found is not None:
                        bundle,frame,evidence=found
                        if legacy is not None and not _same_frame(frame,legacy[0]):
                            raise local.SourceConflict(f'contradictory local sources: {symbol}/{month}')
                        result=_local_result(part,symbol,month,bundle,frame,evidence,cal)
                        result['local_validation_seconds']=round(time.monotonic()-local_start,3)
                        recovery._write(part/'result.json',result)
                        results.append(result)
                        _progress(output,results,total,checkpoint=True)
                        continue
                    if legacy is not None:
                        part.mkdir(parents=True)
                        audit=[{'tier':'legacy_local','sources':legacy[1]}]
                        result=recovery._new_result(part,symbol,month,'local',legacy[0],True,
                                                    None,audit,cal,time.monotonic()-local_start)
                        results.append(result);_progress(output,results,total,checkpoint=True)
                        continue
                except (local.PendingSource,local.SourceConflict) as exc:
                    part.mkdir(parents=True,exist_ok=True)
                    result=recovery._new_result(part,symbol,month,'local_reuse_unresolved',None,
                         False,f'{type(exc).__name__}: {str(exc)[:220]}',
                         [{'stage':'local_source_audit','error':str(exc)[:220]}],cal,
                         time.monotonic()-local_start)
                    results.append(result)
                    _progress(output,results,total,checkpoint=True)
                    return _progress(output,results,total)
                if network_started is None: network_started=time.monotonic()
                if time.monotonic()-network_started > max_minutes*60:
                    return _progress(output,results,total,checkpoint=True)
                part.mkdir(parents=True)
                timer=time.monotonic();audit=[];frame=None;tier=None
                try:
                    if symbol not in checked:
                        key=f'us_5m/{symbol}/2025.parquet';target=output/'r2_cache'/key
                        try:
                            found=R2Client().head_object(key)
                            audit.append({'tier':'r2','key':key,'found':found is not None})
                            if found:
                                target.parent.mkdir(parents=True,exist_ok=True)
                                R2Client().get_object(key,target)
                                remote[symbol]=target
                        except Exception as exc:
                            audit.append({'tier':'r2','error_type':type(exc).__name__})
                        checked.add(symbol)
                    if symbol in remote:
                        candidate=pd.read_parquet(remote[symbol])
                        if batch.quality(candidate,start,end,cal)['rth_complete']:
                            frame=candidate.loc[candidate.session_date.between(start,end)].copy()
                            tier='r2';audit.append({'tier':'r2','source':str(remote[symbol]),
                                                    'sha256':sha(remote[symbol])})
                    complete,failure=True,None
                    if frame is None:
                        tier='futu'
                        pages,events,complete,failure=recovery._fetch_pages(part,symbol,start,end,context)
                        audit.extend(events)
                        if pages:
                            raw=pd.concat(pages,ignore_index=True)
                            if not raw.empty and ('time_key' not in raw or raw.time_key.duplicated().any()):
                                complete,failure=False,'duplicate_or_missing_source_times'
                            frame=recovery.normalize(raw,symbol)
                    result=recovery._new_result(part,symbol,month,tier,frame,complete,failure,
                                                 audit,cal,time.monotonic()-timer)
                except Exception as exc:
                    result=recovery._new_result(part,symbol,month,tier or 'unknown',None,False,
                        f'exception: {type(exc).__name__}',
                        audit+[{'error_type':type(exc).__name__,'message':str(exc)[:220]}],
                        cal,time.monotonic()-timer)
                results.append(result)
                _progress(output,results,total,checkpoint=True)
                if result.get('pagination_complete') is not True or result.get('failure') is not None:
                    return _progress(output,results,total)
        return _progress(output,results,total)
    finally:
        if quote is not None:
            try:
                ret,quota=quote.get_history_kl_quota(get_detail=False)
                if ret == ft.RET_OK: recovery._write(output/'quota_after.json',
                                                       {'used':int(quota[0]),'remaining':int(quota[1])})
            finally: quote.close()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('offline-audit','execute'))
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--max-minutes',type=int,default=180)
    args=parser.parse_args()
    result=(audit_inventory(args.source) if args.command=='offline-audit' else
            run(args.source,args.output,args.max_minutes))
    print(json.dumps({k:v for k,v in result.items() if k!='results'},sort_keys=True),flush=True)


if __name__=='__main__': main()
