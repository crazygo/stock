"""Offline, fail-closed verification of external 2025 provider requests.

Only this module reads the independent history store. A committed bundle is a
copy of source bytes, never a symlink into a store that another task may edit.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from . import acquire_batch as batch
from . import acquire_recovery as recovery
from .evaluate import sha


VERSION = 'r03_acquisition_local_reuse_v1'
EXTERNAL_ROOT = Path('market_data/model_training_history_v1')
EXTERNAL_CACHE = Path('.cache/model_history_20260927')
EXPECTED_EXTERNAL_SCRIPT_SHA = '4f194799c28e7fd1ba4f8cb385ac72b2caa8902c9c25438f927251799a98f5a3'
EXPECTED_CALENDAR_SCRIPT_SHA = '9e4ff2049983c02a464be3c2204bda7e6439eccd0175b05fab007cf330abc1ba'


class PendingSource(ValueError):
    """A completion marker precedes the provider request commit."""


class SourceConflict(ValueError):
    """A present source contradicts its claimed provenance or raw data."""


def _json(path: Path):
    return json.loads(path.read_text())


def _inside(path: Path, root: Path) -> Path:
    result = path.resolve(strict=True)
    if not result.is_relative_to(root.resolve(strict=True)):
        raise SourceConflict(f'source outside registered root: {path}')
    return result


def source_identity(external: Path, cache: Path, expected_symbols: list[str],
                    script: Path, calendar_script: Path) -> dict:
    external = external.resolve(strict=True)
    cache = cache.resolve(strict=True)
    reg = _json(external / 'registration.json')
    if (reg.get('symbols') != expected_symbols or len(set(expected_symbols)) != 109 or
            reg.get('interval') != '5m' or reg.get('session') != 'ALL' or
            reg.get('price_basis') != 'NONE' or reg.get('no_automatic_cloud_writes') is not True or
            reg.get('script_sha256') != sha(script) or
            sha(script) != EXPECTED_EXTERNAL_SCRIPT_SHA or
            sha(calendar_script) != EXPECTED_CALENDAR_SCRIPT_SHA):
        raise SourceConflict('external registration/code does not match frozen R03 scope')
    cal = _json(external / 'calendar.json')
    own = {s['session_date']:s for s in cal['sessions'] if '2025-08-01' <= s['session_date'] <= '2025-12-31'}
    expected = {s['session_date']:s for s in batch.make_calendar()['sessions']
                if '2025-08-01' <= s['session_date'] <= '2025-12-31'}
    if set(own) != set(expected) or any(
            (own[d]['open_at'], own[d]['close_at'], own[d]['duration_minutes']) !=
            (expected[d]['open_at'], expected[d]['close_at'], expected[d]['duration_minutes'])
            for d in expected):
        raise SourceConflict('external calendar Aug-Dec session projection differs')
    return {'registration_sha256':sha(external/'registration.json'),
            'calendar_sha256':sha(external/'calendar.json'),
            'script_sha256':sha(script),'calendar_script_sha256':sha(calendar_script),
            'external_root':str(external),'cache_root':str(cache),
            'candidate_symbols':expected_symbols}


def locate(external: Path, cache: Path, symbol: str, month: str) -> tuple[str, Path | None]:
    """Return missing/metadata_only/pending/request without interpreting bars."""
    marker = external / 'parts' / symbol / month / 'complete.json'
    if not marker.is_file():
        return 'missing', None
    complete = _json(marker)
    if complete.get('pagination_complete') is not True:
        raise PendingSource(f'local complete marker not committed: {symbol}/{month}')
    if complete.get('status') == 'before_current_regular_trading':
        return 'metadata_only', None
    if complete.get('status') not in ('source_returned','source_empty'):
        raise SourceConflict(f'unsupported local complete status: {symbol}/{month}')
    raw_source = complete.get('source')
    if not isinstance(raw_source, str):
        raise PendingSource(f'local request source missing: {symbol}/{month}')
    try:
        request = _inside(Path(raw_source), cache)
    except FileNotFoundError as exc:
        raise PendingSource(f'local request directory missing: {symbol}/{month}') from exc
    if not (request/'result.json').is_file():
        raise PendingSource(f'local request result not committed: {symbol}/{month}')
    return 'request', request


def _candidate_files(external: Path, request: Path, symbol: str, month: str,
                     script: Path, calendar_script: Path, cache: Path) -> tuple[dict[str, Path],list[str]]:
    files = {
        'external/registration.json':external/'registration.json',
        'external/calendar.json':external/'calendar.json',
        'request/result.json':request/'result.json',
        'request/audit.json':request/'audit.json',
        'source_code/backfill_model_history.py':script,
        'source_code/model_history_calendar.py':calendar_script,
    }
    for optional, path in (
            ('external/security_metadata.json',external/'security_metadata.json'),
            ('request/normalized.parquet',request/'normalized.parquet')):
        if path.is_file(): files[optional] = path
    for path in sorted(request.glob('raw_*.parquet')):
        files[f'request/{path.name}'] = path
    result = _json(request/'result.json')
    months = []
    for start,end in batch.MONTHS:
        other = start[:7]
        if not (result['start'] <= start and result['end'] >= end):
            continue
        part = external/'parts'/symbol/other
        marker = part/'complete.json'
        if not marker.is_file():
            continue
        doc = _json(marker)
        if doc.get('source') != str(request) or doc.get('status') not in ('source_returned','source_empty'):
            continue
        months.append(other)
        files[f'external/months/{other}/complete.json'] = marker
        for name in ('imports.json','bars.parquet'):
            if (part/name).is_file(): files[f'external/months/{other}/{name}'] = part/name
        revision = cache/'source_revisions'/symbol/other
        if revision.is_dir():
            for path in sorted(revision.rglob('*')):
                if path.is_file(): files[f'revision/{other}/'+str(path.relative_to(revision))] = path
    if month not in months:
        raise SourceConflict('request group does not cover marked month')
    return files,months


def _replay_request(request: Path, symbol: str, month: str) -> tuple[pd.DataFrame, dict]:
    result = _json(request/'result.json')
    start, end = result.get('start'), result.get('end')
    month_start, month_end = recovery._month_bounds(month)
    if (result.get('symbol') != symbol or result.get('status') != 'fetched' or
            not isinstance(start,str) or not isinstance(end,str) or
            start > month_start or end < month_end or end > '2025-12-31'):
        raise SourceConflict(f'provider request identity/range invalid: {symbol}/{month}')
    audit = _json(request/'audit.json')
    if not isinstance(audit,list) or not audit:
        raise PendingSource(f'provider audit absent/empty: {symbol}/{month}')
    successes = {}
    expected_page, expected_attempt = 1, 1
    last_received = None
    for event in audit:
        if (event.get('symbol'),event.get('start'),event.get('end')) != (symbol,start,end):
            raise SourceConflict('provider audit symbol or range mismatch')
        try:
            sent = pd.Timestamp(event['requested_at']); received = pd.Timestamp(event['received_at'])
        except (KeyError,TypeError,ValueError) as exc:
            raise SourceConflict('provider receipt time invalid') from exc
        if (pd.isna(sent) or pd.isna(received) or sent.tzinfo is None or
                received.tzinfo is None or sent > received or
                (last_received is not None and sent < last_received)):
            raise SourceConflict('provider receipt precedes request')
        last_received = received
        page, attempt = event.get('page'),event.get('attempt')
        if (not isinstance(page,int) or page < 1 or page > 160 or
                not isinstance(attempt,int) or attempt < 1 or attempt > 3):
            raise SourceConflict('provider page/attempt out of bounds')
        if (page,attempt) != (expected_page,expected_attempt):
            raise SourceConflict('provider audit page/attempt order invalid')
        if event.get('ok') is True:
            if page in successes or not isinstance(event.get('has_more'),bool):
                raise SourceConflict('duplicate/invalid successful provider page')
            successes[page] = event
            expected_page,expected_attempt = page+1,1
        elif event.get('ok') is False:
            if attempt == 3:
                raise SourceConflict('provider retry exhausted before success')
            expected_attempt += 1
        else:
            raise SourceConflict('provider audit success state invalid')
    if not successes or sorted(successes) != list(range(1,len(successes)+1)):
        raise SourceConflict('provider page sequence incomplete')
    if audit[-1].get('ok') is not True:
        raise SourceConflict('provider audit ends without successful page')
    if any(successes[n]['has_more'] is not (n < len(successes)) for n in successes):
        raise SourceConflict('provider pagination has_more chain incomplete')
    names = {f'raw_{page:03d}.parquet' for page in successes}
    if {p.name for p in request.glob('raw_*.parquet')} != names:
        raise SourceConflict('provider raw page inventory differs from audit')
    raw = []
    for page, event in sorted(successes.items()):
        path = request/f'raw_{page:03d}.parquet'
        if sha(path) != event.get('sha256'):
            raise SourceConflict(f'provider raw page hash changed: {path.name}')
        frame = pd.read_parquet(path)
        if len(frame) != event.get('rows'):
            raise SourceConflict(f'provider raw page row count changed: {path.name}')
        raw.append(frame)
    source = pd.concat(raw,ignore_index=True)
    if len(source):
        if (not {'code','time_key'} <= set(source.columns) or
                source.code.isna().any() or not source.code.eq('US.'+symbol).all()):
            raise SourceConflict('provider raw symbol changed')
        times = source.time_key.astype('string')
        if not times.str.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}').fillna(False).all():
            raise SourceConflict('provider time_key is not naive exchange time')
        keys = pd.to_datetime(times,format='%Y-%m-%d %H:%M:%S',errors='coerce')
        if (keys.isna().any() or source.time_key.duplicated().any() or
                keys.dt.second.ne(0).any() or keys.dt.minute.mod(5).ne(0).any()):
            raise SourceConflict('provider time_key invalid or duplicated')
        if not keys.dt.strftime('%Y-%m-%d').between(start,end).all():
            raise SourceConflict('provider raw time outside request')
    normalized = recovery.normalize(source,symbol)
    if list(normalized.columns) != list(recovery.CANONICAL_COLUMNS):
        raise SourceConflict('R03 normalizer canonical schema changed')
    saved_normalized = request/'normalized.parquet'
    if saved_normalized.is_file():
        saved = pd.read_parquet(saved_normalized)
        # The independent request may start before Aug-2025. Its calendar can
        # classify earlier half-days differently; only imported R03 months are
        # eligible for equality, while every raw page still needs verification.
        saved = saved.loc[saved.session_date.str[:7] == month]
        comparable = normalized.loc[normalized.session_date.str[:7] == month]
        if list(saved.columns) != list(comparable.columns) or len(saved) != len(comparable):
            raise SourceConflict('external month normalization differs from R03 raw replay')
        try:
            pd.testing.assert_frame_equal(saved.sort_values('start_at').reset_index(drop=True),
                                           comparable.sort_values('start_at').reset_index(drop=True),
                                           check_dtype=False,check_exact=True)
        except AssertionError as exc:
            raise SourceConflict('external month normalization differs from R03 raw replay') from exc
    if len(normalized) and (set(normalized.symbol.dropna()) != {symbol} or
                            set(normalized.price_basis.dropna()) != {'NONE'}):
        raise SourceConflict('R03 normalized symbol/basis changed')
    month_frame = normalized.loc[normalized.session_date.str[:7] == month].copy()
    if month_frame.empty: month_frame = recovery.canonical_empty()
    return month_frame, {'request_start':start,'request_end':end,
                         'request_pages':len(successes),'request_raw_rows':len(source),
                         'request_received_at':successes[len(successes)]['received_at'],
                         'request_covers_month':True,
                         'parameter_evidence':'registration_and_frozen_source_code_not_provider_page_echo',
                         'pagination_complete':True}


def _assert_external_month(bundle: Path, frame: pd.DataFrame, month: str) -> None:
    month_root = bundle/'external/months'/month
    complete = _json(month_root/'complete.json')
    if complete.get('pagination_complete') is not True or complete.get('status') not in ('source_returned','source_empty'):
        raise SourceConflict('external complete marker is not provider success')
    external_bars = month_root/'bars.parquet'
    imports = month_root/'imports.json'
    if frame.empty:
        if external_bars.is_file() and not pd.read_parquet(external_bars).empty:
            raise SourceConflict('provider empty conflicts with external stored bars')
        return
    if not external_bars.is_file() or not imports.is_file():
        raise SourceConflict('external nonempty bars/imports missing')
    lineage = _json(imports)
    if not isinstance(lineage,list) or len(lineage) != 1:
        raise SourceConflict('merged/revised external bars require separate source resolution')
    request_source = bundle/'request/normalized.parquet'
    manifest = _json(bundle/'manifest.json')
    frozen_origin = next((item['origin'] for item in manifest['files']
                          if item['bundle_path'] == 'request/normalized.parquet'),None)
    if (not request_source.is_file() or
            lineage[0].get('sha256') != sha(request_source) or
            not isinstance(lineage[0].get('source'),str) or
            frozen_origin is None or
            Path(lineage[0]['source']).resolve() != Path(frozen_origin).resolve()):
        raise SourceConflict('external imports do not identify the frozen request')
    saved = pd.read_parquet(external_bars)
    if len(saved) != len(frame):
        raise SourceConflict('external monthly bars rows differ from raw replay')
    left = saved.sort_values('start_at').reset_index(drop=True)
    right = frame.sort_values('start_at').reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left,right,check_dtype=False,check_exact=True)
    except AssertionError as exc:
        raise SourceConflict(f'external monthly bars differ from raw replay: {str(exc)[:180]}') from exc


def verify_bundle(bundle: Path, symbol: str, month: str) -> tuple[pd.DataFrame, dict]:
    """Revalidate every copied byte and replay only the frozen request raw pages."""
    manifest_path = bundle/'manifest.json'
    manifest = _json(manifest_path)
    if (manifest.get('version') != VERSION or manifest.get('symbol') != symbol or
            month not in manifest.get('months',[]) or not isinstance(manifest.get('files'),list)):
        raise SourceConflict('source bundle manifest identity mismatch')
    for item in manifest['files']:
        name = item['bundle_path']
        if name.startswith('/') or '..' in Path(name).parts or sha(bundle/name) != item['sha256']:
            raise SourceConflict(f'source bundle file changed: {name}')
    reg = _json(bundle/'external/registration.json')
    if (reg.get('interval'),reg.get('session'),reg.get('price_basis'),reg.get('script_sha256')) != \
            ('5m','ALL','NONE',sha(bundle/'source_code/backfill_model_history.py')):
        raise SourceConflict('source bundle registration/code mismatch')
    complete = _json(bundle/'external/months'/month/'complete.json')
    result_origin = next((item['origin'] for item in manifest['files']
                          if item['bundle_path'] == 'request/result.json'),None)
    if (not isinstance(complete.get('source'),str) or result_origin is None or
            Path(complete['source']).resolve() != Path(result_origin).resolve().parent or
            Path(complete['source']).name != manifest.get('request_name')):
        raise SourceConflict('complete marker/request identity mismatch')
    frame, evidence = _replay_request(bundle/'request',symbol,month)
    if complete.get('request_received_at') != evidence['request_received_at']:
        raise SourceConflict('complete marker/provider receipt mismatch')
    _assert_external_month(bundle,frame,month)
    start,end = recovery._month_bounds(month)
    q = batch.quality(frame,start,end,batch.make_calendar())
    if q['invalid_ohlc'] or q['duplicates'] or (not frame.empty and q['basis'] != ['NONE']):
        raise SourceConflict('raw replay has invalid OHLC, duplicate bars, or wrong basis')
    evidence['quality'] = q
    evidence['data_state'] = ('provider_returned_no_bars_unknown_listing_state' if frame.empty
                              else 'observed_bars' if q['rth_complete'] else 'observed_partial_coverage')
    evidence['empty_success'] = bool(frame.empty)
    evidence['bundle_manifest_sha256'] = sha(manifest_path)
    return frame,evidence


def freeze_request(external: Path, cache: Path, script: Path, calendar_script: Path,
                   symbol: str, month: str, bundle_root: Path,
                   *, copy_hook=None) -> tuple[Path,pd.DataFrame,dict] | None:
    status, request = locate(external,cache,symbol,month)
    if status in ('missing','metadata_only'):
        return None
    assert request is not None
    request_name = request.name
    request_key = hashlib.sha256(str(request).encode()).hexdigest()[:20]
    bundle = bundle_root/symbol/request_key
    if bundle.exists():
        frame,evidence = verify_bundle(bundle,symbol,month)
        return bundle,frame,evidence
    files,months = _candidate_files(external,request,symbol,month,script,calendar_script,cache)
    temp = bundle.with_name(bundle.name+'.staging')
    if temp.exists():
        raise SourceConflict(f'leftover uncommitted bundle: {temp}')
    temp.mkdir(parents=True)
    try:
        before = {name:sha(source) for name,source in files.items()}
        for name,source in files.items():
            dest = temp/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
            if copy_hook is not None: copy_hook(source,dest)
            if sha(dest) != before[name]:
                raise SourceConflict(f'copied source changed: {name}')
        after = {name:sha(source) for name,source in files.items()}
        if before != after:
            raise SourceConflict('local source changed during freeze')
        manifest = {'version':VERSION,'symbol':symbol,'months':months,
                    'request_name':request_name,
                    'frozen_at':datetime.now(timezone.utc).isoformat(),
                    'files':[{'bundle_path':name,'origin':str(files[name].resolve()),
                              'sha256':before[name]} for name in sorted(files)]}
        (temp/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
        # Validate every monthly projection before one immutable request commit.
        validated = {name:verify_bundle(temp,symbol,name) for name in months}
        frame,evidence = validated[month]
        bundle.parent.mkdir(parents=True,exist_ok=True)
        temp.rename(bundle)
        return bundle,frame,evidence
    except Exception as exc:
        # A failed freeze is evidence, never a source of accepted bars. Keep
        # the staging bytes for review and force a separately registered retry.
        after_failure = {}
        for name,source in files.items():
            try: after_failure[name] = sha(source)
            except (OSError,ValueError): after_failure[name] = None
        failure = {'status':'freeze_failed_uncommitted','version':VERSION,
                   'symbol':symbol,'months':months,'request_name':request_name,
                   'failed_at':datetime.now(timezone.utc).isoformat(),
                   'error_type':type(exc).__name__,'error':str(exc)[:500],
                   'source_before_sha256':locals().get('before',{}),
                   'source_after_sha256':after_failure}
        (temp/'failure.json').write_text(json.dumps(failure,sort_keys=True,indent=2)+'\n')
        raise
