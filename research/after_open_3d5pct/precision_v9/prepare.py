"""Assemble R03 inputs and rebuild causal groups before fitting any new model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

from . import data as vd
from . import acquire_batch as batch
from . import acquire_recovery as recovery
from . import acquire_local_reuse as local_reuse
from .evaluate import HERE, ROOT, sha, write, GROUPS


def _discard(stream, count):
    while count:
        chunk = stream.read(min(count, 1 << 20))
        if not chunk:
            raise ValueError('truncated feature archive')
        count -= len(chunk)


def _array_stream(archive, name):
    stream = archive.open(name + '.npy')
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in ((2, 0), (3, 0)):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f'unsupported feature format: {version}')
    if fortran or dtype.hasobject or not shape:
        raise ValueError(f'unsupported feature array: {name}')
    return stream, shape, dtype


def _compare_array(old_file, new_file, name, pairs):
    """Compare matched rows in compressed NPZs without loading full tensors."""
    with zipfile.ZipFile(old_file) as old_zip, zipfile.ZipFile(new_file) as new_zip:
        a, a_shape, a_dtype = _array_stream(old_zip, name)
        b, b_shape, b_dtype = _array_stream(new_zip, name)
        try:
            if a_shape[1:] != b_shape[1:] or a_dtype != b_dtype:
                raise ValueError(f'{name}: feature shape/dtype changed')
            row_bytes = int(np.prod(a_shape[1:])) * a_dtype.itemsize
            old_next = new_next = changed = 0
            channels = np.zeros(a_shape[-1], np.int64)
            slot_channels = np.zeros(a_shape[-2:], np.int64) if name in ('group', 'group_seq') else None
            examples = []
            for key, old_i, new_i in pairs:
                _discard(a, (old_i-old_next)*row_bytes)
                _discard(b, (new_i-new_next)*row_bytes)
                left = np.frombuffer(a.read(row_bytes), dtype=a_dtype)
                right = np.frombuffer(b.read(row_bytes), dtype=b_dtype)
                if left.size != int(np.prod(a_shape[1:])) or right.size != left.size:
                    raise ValueError(f'{name}: truncated row')
                old_next, new_next = old_i+1, new_i+1
                unequal = ~np.isclose(left, right, rtol=0, atol=0, equal_nan=True)
                if unequal.any():
                    changed += 1
                    channels += unequal.reshape(-1, a_shape[-1]).any(axis=0)
                    if slot_channels is not None:
                        slot_channels += unequal.reshape(-1, *a_shape[-2:]).any(axis=0)
                    if len(examples) < 3:
                        examples.append({'symbol': key[0], 'session_date': key[1]})
            result = {'changed_rows': changed, 'changed_channel_rows': channels.tolist(),
                      'examples': examples, 'shape_per_row': list(a_shape[1:])}
            if slot_channels is not None:
                result['changed_slot_channel_rows'] = slot_channels.tolist()
            return result
        finally:
            a.close()
            b.close()


def audit_2026_pairs(old_dataset: Path, new_dataset: Path, report: Path):
    """Replay old symbol/date rows; retain exact field and tensor differences."""
    old = pd.read_parquet(old_dataset/'rows.parquet')
    new = pd.read_parquet(new_dataset/'rows.parquet')
    keys = ['symbol', 'session_date']
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError('duplicate paired sample key')
    old_index = {(r.symbol, r.session_date): i for i, r in enumerate(old.itertuples())}
    new_index = {(r.symbol, r.session_date): i for i, r in enumerate(new.itertuples())}
    shared = sorted(old_index.keys() & new_index.keys(), key=old_index.get)
    pairs = [(key, old_index[key], new_index[key]) for key in shared]
    if not pairs:
        raise ValueError('no 2026 rows pair with the frozen v8 dataset')
    if any(pairs[i][2] >= pairs[i+1][2] for i in range(len(pairs)-1)):
        raise ValueError('paired rows are not in consistent symbol/date order')
    fields = {}
    for field in old.columns.intersection(new.columns):
        differences = []
        for key, oi, ni in pairs:
            a, b = old.iloc[oi][field], new.iloc[ni][field]
            if (pd.isna(a) and pd.isna(b)) or a == b:
                continue
            if len(differences) < 3:
                differences.append({'symbol': key[0], 'session_date': key[1],
                                    'old': str(a), 'new': str(b)})
        # Count separately so examples stay bounded and complete differences
        # remain auditable even on large shared evaluations.
        left = old.iloc[[old_index[k] for k in shared]][field].reset_index(drop=True)
        right = new.iloc[[new_index[k] for k in shared]][field].reset_index(drop=True)
        changed = int((~(left.eq(right) | (left.isna() & right.isna()))).sum())
        fields[field] = {'changed_rows': changed, 'examples': differences}
    arrays = {name: _compare_array(old_dataset/'features.npz', new_dataset/'features.npz', name, pairs)
              for name in ('x5', 'x60', 'xday', 'group', 'group_seq', 'y')}
    array_channels = {name: list(vd.FEATURE_COLUMNS) for name in ('x5', 'x60', 'xday')}
    peer_columns = ['category_0', 'category_1', 'category_2', 'median_peer_return',
                    'peer_return_iqr', 'positive_peer_fraction', 'median_peer_rvol',
                    'log_peer_count', 'peer_coverage', 'valid']
    qqq_columns = ['benchmark_present', 'unused_1', 'unused_2', 'qqq_cutoff_return',
                   'qqq_cutoff_rvol', 'unused_5', 'unused_6', 'unused_7', 'unused_8', 'valid']
    for name in ('group', 'group_seq'):
        array_channels[name] = {'slot_names': [*vd.STRATEGIES, 'QQQ'],
                                'labels_by_slot': [peer_columns]*5 + [qqq_columns]}
    array_channels['y'] = ['3pct', '5pct', '8pct']
    result = {'paired_rows': len(pairs), 'old_only_rows': len(old_index.keys()-new_index.keys()),
              'old_only_examples': sorted(old_index.keys()-new_index.keys())[:10],
              'new_only_rows': len(new_index.keys()-old_index.keys()),
              'new_only_examples': sorted(new_index.keys()-old_index.keys())[:10],
              'fields': fields, 'arrays': arrays, 'array_channel_labels': array_channels,
              'old_dataset_sha256': {p: sha(old_dataset/p) for p in ('rows.parquet', 'features.npz', 'manifest.json')},
              'new_dataset_sha256': {p: sha(new_dataset/p) for p in ('rows.parquet', 'features.npz', 'manifest.json')}}
    write(report, result)
    protected = ('cutoff_at', 'decision_at', 'entry_at', 'entry_price', 'label_end_at',
                 'label_available_at', 'terminal_1d', 'terminal_3d', 'terminal_5d')
    if any(fields.get(name, {}).get('changed_rows', 0) for name in protected) or arrays['y']['changed_rows']:
        raise ValueError(f'2026 paired labels changed; inspect {report}')
    return result


def make_groups(inputs, config):
    sessions=vd._calendar(config);dates=[s['session_date'] for s in sessions]
    universe=json.loads((inputs/config['universe']).read_text())
    symbols=sorted(x['symbol'] for x in universe['members'] if x['role']=='candidate')
    weeks={}
    for i,d in enumerate(dates):
        iso=pd.Timestamp(d).isocalendar();wk=f'{iso.year}-W{iso.week:02d}'
        weeks.setdefault(wk,i)
    versions=[];members=[]
    for j,(week,ix) in enumerate(weeks.items()):
        first=pd.Timestamp(sessions[ix]['open_at'])
        next_first=(list(weeks.values())[j+1] if j+1<len(weeks) else None)
        until=(pd.Timestamp(sessions[next_first]['open_at']) if next_first is not None else
               (first.tz_convert('America/New_York').normalize()+pd.Timedelta(days=7-first.tz_convert('America/New_York').weekday(),hours=9,minutes=30)).tz_convert('UTC'))
        for strategy in vd.STRATEGIES:
            versions.append({'week_id':week,'strategy_id':strategy,'version_id':f'v9:{strategy}:{week}',
                'effective_from':first.isoformat(),'effective_to':until.isoformat(),
                'feature_cutoff_at':first.isoformat(),'membership_basis':'causal_weekly_reconstruction',
                'first_session':dates[ix]})
    for symbol in symbols:
        view=vd.load_symbol(symbol,sessions,config)
        if view is None:continue
        actions=vd._split_days(symbol)
        for v in versions:
            strategy=v['strategy_id'];cutoff=pd.Timestamp(v['feature_cutoff_at'])
            ids=[i for i,s in enumerate(sessions) if pd.Timestamp(s['close_at'])<cutoff]
            n=int(strategy.split('_')[1]) if strategy.startswith('trend_') else 20
            need=n if strategy=='liquidity' else n+1;ids=ids[-need:]
            if len(ids)!=need or not view.daily_full[ids].all():continue
            if any(dates[ids[0]]<a<=dates[ids[-1]] for a in actions):continue
            available=max(view.effective_available[i,66:66+int(sessions[i]['duration_minutes'])//5].view('int64').max() for i in ids)
            if available>=cutoff.value:continue
            day=view.seqday[ids];closes=np.exp(day[:,12].astype(float)*5+day[:,0]);returns=np.diff(np.log(closes))
            if strategy=='liquidity':
                value=float(np.median([view.raw5[i,66:66+int(sessions[i]['duration_minutes'])//5,5].sum() for i in ids]))
                category=['low','medium','high'][int(value>=50e6)+int(value>=200e6)]
            elif strategy=='volatility':
                value=float(np.std(returns,ddof=1)*np.sqrt(252))
                category=['low','medium','high'][int(value>=.2)+int(value>=.4)]
            else:
                value=float(returns.sum()/max(np.std(returns,ddof=1)*np.sqrt(n),1e-12))
                category='up' if value>1 else 'down' if value< -1 else 'range'
            members.append({'week_id':v['week_id'],'strategy_id':strategy,'symbol':symbol,
                'version_id':v['version_id'],'group_ids':[strategy+':'+category],
                'facts':{'history_end':dates[ids[-1]],'classifier_value':value}})
    dest=inputs/config['group_run']
    write(dest/'versions.json',versions);write(dest/'memberships.json',members)
    write(dest/'groups.json',[{'group_id':f'{s}:{c}','strategy_id':s} for s in vd.STRATEGIES
                              for c in (['up','range','down'] if s.startswith('trend') else ['low','medium','high'])])


def validate_acquisition(acquisition: Path, symbols: list[str], baseline: Path | None = None):
    """Read-only gate for every frozen R03 part, including recovery inheritance."""
    months = [list(m) for m in batch.MONTHS]
    registration_path = acquisition / 'registration.json'
    registration = json.loads(registration_path.read_text())
    identity = registration.get('identity')
    if identity is None:
        if 'version' in registration or 'config' not in registration:
            raise ValueError('unsupported acquisition registration')
        version = 'r03_acquisition_original'
        config = registration['config']
        source = None
    elif isinstance(identity, dict) and identity.get('version') in (recovery.VERSION,local_reuse.VERSION):
        if 'config' in registration:
            raise ValueError('unsupported acquisition recovery version')
        version = identity['version']
        source = Path(identity['source_run']).resolve(strict=True)
        source_registration = source / 'registration.json'
        if sha(source_registration) != identity.get('source_registration_sha256'):
            raise ValueError('changed recovery source registration')
        source_doc = json.loads(source_registration.read_text())
        if ('identity' in source_doc or 'config' not in source_doc or
                source_doc != identity.get('source_registration')):
            raise ValueError('recovery source registration mismatch')
        config = identity['source_registration']['config']
        if (identity.get('no_upload') is not True or identity.get('session') != 'ALL' or
                identity.get('price_basis') != 'NONE'):
            raise ValueError('recovery identity changed frozen acquisition scope')
        if version == local_reuse.VERSION:
            if identity.get('source_scope') != 'original_305_first_candidate_only_108_plus_qqq_aug_dec_2025':
                raise ValueError('local-reuse source scope changed')
            code = identity.get('code_sha256')
            expected_names = set(local_reuse.CODE_PATHS)
            if not isinstance(code,dict) or set(code) != expected_names:
                raise ValueError('local-reuse code identity missing')
            for name, expected in code.items():
                if (sha(acquisition/'source_snapshot'/name) != expected or
                        sha(local_reuse.CODE_PATHS[name]) != expected):
                    raise ValueError(f'local-reuse source code changed: {name}')
            external = identity.get('external_identity')
            if (not isinstance(external,dict) or
                    external.get('candidate_symbols') != config['symbols'] or
                    external.get('script_sha256') != local_reuse.local.EXPECTED_EXTERNAL_SCRIPT_SHA or
                    external.get('calendar_script_sha256') != local_reuse.local.EXPECTED_CALENDAR_SCRIPT_SHA):
                raise ValueError('local-reuse external source identity changed')
    else:
        raise ValueError('unsupported acquisition recovery version')
    if (not isinstance(config.get('symbols'), list) or
            len(config['symbols']) != len(set(config['symbols'])) or
            set(config['symbols']) != set(symbols) or
            config.get('months') != months or config.get('session') != 'ALL' or
            config.get('basis') != 'NONE' or config.get('no_upload') is not True):
        raise ValueError('acquisition does not match frozen R03 scope')
    if baseline is not None and Path(config.get('source', '')).resolve() != baseline.resolve():
        raise ValueError('acquisition uses a different frozen v8 source')
    snapshot = (source or acquisition) / 'source_snapshot'
    for name, key in (('acquire_batch.py', 'code_sha'), ('acquire_pilot.py', 'normalizer_sha')):
        if sha(snapshot / name) != config.get(key):
            raise ValueError(f'acquisition source snapshot changed: {name}')
    calendar_path = acquisition / 'calendar.json'
    calendar = json.loads(calendar_path.read_text())
    if calendar != batch.make_calendar():
        raise ValueError('acquisition calendar changed')
    provenance = [{'role': 'acquisition_registration', 'source': str(registration_path),
                   'sha256': sha(registration_path)},
                  {'role': 'acquisition_calendar', 'source': str(calendar_path),
                   'sha256': sha(calendar_path)}]
    if source is not None:
        provenance.append({'role': 'source_acquisition_registration',
                           'source': str(source / 'registration.json'),
                           'sha256': sha(source / 'registration.json')})
    for name in ('acquire_batch.py', 'acquire_pilot.py'):
        path = snapshot / name
        provenance.append({'role': 'acquisition_source_snapshot', 'source': str(path),
                           'sha256': sha(path)})
    if version == local_reuse.VERSION:
        for name in local_reuse.CODE_PATHS:
            path=acquisition/'source_snapshot'/name
            provenance.append({'role':'local_reuse_source_snapshot','source':str(path),
                               'sha256':sha(path)})
    coverage = []
    empty_parts = []
    unknown_gaps = []
    for symbol in symbols:
        for start, end in batch.MONTHS:
            month = start[:7]
            part = acquisition / 'parts' / symbol / month
            original_complete = False
            if version == local_reuse.VERSION:
                original = source / 'parts' / symbol / month
                original_result = original / 'result.json'
                if original_result.is_file():
                    original_meta = json.loads(original_result.read_text())
                    if original_meta.get('pagination_complete') is True and original_meta.get('failure') is None:
                        recovery.verified_complete(original,symbol,month)
                        original_complete = True
                        if not part.is_symlink():
                            raise ValueError(f'original complete part must be inherited {symbol} {month}')
                if part.is_symlink() and not original_complete:
                    raise ValueError(f'unverified original part inherited {symbol} {month}')
            if not part.exists():
                raise ValueError(f'missing acquisition {symbol} {month}')
            record_path = acquisition / 'inherited' / symbol / f'{month}.json'
            if part.is_symlink():
                if source is None or not record_path.is_file():
                    raise ValueError(f'unregistered inherited part {symbol} {month}')
                source_part = source / 'parts' / symbol / month
                if part.resolve(strict=True) != source_part.resolve(strict=True):
                    raise ValueError(f'inherited link source changed {symbol} {month}')
                record = json.loads(record_path.read_text())
                expected = {'source_part': str(source_part.resolve()),
                            'source_result_sha256': sha(source_part / 'result.json'),
                            'source_audit_sha256': sha(source_part / 'audit.json'),
                            'source_bars_sha256': sha(source_part / 'bars.parquet')}
                if record != expected:
                    raise ValueError(f'inherited source changed {symbol} {month}')
                provenance.append({'role': 'inherited_record', 'source': str(record_path),
                                   'sha256': sha(record_path), 'source_part': str(source_part)})
            elif record_path.exists():
                raise ValueError(f'inherited record without link {symbol} {month}')
            meta_path = part / 'result.json'
            audit_path = part / 'audit.json'
            bars_path = part / 'bars.parquet'
            meta = json.loads(meta_path.read_text())
            if version == local_reuse.VERSION and not part.is_symlink() and meta.get('tier') == 'local_verified_provider':
                _, bundle_provenance = local_reuse.verify_committed_local(part,acquisition,symbol,month)
                provenance.extend(bundle_provenance)
            if (meta.get('symbol') != symbol or meta.get('month') != month or
                    meta.get('pagination_complete') is not True or meta.get('failure') is not None or
                    not bars_path.is_file() or not audit_path.is_file()):
                raise ValueError(f'incomplete acquisition {symbol} {month}')
            bars_sha = sha(bars_path)
            if bars_sha != meta.get('bars_sha256'):
                raise ValueError(f'changed downloaded bars {symbol} {month}')
            frame = pd.read_parquet(bars_path)
            if list(frame.columns) != list(recovery.CANONICAL_COLUMNS):
                raise ValueError(f'noncanonical acquisition bars {symbol} {month}')
            if (not frame.empty and
                    (set(frame.symbol.dropna()) != {symbol} or
                     set(frame.price_basis.dropna()) != {'NONE'} or
                     not frame.session_date.between(start, end).all())):
                raise ValueError(f'acquisition bars outside frozen scope {symbol} {month}')
            quality = batch.quality(frame, start, end, calendar)
            if quality != meta.get('quality'):
                raise ValueError(f'acquisition quality changed {symbol} {month}')
            audit = json.loads(audit_path.read_text())
            if not isinstance(audit, list):
                raise ValueError(f'acquisition audit invalid {symbol} {month}')
            verified_local = (version == local_reuse.VERSION and
                              meta.get('tier') == 'local_verified_provider')
            if version in (recovery.VERSION,local_reuse.VERSION) and not part.is_symlink() and frame.empty and not verified_local:
                successes = [event for event in audit if event.get('ok') is True]
                if (not successes or successes[-1].get('rows') != 0 or
                        successes[-1].get('has_more') is not False):
                    raise ValueError(f'empty provider reply not audited {symbol} {month}')
                if not pd.read_parquet(part / f"raw_{successes[-1]['page']:03d}.parquet").empty:
                    raise ValueError(f'empty provider reply raw page changed {symbol} {month}')
            for event in audit:
                if event.get('ok') is True and 'page' in event:
                    raw_path = part / f"raw_{event['page']:03d}.parquet"
                    if not raw_path.is_file() or sha(raw_path) != event.get('sha'):
                        raise ValueError(f'changed raw acquisition page {symbol} {month}')
            for raw_path in sorted(part.glob('raw_*.parquet')):
                provenance.append({'role': 'acquisition_raw_page', 'source': str(raw_path),
                                   'sha256': sha(raw_path)})
            if version in (recovery.VERSION,local_reuse.VERSION) and not part.is_symlink():
                empty = bool(frame.empty)
                expected_state = ('provider_returned_no_bars_unknown_listing_state' if empty else
                                  'observed_partial_coverage' if verified_local and not quality['rth_complete']
                                  else 'observed_bars')
                allowed_empty_tiers = ('futu','local_verified_provider') if version == local_reuse.VERSION else ('futu',)
                if (meta.get('empty_success') is not empty or meta.get('data_state') != expected_state or
                        (empty and (meta.get('tier') not in allowed_empty_tiers or quality['rth_complete']))):
                    raise ValueError(f'inconsistent successful empty state {symbol} {month}')
                if empty:
                    empty_parts.append({'symbol': symbol, 'month': month,
                                        'data_state': meta['data_state']})
            elif frame.empty:
                raise ValueError(f'unregistered empty acquisition {symbol} {month}')
            for role, path in (('acquisition_result', meta_path),
                               ('acquisition_audit', audit_path), ('acquisition_bars', bars_path)):
                provenance.append({'role': role, 'source': str(path), 'sha256': sha(path)})
            coverage.append({'symbol': symbol, 'month': month,
                             'pagination_complete': True, **quality})
            if quality['missing_rth_bars']:
                unknown_gaps.append({'symbol': symbol, 'month': month,
                                     'missing_rth_bars': quality['missing_rth_bars'],
                                     'reason': 'unknown_market_data_coverage'})
    return provenance, coverage, empty_parts, unknown_gaps


def join_2025_parts(parts: list[pd.DataFrame], source_2026: Path, symbol: str) -> pd.DataFrame:
    earlier = pd.concat(parts, ignore_index=True).sort_values('start_at')
    if earlier.start_at.duplicated().any():
        raise ValueError('overlapping month sources')
    if earlier.empty and pd.read_parquet(source_2026).empty:
        raise ValueError(f'no observed bars in 2025 or 2026 for {symbol}')
    return earlier


def prepare(acquisition, output, focus_only=False):
    if output.exists():raise FileExistsError(output)
    baseline=HERE.parent/'runs/focus_v8_20260927'
    old_inputs=baseline/'inputs'
    old_config=baseline/'config.json'
    config=json.loads(old_config.read_text())
    config.update(protocol='precision_v9_multiyear',feature_start='2025-08-01',data_end='2026-09-24',
                  sample_start='2025-10-01',sample_end='2026-09-17',calendar='market_data/calendars/multiyear.json',
                  group_run='market_data/groups_v9')
    old_universe=old_inputs/config['universe']
    universe=json.loads(old_universe.read_text())
    if focus_only:
        # Different peer/fit universe: allowed only with an explicit follow-up registration.
        raise ValueError('focus-only changes the peer universe; register a separate version first')
    symbols=sorted({m['symbol'] for m in universe['members'] if m['role']=='candidate'}|{'QQQ'})
    if len(symbols) != 109:
        raise ValueError('frozen R03 universe must have 109 symbols')
    acquisition_provenance, coverage, empty_parts, unknown_gaps = validate_acquisition(
        acquisition, symbols, baseline)
    inputs=output/'inputs';inputs.mkdir(parents=True)
    provenance=acquisition_provenance;actions_uncertified=[]
    for role, path in (('v8_config',old_config),('v8_universe',old_universe),
                       ('v8_source_provenance',baseline/'source_provenance.json'),
                       ('v8_calendar',old_inputs/'market_data/calendars/nasdaq_sessions_2026_v1.json'),
                       ('v9_protocol',HERE/'PROTOCOL.md'),
                       ('r03_backlog',HERE/'rounds/R03/BACKLOG.md'),
                       ('r03_matrix',HERE/'rounds/R03/MATRIX.md')):
        provenance.append({'role':role,'source':str(path),'sha256':sha(path)})
    for symbol in symbols:
        parts=[]
        for month in ['2025-08','2025-09','2025-10','2025-11','2025-12']:
            part=acquisition/'parts'/symbol/month
            meta=json.loads((part/'result.json').read_text())
            if sha(part/'bars.parquet')!=meta['bars_sha256']:raise ValueError('changed downloaded bars')
            parts.append(pd.read_parquet(part/'bars.parquet'))
        earlier=join_2025_parts(parts,old_inputs/'market_data/us_5m'/symbol/'2026.parquet',symbol)
        dest=inputs/config['source_dir']/symbol/'2025.parquet';dest.parent.mkdir(parents=True)
        earlier.to_parquet(dest,index=False,compression='zstd',compression_level=7)
        for rel in [f'market_data/us_5m/{symbol}/2026.parquet',f'market_data/corporate_actions/{symbol}.parquet']:
            p=old_inputs/rel;d=inputs/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
            provenance.append({'role':'v8_frozen_source','source':str(p),'sha256':sha(p)})
        actions=pd.read_parquet(old_inputs/'market_data/corporate_actions'/f'{symbol}.parquet')
        if actions.empty:
            actions_uncertified.append(symbol)
    write(inputs/config['universe'],universe)
    calendar_dest=inputs/config['calendar']
    calendar_dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(acquisition/'calendar.json',calendar_dest)
    write(output/'config.json',config);write(output/'source_provenance.json',provenance)
    write(output/'coverage.json',{'months':coverage,
          'successful_empty_parts':empty_parts,'unknown_coverage_gaps':unknown_gaps,
          'corporate_actions_quality':'event_list_only_historical_coverage_uncertified',
          'empty_corporate_action_tables':actions_uncertified})
    oldroot=vd.ROOT
    try:
        vd.ROOT=inputs
        make_groups(inputs,config)
        manifest=vd.build(config,output/'dataset')
    finally:vd.ROOT=oldroot
    pair_audit=audit_2026_pairs(baseline/'dataset',output/'dataset',output/'paired_2026_audit.json')
    write(output/'complete.json',{'rows':manifest['rows'],'manifest_sha':sha(output/'dataset/manifest.json'),
                                'builder_sha':sha(Path(vd.__file__)),'prepare_sha':sha(Path(__file__)),
                                'pair_audit_sha':sha(output/'paired_2026_audit.json'),
                                'paired_2026_rows':pair_audit['paired_rows']})
    print(json.dumps({'rows':manifest['rows'],'counts':manifest['counts']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--acquisition',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    prepare(args.acquisition.resolve(),args.output.resolve())
