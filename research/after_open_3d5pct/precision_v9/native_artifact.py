"""Thin R03 native artifact contracts and offline verification.

No fit, data acquisition, threshold selection, publication, or trading occurs here.
The child command imports forward.inference before any torch-dependent reference.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


GROUP_ROUTES = frozenset(('B_group', 'C_group', 'C_no_daily'))
ROUTES = ('B_group', 'B_no_group', 'C_group', 'C_no_group', 'C_no_daily')
ARMS = ('old_common', 'new_common', 'new_expanded_fit')
FOLDS = ('dev1', 'dev2')
CAUSAL = 'v9_causal_history_end_v1'
LEGACY = 'legacy_static_only'
NATIVE_VERSION = 'r03_native_artifact_v1'
DATA_FILES = ('rows.parquet', 'features.npz', 'manifest.json')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def record(path):
    path = Path(path).expanduser().resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def write_new(path, value):
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    return record(path)


def verify_record(item, label='source'):
    if not isinstance(item, dict) or not {'path', 'sha256'} <= set(item):
        raise ValueError(f'{label}: path/hash absent')
    actual = record(item['path'])
    if actual['sha256'] != item['sha256']:
        raise ValueError(f'{label}: source hash mismatch: {item["path"]}')
    return actual


def dataset_identity(root):
    root = Path(root).expanduser().resolve(strict=True)
    files = {name: record(root / 'dataset' / name) for name in DATA_FILES}
    return {'root': str(root), 'files': files,
            'dataset_sha256': digest({name: files[name]['sha256'] for name in DATA_FILES})}


def peer_semantics(route, arm):
    if route not in ROUTES or arm not in ARMS:
        raise ValueError('unknown native route or arm')
    return (LEGACY if arm == 'old_common' else CAUSAL) if route in GROUP_ROUTES else 'excluded'


def model_version(run_name, route, arm, fold, seed):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', run_name):
        raise ValueError('native run name must be an explicit safe identifier')
    if route not in ROUTES or arm not in ARMS or fold not in FOLDS or seed != 3566:
        raise ValueError('native model identity differs from adopted 30-identity plan')
    semantic = 'legacy' if arm == 'old_common' and route in GROUP_ROUTES else (
        'causal' if route in GROUP_ROUTES else 'excluded')
    return f'r03_{run_name}_{route}_{arm}_{fold}_s3566_{semantic}'


def _candidate_file(path):
    doc = json.loads(Path(path).read_text())
    members = doc.get('members')
    if not isinstance(members, list):
        raise ValueError('candidate universe needs explicit members/roles')
    candidates = [x['symbol'] for x in members if x.get('role') == 'candidate']
    benchmarks = [x['symbol'] for x in members if x.get('role') == 'benchmark']
    if (len(candidates) != 108 or len(set(candidates)) != 108 or
            'QQQ' in candidates or 'QQQ' not in benchmarks or
            len(benchmarks) != len(set(benchmarks))):
        raise ValueError('R03 causal group universe must have 108 distinct candidates and separate QQQ')
    return set(candidates)


def _group_calendar(path):
    calendar = json.loads(Path(path).read_text())
    sessions = calendar.get('sessions')
    if (not calendar.get('calendar_id') or not calendar.get('sources') or
            not isinstance(sessions, list) or not sessions):
        raise ValueError('causal calendar lacks official session evidence')
    parsed = []
    eastern = ZoneInfo('America/New_York')
    for item in sessions:
        try:
            day = date.fromisoformat(item['session_date'])
            opened = datetime.fromisoformat(item['open_at'])
            closed = datetime.fromisoformat(item['close_at'])
            minutes = int(item['duration_minutes'])
            valid = (opened.tzinfo is not None and closed.tzinfo is not None and
                     opened.astimezone(eastern).date() == day and
                     opened < closed and
                     (closed-opened).total_seconds() == minutes*60 and
                     minutes in (210, 390))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid or (parsed and (day <= parsed[-1][0] or opened <= parsed[-1][1])):
            raise ValueError('causal calendar session order/time invalid')
        parsed.append((day, opened, closed))
    return parsed


def _causal_sources(plan, selected_root):
    """Bind the actual prepare/data outputs; missing classifications stay missing."""
    selected_root = Path(selected_root).resolve(strict=True)
    source_root = selected_root / 'inputs'
    config_file = selected_root / 'config.json'
    complete_file = selected_root / 'complete.json'
    provenance_file = selected_root / 'source_provenance.json'
    config = json.loads(config_file.read_text())
    paths = {'native_group_versions': source_root / config['group_run'] / 'versions.json',
             'native_group_memberships': source_root / config['group_run'] / 'memberships.json',
             'native_group_catalog': source_root / config['group_run'] / 'groups.json',
             'native_candidate_universe': source_root / config['universe'],
             'native_calendar': source_root / config['calendar'],
             'native_builder_source': Path(__file__).parent / 'data.py',
             'native_prepare_source': Path(__file__).parent / 'prepare.py',
             'native_config': config_file, 'native_complete': complete_file,
             'native_source_provenance': provenance_file}
    paths = {key: path.resolve(strict=True) for key, path in paths.items()}
    required = ('native_group_versions', 'native_group_memberships',
                'native_candidate_universe', 'native_builder_source',
                'native_source_root')
    if not set(required) <= set(plan):
        raise ValueError(f'causal plan lacks actual source paths: {sorted(set(required)-set(plan))}')
    if Path(plan['native_source_root']).expanduser().resolve(strict=True) != source_root:
        raise ValueError('native_source_root differs from actual builder inputs')
    for key, value in plan.items():
        if key.startswith('native_') and key in paths and \
                Path(value).expanduser().resolve(strict=True) != paths[key]:
            raise ValueError(f'{key}: plan differs from actual builder output')
    dataset = json.loads((Path(selected_root) / 'dataset' / 'manifest.json').read_text())
    complete = json.loads(complete_file.read_text())
    if (dataset.get('builder_sha256') != sha(paths['native_builder_source']) or
            complete.get('builder_sha') != dataset['builder_sha256'] or
            complete.get('prepare_sha') != sha(paths['native_prepare_source']) or
            complete.get('manifest_sha') != sha(selected_root / 'dataset' / 'manifest.json') or
            dataset.get('group_quality') !=
            'causal_weekly_reconstruction_on_current_universe_not_original_PIT'):
        raise ValueError('causal dataset/complete builder provenance differs')
    source_map = dataset.get('source_hashes')
    if not isinstance(source_map, dict) or not source_map:
        raise ValueError('causal dataset lacks source hash manifest')
    source_records = {}
    for relative, expected in sorted(source_map.items()):
        path = (source_root / relative).resolve(strict=True)
        if not path.is_relative_to(source_root) or sha(path) != expected:
            raise ValueError(f'causal source changed or escaped root: {relative}')
        source_records[relative] = record(path)
    for key in ('native_group_versions', 'native_group_memberships',
                'native_group_catalog', 'native_candidate_universe', 'native_calendar'):
        path = paths[key]
        relative = str(path.relative_to(source_root))
        if source_map.get(relative) != sha(path):
            raise ValueError(f'{key}: actual dataset source map does not bind file')
    candidates = _candidate_file(paths['native_candidate_universe'])
    old_root = Path(plan['old_dataset_root']).expanduser().resolve(strict=True)
    old_config = old_root / 'config.json'
    old_universe = old_root / 'inputs' / json.loads(old_config.read_text())['universe']
    frozen_candidates = _candidate_file(old_universe)
    if candidates != frozen_candidates:
        raise ValueError('causal candidate set differs from frozen 108-member baseline pool')
    paths['native_frozen_old_config'] = old_config.resolve(strict=True)
    paths['native_frozen_old_universe'] = old_universe.resolve(strict=True)
    versions = json.loads(paths['native_group_versions'].read_text())
    members = json.loads(paths['native_group_memberships'].read_text())
    if not isinstance(versions, list) or not isinstance(members, list) or not versions:
        raise ValueError('causal group metadata is absent')
    all_sessions = _group_calendar(paths['native_calendar'])
    sessions = [(day, opened, closed) for day, opened, closed in all_sessions
                if config['feature_start'] <= day.isoformat() <= config['data_end']]
    if not sessions:
        raise ValueError('causal calendar does not cover builder date window')
    first_by_week = {}
    session_dates = {day.isoformat() for day, _, _ in all_sessions}
    for day, opened, _ in sessions:
        week_year, week_number, _ = day.isocalendar()
        first_by_week.setdefault(f'{week_year}-W{week_number:02d}', (day, opened))
    ordered_weeks = list(first_by_week)
    next_open = {week: first_by_week[ordered_weeks[index+1]][1]
                 for index, week in enumerate(ordered_weeks[:-1])}
    from .forward.features import GROUP_CATEGORIES
    identities = {}
    for v in versions:
        key = (v.get('week_id'), v.get('strategy_id'))
        try:
            first_day, first_open = first_by_week[key[0]]
            effective_from = datetime.fromisoformat(v['effective_from'])
            effective_to = datetime.fromisoformat(v['effective_to'])
            cutoff = datetime.fromisoformat(v['feature_cutoff_at'])
            valid_clock = (effective_from.tzinfo is not None and
                           effective_to.tzinfo is not None and cutoff.tzinfo is not None and
                           effective_from == first_open and cutoff == first_open and
                           first_open < effective_to <= first_open + timedelta(days=8) and
                           (key[0] not in next_open or effective_to == next_open[key[0]]) and
                           v['first_session'] == first_day.isoformat())
        except (KeyError, TypeError, ValueError):
            valid_clock = False
        if (v.get('membership_basis') != 'causal_weekly_reconstruction' or
                key[1] not in GROUP_CATEGORIES or key in identities or
                v.get('version_id') != f'v9:{key[1]}:{key[0]}' or not valid_clock):
            raise ValueError('causal group version identity/semantics invalid')
        identities[key] = v
    seen = set()
    for m in members:
        key = (m.get('week_id'), m.get('strategy_id'))
        v = identities.get(key)
        symbol = m.get('symbol')
        if v is None or symbol not in candidates or m.get('version_id') != v['version_id']:
            raise ValueError('classified member has unknown candidate/version')
        identity = (*key, symbol)
        if identity in seen:
            raise ValueError('duplicate classified candidate')
        seen.add(identity)
        groups = m.get('group_ids')
        allowed = {f'{key[1]}:{cat}' for cat in GROUP_CATEGORIES[key[1]]}
        history = (m.get('facts') or {}).get('history_end')
        try:
            valid_history = (date.fromisoformat(history).isoformat() == history and
                             history in session_dates)
        except (TypeError, ValueError):
            valid_history = False
        if (not isinstance(groups, list) or len(groups) != 1 or groups[0] not in allowed or
                not valid_history or history >= v['first_session']):
            raise ValueError('classified group category/history_end is not causal')
    # make_groups records only successful classifications. Absence is a genuine
    # zero-mask/insufficient-history state, never a fabricated category.
    refs = {key: record(path) for key, path in paths.items()}
    return {'peer_semantics': CAUSAL, 'references': refs,
            'source_hashes': source_records, 'candidate_count': len(candidates),
            'version_count': len(versions), 'membership_count': len(members),
            'unclassified_count': len(identities) * len(candidates) - len(seen),
            'membership_sha256': refs['native_group_memberships']['sha256']}


def _split_evidence(rows, fold_ids, recipe, seed, selected_root):
    from ..focus_v8 import run as old
    ordered = {}
    for name, ids in fold_ids.items():
        keys = rows.iloc[ids].sample_id.astype(str).tolist()
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate split sample IDs')
        ordered[name] = {'ids': [int(i) for i in ids], 'sample_ids': keys,
                         'sample_ids_sha256': digest(keys)}
    weights = old.training_weights(rows, fold_ids['fit'], recipe, seed)
    if len(weights) != len(fold_ids['fit']) or not np.isfinite(weights).all():
        raise ValueError('fit weights invalid')
    positive = fold_ids['fit'][weights > 0]
    if not len(positive):
        raise ValueError('fit has no positive-weight IDs')
    supervision = {'positive_fit_ids': [int(i) for i in positive],
                   'positive_fit_sample_ids': rows.iloc[positive].sample_id.astype(str).tolist(),
                   'weights_float64_sha256': hashlib.sha256(np.asarray(weights, np.float64).tobytes()).hexdigest()}
    prior = None
    if recipe.get('prior'):
        from ..focus_v8.core import mature_prior
        with np.load(Path(selected_root) / 'dataset' / 'features.npz') as archive:
            y = archive['y'].reshape(-1, 9)
        _, counts = mature_prior(rows, y)
        available = pd.to_datetime(rows.label_available_at, utc=True).astype('int64').to_numpy()
        end = pd.to_datetime(rows.label_end_at, utc=True).astype('int64').to_numpy()
        decisions = pd.to_datetime(rows.decision_at, utc=True).astype('int64').to_numpy()
        history = [[] for _ in range(len(rows))]
        for _, group in rows.groupby('symbol').indices.items():
            ids = np.asarray(sorted(group, key=lambda i: decisions[i]))
            for i in ids:
                earlier = ids[(decisions[ids] < decisions[i]) &
                              (available[ids] < decisions[i]) &
                              (end[ids] < decisions[i])][-63:]
                if len(earlier) != counts[i]:
                    raise ValueError('frozen mature prior IDs/counts differ')
                history[i] = rows.iloc[earlier].sample_id.astype(str).tolist()
        prior = {'counts': counts.astype(int).tolist(), 'history_ids': history,
                 'all_candidate_rows': len(rows)}
    return {'splits': ordered, 'supervision': supervision,
            'prior': prior,
            'prior_scope': 'all_rows_in_selected_immutable_dataset_before_allowed_key_filter'}


def _roled_prefit_sources(plan_file, plan, registration, selected_root):
    from . import baseline as baseline_module
    plan_file = Path(plan_file).resolve(strict=True)
    if json.loads(plan_file.read_text()) != plan:
        raise ValueError('native plan dictionary differs from immutable plan file')
    required = ('old_dataset_root', 'new_dataset_root', 'dataset_root',
                'actual_manifest', 'paired_audit', 'calendar', 'allowed_keys',
                'incumbent_summary', 'native_fit_adoption')
    if not set(required) <= set(plan):
        raise ValueError(f'native plan lacks adopted source roles: {sorted(set(required)-set(plan))}')
    old_root = Path(plan['old_dataset_root']).resolve(strict=True)
    new_root = Path(plan['new_dataset_root']).resolve(strict=True)
    chosen = old_root if registration['arm'] == 'old_common' else new_root
    if (old_root == new_root or chosen != Path(selected_root).resolve(strict=True) or
            chosen != Path(plan['dataset_root']).resolve(strict=True)):
        raise ValueError('native arm dataset differs from distinct registered snapshots')
    actual_path = Path(plan['actual_manifest']).resolve(strict=True)
    actual = json.loads(actual_path.read_text())
    if actual.get('status') != 'accepted_for_fit':
        raise ValueError('native formal fit lacks accepted actual_manifest')
    baseline_module._validate_actual_manifest(
        actual_path, new_root, Path(plan['calendar']).resolve(strict=True),
        Path(plan['paired_audit']).resolve(strict=True), 'r03_baseline')
    roles = {'plan': plan_file, **{name: Path(plan[name]).resolve(strict=True)
                                  for name in required if name not in
                                  ('old_dataset_root', 'new_dataset_root', 'dataset_root')}}
    roles['native_registration'] = (Path(__file__).parent /
        'rounds/R03/native_artifact/REGISTRATION_DRAFT.md').resolve(strict=True)
    roles['native_engineering_adoption'] = (Path(__file__).parent /
        'rounds/R03/native_artifact/ADOPTION.md').resolve(strict=True)
    refs = {name: record(path) for name, path in roles.items()}
    for name, ref in refs.items():
        if registration['source_hashes'].get(ref['path']) != ref['sha256']:
            raise ValueError(f'{name}: role file omitted from baseline source registration')
    return {'roles': refs, 'old_dataset': dataset_identity(old_root),
            'new_dataset': dataset_identity(new_root),
            'accepted_status': 'accepted_for_fit'}


def register_prefit(plan_file, plan, registration, rows, fold_ids, selected_root, output,
                    source_code_paths):
    """Write schema/contract before fit; caller links their byte hashes in registration."""
    from .forward.schema import build_schema
    route, arm, fold = registration['route'], registration['arm'], registration['fold']['id']
    if fold not in FOLDS:
        raise ValueError('native pre-fit contract is limited to dev1/dev2')
    if registration['purpose'] != 'r03_baseline':
        raise ValueError('engineering smoke cannot be represented as native prefit R03')
    if 'native_run_id' not in plan:
        raise ValueError('formal R03 plan lacks adopted native_run_id')
    semantics = peer_semantics(route, arm)
    model = model_version(plan['native_run_id'], route, arm, fold, registration['seed'])
    schema = build_schema(route, registration['recipe'], semantics)
    schema_file = write_new(Path(output) / 'native_schema.json', schema)
    selected = dataset_identity(selected_root)
    roled = _roled_prefit_sources(plan_file, plan, registration, selected_root)
    split = _split_evidence(rows, fold_ids, registration['recipe'], registration['seed'],
                            selected_root)
    split_file = write_new(Path(output) / 'native_split_ids.json', split)
    source_records = [record(path) for path in source_code_paths]
    if (len(set(x['path'] for x in source_records)) != len(source_records) or
            any(registration['source_hashes'].get(x['path']) != x['sha256'] for x in source_records)):
        raise ValueError('native training source list differs from baseline registration')
    causal = _causal_sources(plan, selected_root) if semantics == CAUSAL else None
    core = digest(registration)
    contract = {'version': NATIVE_VERSION, 'mode': 'prefit_native',
                'run_id': plan['native_run_id'],
                'identity': {'route': route, 'arm': arm, 'fold': fold,
                             'seed': registration['seed'], 'model_version': model},
                'peer_semantics': semantics, 'recipe': registration['recipe'],
                'schema_file': schema_file, 'selected_dataset': selected,
                'roled_preflight_sources': roled,
                'registration_core_sha256': core,
                'registered_source_hashes': registration['source_hashes'],
                'training_code_files': source_records,
                'split_file': split_file, 'split_support': registration['split_support'],
                'fit_date_floor': registration['fit_date_floor'],
                'prior_scope': registration['prior_scope'],
                'causal_sources': causal,
                'time_contract': {'training_rows_decision_at': '11:30:30',
                                  'forward_decision_anchor': '11:30',
                                  'information_deadline': '11:30:30'}}
    contract_file = write_new(Path(output) / 'native_contract.json', contract)
    return {'native_contract_file': contract_file,
            'native_schema_file': schema_file,
            'native_model_version': model,
            'native_peer_semantics': semantics,
            'native_registration_core_sha256': core}


def _check_dataset(identity):
    files = identity['files']
    if set(files) != set(DATA_FILES):
        raise ValueError('native dataset identity omits a required file')
    for name in DATA_FILES:
        actual = verify_record(files[name], f'dataset {name}')
        if actual['path'] != str(Path(identity['root']) / 'dataset' / name):
            raise ValueError('dataset file moved outside registered root')
    if digest({name: files[name]['sha256'] for name in DATA_FILES}) != identity['dataset_sha256']:
        raise ValueError('aggregate dataset identity differs')


def _verify_causal_export(contract, exported):
    source = contract['causal_sources']
    expected = ({'dataset_sha256': contract['selected_dataset']['dataset_sha256'],
                 'membership_sha256': source['membership_sha256'],
                 'references': source['references'],
                 'source_hashes': source['source_hashes']}
                if source else None)
    if exported != expected:
        raise ValueError('exported causal files differ from pre-fit training source contract')
    return expected


def verify_prefit(run, contract_ref):
    run = Path(run).expanduser().resolve(strict=True)
    verify_record(contract_ref, 'native contract')
    contract = json.loads(Path(contract_ref['path']).read_text())
    if contract.get('version') != NATIVE_VERSION or contract.get('mode') != 'prefit_native':
        raise ValueError('not a pre-fit native contract')
    if Path(contract_ref['path']) != run / 'native_contract.json':
        raise ValueError('contract does not belong to completed run')
    for name in ('schema_file', 'split_file'):
        verify_record(contract[name], name)
    _check_dataset(contract['selected_dataset'])
    roled = contract.get('roled_preflight_sources')
    if not isinstance(roled, dict) or roled.get('accepted_status') != 'accepted_for_fit':
        raise ValueError('native contract lacks accepted role-bound preflight')
    for item in roled['roles'].values():
        verify_record(item, 'role-bound pre-fit source')
    _check_dataset(roled['old_dataset'])
    _check_dataset(roled['new_dataset'])
    plan_file = Path(roled['roles']['plan']['path'])
    plan = json.loads(plan_file.read_text())
    registration_path = run / 'registration.json'
    registration = json.loads(registration_path.read_text())
    if _roled_prefit_sources(plan_file, plan, registration,
                            contract['selected_dataset']['root']) != roled:
        raise ValueError('native role-bound preflight changed')
    if registration.get('native_contract_file') != contract_ref:
        raise ValueError('registration does not bind native contract bytes')
    core = {key: value for key, value in registration.items() if not key.startswith('native_')}
    if digest(core) != contract['registration_core_sha256'] or \
            registration.get('native_registration_core_sha256') != digest(core):
        raise ValueError('native registration core changed')
    identity = contract['identity']
    if (identity != {'route': registration['route'], 'arm': registration['arm'],
                     'fold': registration['fold']['id'], 'seed': registration['seed'],
                     'model_version': registration['native_model_version']} or
            contract['recipe'] != registration['recipe'] or
            contract['peer_semantics'] != registration['native_peer_semantics'] or
            contract['schema_file'] != registration['native_schema_file']):
        raise ValueError('native route/arm/fold/recipe/schema linkage differs')
    if identity['model_version'] != model_version(contract['run_id'], identity['route'],
                                                identity['arm'], identity['fold'],
                                                identity['seed']):
        raise ValueError('native model version was renamed after fit')
    if contract['registered_source_hashes'] != registration['source_hashes']:
        raise ValueError('native source map differs from baseline registration')
    for path, expected in contract['registered_source_hashes'].items():
        if sha(path) != expected:
            raise ValueError(f'pre-fit source changed after registration: {path}')
    for item in contract['training_code_files']:
        verify_record(item, 'training code')
    split = json.loads(Path(contract['split_file']['path']).read_text())
    if (set(split['splits']) != {'fit', 'tune', 'cal', 'eval', 'eval_full'} or
            split['prior_scope'] != registration['prior_scope'] or
            contract['split_support'] != registration['split_support']):
        raise ValueError('pre-fit split/prior contract differs')
    for name, item in split['splits'].items():
        if len(item['ids']) != len(item['sample_ids']) or \
                digest(item['sample_ids']) != item['sample_ids_sha256']:
            raise ValueError(f'{name}: ordered split IDs differ')
    causal = contract.get('causal_sources')
    if contract['peer_semantics'] == CAUSAL:
        if not isinstance(causal, dict) or causal.get('peer_semantics') != CAUSAL:
            raise ValueError('causal group lacks frozen training sources')
        for item in causal['references'].values():
            verify_record(item, 'causal metadata/builder')
        for item in causal['source_hashes'].values():
            verify_record(item, 'causal source')
        if causal['membership_sha256'] != causal['references']['native_group_memberships']['sha256']:
            raise ValueError('causal membership digest differs')
    elif causal is not None:
        raise ValueError('legacy/excluded route carries causal training claim')
    return registration, contract, split


def _model_files(run, route):
    names = [f'target_{j}.txt' for j in range(9)] if route.startswith('B') else ['model.pt']
    return [record(Path(run) / name) for name in names]


def prior_history_rows(path):
    """Normalize Parquet list cells (often ndarray) before exact ID comparison."""
    frame = pd.read_parquet(path)
    return {'sample_ids': frame.sample_id.astype(str).tolist(),
            'counts': frame.mature_prior_count.astype(int).tolist(),
            'history_ids': [list(value) for value in frame.prior_history_ids]}


def _copy_serving_sources(output, paths):
    root = Path(__file__).resolve().parents[3]
    records = []
    seen = set()
    for path in paths:
        path = Path(path).resolve(strict=True)
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(root)
        target = Path(output) / 'serving_sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != sha(path):
            raise ValueError('serving code copy differs')
        records.append(record(target))
    return records


def _verified_run(run, registration, *, posthoc=False, split=None):
    run = Path(run).resolve(strict=True)
    result = json.loads((run / 'result.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    identity = (registration['route'], registration['arm'], registration['fold']['id'])
    if (result.get('route'), result.get('arm'), result.get('fold')) != identity or \
            result.get('recipe') != registration['recipe'] or \
            result.get('seed') != registration['seed'] or \
            not result.get('source_unchanged') or \
            result.get('thresholds') != selection.get('thresholds') or \
            selection.get('selection_source') != 'cal_only':
        raise ValueError('completed result/selection differs from registered fit')
    from ..focus_v8.core import GROUPS
    thresholds = selection['thresholds']
    if set(thresholds) != set(GROUPS) or any(
            value is not None and (not isinstance(value, (int, float)) or
                                   not 0 <= value <= 1) for value in thresholds.values()):
        raise ValueError('cal-only thresholds are incomplete or invalid')
    calibration = json.loads((run / 'calibration.json').read_text())
    if (not registration['recipe'].get('calibration') or
            calibration.get('method') != 'shrunken_nonnegative_platt' or
            calibration.get('fitted_on') != 'cal_only' or
            np.asarray(calibration.get('parameters')).shape != (9, 2)):
        raise ValueError('registered calibration differs from fitted artifact')
    if split is None and posthoc:
        from . import baseline as baseline_module
        plan = json.loads((run.parent / 'registration' /
            f'{identity[0]}_{identity[2]}_plan.json').read_text())
        rows = pd.read_parquet(Path(plan['dataset_root']) / 'dataset/rows.parquet')
        allowed = baseline_module._allowed_keys(Path(plan['allowed_keys']))
        fold_ids = baseline_module.split_indices(
            rows, baseline_module._fold_config(identity[2]), allowed,
            identity[1], registration['fit_date_floor'])
        split = {'splits': {name: {'sample_ids': rows.iloc[index].sample_id.astype(str).tolist()}
                            for name, index in fold_ids.items()}}
    if split is None:
        raise ValueError('completed native run lacks pre-fit ordered split evidence')
    for name in ('fit', 'tune', 'cal', 'eval', 'eval_full'):
        frame = pd.read_parquet(run / f'{name}.parquet', columns=['sample_id'])
        actual = frame.sample_id.astype(str).tolist()
        if (len(actual) != registration['split_support'][name]['rows'] or
                len(actual) != len(set(actual)) or
                actual != split['splits'][name]['sample_ids']):
            raise ValueError(f'{name}: completed prediction IDs/order differ from pre-fit split')
    if (not posthoc and pd.read_parquet(run/'fit_supervision_ids.parquet',
                                       columns=['sample_id']).sample_id.astype(str).tolist() !=
            split['supervision']['positive_fit_sample_ids']):
        raise ValueError('positive-weight fit IDs differ from pre-fit contract')
    if posthoc and (registration['purpose'] != 'engineering_validation_only' or
                    identity[1:] != ('old_common', 'dev1') or identity[0] in GROUP_ROUTES):
        raise ValueError('posthoc engine wrapper is limited to two no-group dev1 runs')
    return result, selection


def _assemble_manifest(output, run, registration, schema_ref, model,
                       serving_sources, provenance, version, semantics, *, posthoc):
    schema = json.loads(Path(schema_ref['path']).read_text())
    calibration = record(Path(run) / 'calibration.json')
    selection = json.loads((Path(run) / 'selection.json').read_text())
    manifest = {'route_id': registration['route'], 'model_version': version,
                'recipe': registration['recipe'], 'schema': schema,
                'schema_file': schema_ref, 'schema_sha256': schema_ref['sha256'],
                'model_files': model, 'model_sha256': digest([x['sha256'] for x in model]),
                'calibration_file': calibration,
                'calibration_sha256': calibration['sha256'],
                'source_files': serving_sources,
                'source_code_sha256': digest([x['sha256'] for x in serving_sources]),
                'score_column': 'raw_3d_5pct', 'thresholds': selection['thresholds'],
                'action_frozen': False,
                'native_provenance_file': provenance,
                'native_provenance_sha256': provenance['sha256'],
                'engineering_only': posthoc}
    if semantics == CAUSAL:
        source = json.loads(Path(provenance['path']).read_text())['causal_sources']
        manifest['group_training_provenance'] = {
            'peer_semantics': CAUSAL,
            'model_sha256': manifest['model_sha256'],
            'schema_sha256': manifest['schema_sha256'],
            'dataset_sha256': source['dataset_sha256'],
            'membership_sha256': source['membership_sha256']}
    write_new(Path(output) / 'route_artifact.json', manifest)
    return manifest


def _finish_export(output, run, registration, contract, *, posthoc, training_sources):
    from .forward.schema import build_schema
    route, arm, fold = registration['route'], registration['arm'], registration['fold']['id']
    semantics = peer_semantics(route, arm)
    model = _model_files(run, route)
    if contract is None:
        schema = build_schema(route, registration['recipe'], semantics)
        schema_ref = write_new(Path(output) / 'native_schema_posthoc.json', schema)
        version = f'posthoc_engineering_{Path(run).parent.name}_{route}_{fold}'
        data_id = dataset_identity(json.loads((Path(run).parent / 'registration' /
            f'{route}_{fold}_plan.json').read_text())['dataset_root'])
        causal_sources = None
    else:
        schema_ref = contract['schema_file']
        version = contract['identity']['model_version']
        data_id = contract['selected_dataset']
        causal = contract.get('causal_sources')
        causal_sources = ({'dataset_sha256': data_id['dataset_sha256'],
                           'membership_sha256': causal['membership_sha256'],
                           'references': causal['references'],
                           'source_hashes': causal['source_hashes']}
                          if causal else None)
    _check_dataset(data_id)
    from .forward import inference
    from . import baseline as baseline_module
    serving = _copy_serving_sources(output, [Path(__file__), Path(inference.__file__),
        Path(__file__).parent / 'forward' / 'schema.py',
        Path(__file__).parent / 'forward' / 'features.py',
        *baseline_module.SOURCE_CODE])
    checks = {name: record(Path(run) / name) for name in
              ('result.json', 'selection.json', 'calibration.json', 'registration.json',
               'fit_supervision_ids.parquet', 'fit.parquet', 'tune.parquet',
               'cal.parquet', 'eval.parquet', 'eval_full.parquet')
             if (Path(run) / name).exists()}
    if contract and (Path(run) / 'prior_history_ids.parquet').exists():
        checks['prior_history_ids.parquet'] = record(Path(run) / 'prior_history_ids.parquet')
    for name in ('dataset_identity.json', 'path_summary_identity.json', 'path_summary.npy'):
        checks[f'cache/{name}'] = record(Path(run) / 'cache' / name)
    provenance = {'version': NATIVE_VERSION,
                  'mode': 'posthoc_engineering_only' if posthoc else 'prefit_native',
                  'identity': {'route': route, 'arm': arm, 'fold': fold,
                               'seed': registration['seed'], 'model_version': version},
                  'peer_semantics': semantics,
                  'registered_before_fit': not posthoc,
                  'pre_fit_contract': contract['contract_ref'] if contract else None,
                  'registration': checks['registration.json'],
                  'run_files': checks, 'models': model,
                  'schema_file': schema_ref, 'dataset': data_id,
                  'training_sources': training_sources,
                  'causal_sources': causal_sources,
                  'serving_sources': serving,
                  'result_source_unchanged': True,
                  'development_only': True}
    provenance_ref = write_new(Path(output) / 'training_provenance.json', provenance)
    return _assemble_manifest(output, run, registration, schema_ref, model,
                              serving, provenance_ref, version, semantics, posthoc=posthoc)


def export_route(completed_run, native_contract, new_output):
    """Read-only export from an actually completed, pre-registered R03 fit."""
    run = Path(completed_run).expanduser().resolve(strict=True)
    output = Path(new_output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        contract_ref = record(native_contract)
        registration, contract, split = verify_prefit(run, contract_ref)
        _verified_run(run, registration, split=split)
        if registration['purpose'] != 'r03_baseline':
            raise ValueError('native export requires accepted R03 baseline purpose')
        fit = pd.read_parquet(run / 'fit_supervision_ids.parquet', columns=['sample_id'])
        if fit.sample_id.astype(str).tolist() != split['supervision']['positive_fit_sample_ids']:
            raise ValueError('completed fit supervision differs from pre-fit IDs')
        if split['prior'] is not None:
            history = prior_history_rows(run / 'prior_history_ids.parquet')
            if (history['sample_ids'] !=
                    pd.read_parquet(contract['selected_dataset']['files']['rows.parquet']['path'])
                    .sample_id.astype(str).tolist() or
                    history['counts'] != split['prior']['counts'] or
                    history['history_ids'] != split['prior']['history_ids']):
                raise ValueError('completed mature-prior IDs differ from pre-fit contract')
        contract['contract_ref'] = contract_ref
        sources = [{'original_path': path, 'verified_file': record(path)}
                   for path in contract['registered_source_hashes']]
        manifest = _finish_export(output, run, registration, contract,
                                  posthoc=False, training_sources=sources)
        verify_native_manifest(manifest)
        return output / 'route_artifact.json'
    except Exception as exc:
        write_new(output / 'export_failure.json', {'type': type(exc).__name__,
                   'message': str(exc), 'recorded_at_utc': datetime.now(timezone.utc).isoformat()})
        raise


def export_posthoc_engineering(completed_run, new_output):
    """Wrap the two existing no-group engine checks without backdating a contract."""
    run = Path(completed_run).expanduser().resolve(strict=True)
    output = Path(new_output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        registration = json.loads((run / 'registration.json').read_text())
        _verified_run(run, registration, posthoc=True)
        old_sources = []
        root = Path(__file__).resolve().parents[3]
        for original, expected in sorted(registration['source_hashes'].items()):
            relative = Path(original).relative_to(root)
            frozen = run / 'source_snapshot' / relative
            check = frozen if frozen.exists() else Path(original)
            verified = record(check)
            if verified['sha256'] != expected:
                raise ValueError(f'original engineering source bytes changed: {original}')
            old_sources.append({'original_path': original, 'verified_file': verified})
        manifest = _finish_export(output, run, registration, None,
                                  posthoc=True, training_sources=old_sources)
        verify_native_manifest(manifest)
        return output / 'route_artifact.json'
    except Exception as exc:
        write_new(output / 'export_failure.json', {'type': type(exc).__name__,
                   'message': str(exc), 'recorded_at_utc': datetime.now(timezone.utc).isoformat()})
        raise


def verify_native_manifest(manifest):
    """Verify actual registered sources and their binding to model/schema.

    Returns every verified source record for post-load mutation checks.
    Legacy hand-assembled fixture manifests intentionally have no such claim.
    """
    if 'native_provenance_file' not in manifest:
        raise ValueError('native artifact lacks provenance file')
    provenance_ref = verify_record(manifest['native_provenance_file'], 'native provenance')
    if provenance_ref['sha256'] != manifest.get('native_provenance_sha256'):
        raise ValueError('native provenance SHA differs from manifest')
    provenance = json.loads(Path(provenance_ref['path']).read_text())
    if provenance.get('version') != NATIVE_VERSION:
        raise ValueError('unknown native provenance version')
    identity = provenance['identity']
    if (identity['route'] != manifest['route_id'] or
            identity['model_version'] != manifest['model_version'] or
            provenance['schema_file'] != manifest['schema_file'] or
            provenance['models'] != manifest['model_files'] or
            provenance['serving_sources'] != manifest['source_files'] or
            provenance['peer_semantics'] != peer_semantics(identity['route'], identity['arm'])):
        raise ValueError('native provenance differs from route artifact')
    verified = [provenance_ref]
    for item in provenance['run_files'].values():
        verified.append(verify_record(item, 'completed run'))
    for item in provenance['models']:
        verified.append(verify_record(item, 'trained model'))
    verified.append(verify_record(provenance['schema_file'], 'native schema'))
    for item in provenance['serving_sources']:
        verified.append(verify_record(item, 'serving source'))
    _check_dataset(provenance['dataset'])
    verified.extend(provenance['dataset']['files'].values())
    for entry in provenance['training_sources']:
        item = verify_record(entry['verified_file'], 'training source')
        verified.append(item)
    if provenance['mode'] == 'prefit_native':
        if (manifest.get('engineering_only') or not provenance['registered_before_fit'] or
                provenance['pre_fit_contract'] is None):
            raise ValueError('pre-fit native provenance claims an engineering wrapper')
        run = Path(provenance['registration']['path']).parent
        registration, contract, split = verify_prefit(run, provenance['pre_fit_contract'])
        if (identity != contract['identity'] or
                registration['native_model_version'] != identity['model_version'] or
                contract['selected_dataset'] != provenance['dataset'] or
                provenance['causal_sources'] is None and provenance['peer_semantics'] == CAUSAL):
            raise ValueError('native contract/source identity differs from fitted artifact')
        expected_causal = _verify_causal_export(contract, provenance['causal_sources'])
        verified.append(provenance['pre_fit_contract'])
        verified.extend({'path': path, 'sha256': expected} for path, expected in
                        contract['registered_source_hashes'].items())
        if provenance['peer_semantics'] == CAUSAL:
            causal = provenance['causal_sources']
            claimed = manifest.get('group_training_provenance')
            if (causal['dataset_sha256'] != provenance['dataset']['dataset_sha256'] or
                    causal['membership_sha256'] !=
                    causal['references']['native_group_memberships']['sha256'] or
                    claimed != {'peer_semantics': CAUSAL,
                                'model_sha256': manifest['model_sha256'],
                                'schema_sha256': manifest['schema_sha256'],
                                'dataset_sha256': causal['dataset_sha256'],
                                'membership_sha256': causal['membership_sha256']}):
                raise ValueError('causal training provenance is not bound to source bytes')
            verified.extend(verify_record(x, 'causal metadata') for x in causal['references'].values())
            verified.extend(verify_record(x, 'causal bars/actions')
                            for x in causal['source_hashes'].values())
        elif manifest.get('group_training_provenance') is not None:
            raise ValueError('legacy/excluded fit carries causal training provenance')
    elif provenance['mode'] == 'posthoc_engineering_only':
        if (not manifest.get('engineering_only') or provenance['registered_before_fit'] or
                provenance['pre_fit_contract'] is not None or
                identity['arm'] != 'old_common' or identity['fold'] != 'dev1' or
                identity['route'] in GROUP_ROUTES or
                provenance['causal_sources'] is not None):
            raise ValueError('posthoc wrapper is misidentified as pre-fit/causal')
        registration = json.loads(Path(provenance['registration']['path']).read_text())
        if (identity != {'route': registration['route'], 'arm': registration['arm'],
                         'fold': registration['fold']['id'], 'seed': registration['seed'],
                         'model_version': manifest['model_version']} or
                registration['purpose'] != 'engineering_validation_only' or
                len(provenance['training_sources']) != len(registration['source_hashes'])):
            raise ValueError('posthoc original registration/source map differs')
        source_map = {x['original_path']: x['verified_file']['sha256']
                      for x in provenance['training_sources']}
        if source_map != registration['source_hashes']:
            raise ValueError('posthoc training bytes do not match original registration')
    else:
        raise ValueError('unknown native export provenance mode')
    run = Path(provenance['registration']['path']).parent
    _verified_run(run, registration,
                  posthoc=provenance['mode'] == 'posthoc_engineering_only',
                  split=split if provenance['mode'] == 'prefit_native' else None)
    from .forward.schema import build_schema
    expected_posthoc_dataset = None
    if provenance['mode'] == 'posthoc_engineering_only':
        plan = json.loads((run.parent / 'registration' /
            f'{identity["route"]}_{identity["fold"]}_plan.json').read_text())
        expected_posthoc_dataset = dataset_identity(plan['dataset_root'])
    if (manifest['recipe'] != registration['recipe'] or
            manifest['thresholds'] != json.loads((run / 'selection.json').read_text())['thresholds'] or
            manifest['calibration_file'] != record(run / 'calibration.json') or
            manifest['calibration_sha256'] != sha(run / 'calibration.json') or
            manifest['model_files'] != _model_files(run, identity['route']) or
            manifest['schema'] != build_schema(identity['route'], registration['recipe'],
                                               provenance['peer_semantics']) or
            manifest['action_frozen'] is not False or
            (expected_posthoc_dataset is not None and
             provenance['dataset'] != expected_posthoc_dataset)):
        raise ValueError('native recipe/threshold/calibration/model/dataset differs from fitted run')
    current = {str(Path(__file__).resolve()): sha(__file__)}
    from .forward import inference, schema as schema_module, features
    from . import baseline as baseline_module
    current.update({str(Path(inference.__file__).resolve()): sha(inference.__file__),
                    str(Path(schema_module.__file__).resolve()): sha(schema_module.__file__),
                    str(Path(features.__file__).resolve()): sha(features.__file__)})
    current.update({str(Path(path).resolve(strict=True)): sha(path)
                    for path in baseline_module.SOURCE_CODE})
    relative_to_digest = {str(Path(x['path']).relative_to(
        Path(provenance_ref['path']).parent / 'serving_sources')): x['sha256']
        for x in provenance['serving_sources']}
    root = Path(__file__).resolve().parents[3]
    for path, actual in current.items():
        relative = str(Path(path).relative_to(root))
        if relative_to_digest.get(relative) != actual:
            raise ValueError(f'installed native runtime differs from exported bytes: {relative}')
    return verified


def _before_after(records):
    result = []
    for item in records:
        try:
            result.append({'path': item['path'], 'sha256': sha(item['path'])})
        except (OSError, ValueError) as exc:
            result.append({'path': item['path'], 'sha256': None,
                           'error_type': type(exc).__name__, 'error': str(exc)})
    return result


def _run_child(artifact_path, run, selection_path, output):
    # This is the first forward import in the child. Do not move torch imports above it.
    from .forward.inference import load_route, predict_route
    from .forward.features import GROUP_ROUTES
    import lightgbm as lgb
    import scipy.special
    import torch
    from ..focus_v8 import run as old
    from ..focus_v8.core import TARGETS
    from ..train_multiscale_v6 import project_monotone

    started = time.monotonic()
    artifact = load_route(artifact_path)
    manifest = artifact['manifest']
    provenance = json.loads(Path(manifest['native_provenance_file']['path']).read_text())
    if Path(run).resolve(strict=True) != Path(provenance['registration']['path']).parent:
        raise ValueError('child --run differs from artifact training run')
    route, recipe = manifest['route_id'], manifest['recipe']
    selection = json.loads(Path(selection_path).read_text())
    sample_ids = selection.get('sample_ids')
    if not isinstance(sample_ids, list) or not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError('child needs preselected unique sample IDs')
    run = Path(run)
    for name in ('dataset', 'path_summary.npy', 'path_summary_identity.json',
                 'dataset_identity.json'):
        if not (run / 'cache' / name).exists():
            raise ValueError('completed cache lacks frozen reference input')
    if ((run / 'cache' / 'dataset').resolve(strict=True) !=
            Path(provenance['dataset']['root']) / 'dataset'):
        raise ValueError('child cache points to a different dataset')
    cache_identity = json.loads((run / 'cache/dataset_identity.json').read_text())
    expected_identity = {name: provenance['dataset']['files'][name]['sha256']
                         for name in DATA_FILES}
    if cache_identity != expected_identity:
        raise ValueError('child cache dataset identity differs from provenance')
    cache_seal = json.loads((run / 'cache/path_summary_identity.json').read_text())
    if (not cache_seal.get('cache_sha256') or
            cache_seal['cache_sha256'] != sha(run / 'cache/path_summary.npy')):
        raise ValueError('child path cache lacks valid immutable seal')
    rows, data, y, prior, _, paths, xbase, curve, relative, arrays, _ = old.load_data(run / 'cache')
    ids = {key: i for i, key in enumerate(rows.sample_id.astype(str))}
    eval_frame = pd.read_parquet(run / 'eval.parquet')
    eval_keys = eval_frame.sample_id.astype(str).tolist()
    if len(eval_keys) != len(set(eval_keys)) or any(key not in ids for key in eval_keys):
        raise ValueError('frozen eval key/order differs from selected dataset')
    eval_ids = np.asarray([ids[key] for key in eval_keys], dtype=int)
    if route.startswith('B'):
        x = xbase if route == 'B_group' else xbase[:, :585]
        if recipe.get('representation'):
            x = np.column_stack((x, paths))
        if recipe.get('curves'):
            x = np.column_stack((x, curve))
            if route == 'B_group':
                x = np.column_stack((x, relative))
        separate = [lgb.Booster(model_file=item['path']) for item in manifest['model_files']]

        def reference(index):
            raw = []
            for j, booster in enumerate(separate):
                z = booster.predict(x[index], raw_score=True, num_threads=1)
                if recipe.get('prior'):
                    z += scipy.special.logit(np.clip(prior[index, j], .01, .99))
                raw.append(scipy.special.expit(z))
            return np.column_stack(raw)
    else:
        saved = torch.load(manifest['model_files'][0]['path'], map_location='cpu',
                           weights_only=True)
        separate = old.build_c(route, recipe)
        separate.load_state_dict(saved['state_dict'], strict=True)
        separate.eval()

        def reference(index):
            return old.predict_c(separate, arrays, np.asarray(index, dtype=int))

    raw = reference(eval_ids)
    params = np.asarray(artifact['params'])
    processed = project_monotone(old.apply_cal(raw, params)).reshape(-1, 9)
    saved_raw = eval_frame[[f'raw_{name}' for name in TARGETS]].to_numpy(float)
    saved_p = eval_frame[[f'p_{name}' for name in TARGETS]].to_numpy(float)
    if (raw.shape != saved_raw.shape or processed.shape != saved_p.shape or
            not np.isfinite(raw).all() or not np.isfinite(processed).all() or
            not ((0 <= raw) & (raw <= 1)).all() or
            not ((0 <= processed) & (processed <= 1)).all()):
        raise ValueError('child eval prediction shape/probability invalid')
    eval_raw_error = float(np.max(np.abs(raw - saved_raw)))
    eval_p_error = float(np.max(np.abs(processed - saved_p)))
    results = []
    for sample_id in sample_ids:
        if sample_id not in ids:
            raise ValueError(f'preselected key absent: {sample_id}')
        i = ids[sample_id]
        grouped = route in GROUP_ROUTES
        X = {'x5': data['x5'][i], 'x60': data['x60'][i],
             'prior': prior[i], 'xbase': (xbase[i] if grouped else xbase[i, :585]),
             'paths': paths[i], 'curve': curve[i],
             'relative': relative[i] if route == 'B_group' else np.empty(0, np.float32)}
        if route != 'C_no_daily':
            X['xday'] = data['xday'][i]
        if grouped:
            X['group_seq'] = data['group_seq'][i]
        lineage = {'sample_id': sample_id, 'symbol': str(rows.iloc[i].symbol),
                   'session_date': str(rows.iloc[i].session_date),
                   'mode': 'historical_fixture', 'g2_eligible': False,
                   'group_peer_semantics': manifest['schema'].get('group_peer_semantics', 'excluded')}
        if grouped and manifest['schema'].get('group_peer_semantics') == CAUSAL:
            lineage['group_metadata_sha256'] = json.loads(
                Path(manifest['native_provenance_file']['path']).read_text())[
                    'causal_sources']['membership_sha256']
        forecast = predict_route(artifact, {'X': X, 'lineage': lineage, 'rejections': []})
        singleton = reference(np.asarray([i]))
        singleton_p = project_monotone(old.apply_cal(singleton, params)).reshape(1, 9)
        actual_raw = np.asarray([forecast['raw'][name] for name in TARGETS])
        actual_p = np.asarray([forecast['processed'][name] for name in TARGETS])
        results.append({'sample_id': sample_id,
                        'raw_max_error': float(np.max(np.abs(actual_raw - singleton[0]))),
                        'processed_max_error': float(np.max(np.abs(actual_p - singleton_p[0]))),
                        'g2_eligible': forecast['quality']['g2_eligible'],
                        'publication_eligible': forecast['quality']['publication_eligible']})
    passed = (eval_raw_error <= 1e-7 and eval_p_error <= 1e-7 and
              all(x['raw_max_error'] <= 1e-7 and x['processed_max_error'] <= 1e-7 and
                  not x['g2_eligible'] and not x['publication_eligible'] for x in results))
    out = {'route': route, 'eval_rows': len(eval_keys),
           'eval_keys_sha256': digest(eval_keys), 'eval_raw_max_error': eval_raw_error,
           'eval_processed_max_error': eval_p_error, 'singleton': results,
           'pass': passed, 'seconds': time.monotonic() - started,
           'evidence_scope': 'historical_static_tensor_engineering_only'}
    write_new(Path(output) / 'child_result.json', out)
    if not passed:
        raise ValueError('formal child route comparison exceeded 1e-7')


def verify_in_child(artifact_path, completed_run, selection_path, new_output):
    """Run one immutable formal-interface check in a clean, single-worker child."""
    output = Path(new_output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1'})
    cmd = [sys.executable, '-m',
           'research.after_open_3d5pct.precision_v9.native_artifact',
           'child_verify', '--artifact', str(artifact_path), '--run', str(completed_run),
           '--selection', str(selection_path), '--output', str(output)]
    cwd = Path(__file__).resolve().parents[3]
    invocation = {'command': cmd, 'cwd': str(cwd),
                  'python': sys.version, 'platform': platform.platform(),
                  'thread_env': {name: env[name] for name in
                                 ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                  'started_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        artifact_path = Path(artifact_path).resolve(strict=True)
        completed_run = Path(completed_run).resolve(strict=True)
        selection_path = Path(selection_path).resolve(strict=True)
        manifest = json.loads(artifact_path.read_text())
        provenance = json.loads(Path(manifest['native_provenance_file']['path']).read_text())
        if completed_run != Path(provenance['registration']['path']).parent:
            raise ValueError('child reference run differs from native provenance')
        before = [record(artifact_path), record(selection_path)] + verify_native_manifest(manifest)
        invocation['source_before'] = before
        invocation['launch_state'] = 'preflight_passed'
        write_new(output / 'invocation.json', invocation)
    except Exception as exc:
        invocation['launch_state'] = 'not_started_preflight_failed'
        invocation['preflight_error'] = {'error_type': type(exc).__name__, 'error': str(exc)}
        write_new(output / 'invocation.json', invocation)
        write_new(output / 'child_final_status.json',
                  {'pass': False, 'stage': 'preflight', 'exit_code': None,
                   'fatal': invocation['preflight_error'], 'source_drift': None})
        raise
    started = time.monotonic()
    try:
        result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                                text=True, timeout=1200, check=False)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        code, stdout, stderr = -1, str(exc.stdout or ''), str(exc.stderr or '')
        stderr += '\nchild timeout after 1200s'
    except Exception as exc:
        code, stdout, stderr = -2, '', f'{type(exc).__name__}: {exc}'
    (output / 'stdout.txt').write_text(stdout)
    (output / 'stderr.txt').write_text(stderr)
    after = _before_after(before)
    drift = [x['path'] for x, y in zip(before, after) if x != y]
    final = {'exit_code': code, 'seconds': time.monotonic() - started,
             'source_after': after, 'source_drift': drift,
             'stage': 'child_exited' if code >= 0 else 'child_launch_or_timeout_failed',
             'stdout_sha256': sha(output / 'stdout.txt'),
             'stderr_sha256': sha(output / 'stderr.txt'),
             'pass': code == 0 and not drift and (output / 'child_result.json').exists() and
                     json.loads((output / 'child_result.json').read_text()).get('pass') is True}
    write_new(output / 'child_final_status.json', final)
    if not final['pass']:
        raise ValueError('isolated child verification failed; inspect retained stdout/stderr/status')
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description='R03 native artifact offline checks only')
    sub = parser.add_subparsers(dest='command', required=True)
    child = sub.add_parser('child_verify')
    child.add_argument('--artifact', type=Path, required=True)
    child.add_argument('--run', type=Path, required=True)
    child.add_argument('--selection', type=Path, required=True)
    child.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'child_verify':
        # The child has imported this module but no torch. _run_child first
        # imports inference; its own torch import follows that gate.
        _run_child(args.artifact, args.run, args.selection, args.output)


if __name__ == '__main__':
    main()
