"""Isolated, offline numeric audit of the registered R03M/v8 evidence stage.

This is deliberately not a production route loader or certificate authority.
LightGBM must initialize before torch on the pinned macOS runtime.
"""
from __future__ import annotations

import lightgbm as lgb

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
import scipy.special
import torch

from ...focus_v8 import run as original
from ...focus_v8.core import TARGETS, mature_prior, path_summary, splits
from ...focus_v8.support import build_c
from ...iterations_v7.b_group.experiment import curve_features, group_relative_features
from ...train_multiscale_v6 import project_monotone, tabular
from .. import data as v9data
from .features import BAR_COLUMNS, build_features_asof
from .ledger import canonical, file_sha
from .test_features import CAUSAL_GROUP_PEERS, LEGACY_GROUP_PEERS, schema

HERE = Path(__file__).resolve().parents[2]
V8 = HERE / 'runs/focus_v8_20260927'
R03M = HERE / 'runs/precision_v9_20260927/R03M_group_only_v1_retry1'
REGISTRATION = HERE / 'precision_v9/rounds/R03M/CERTIFICATE_STAGE_REGISTRATION_DRAFT.md'
REGISTRATION_SHA = 'f58b9611ae5c5a3220c2d9fae9bf52098685574241073b11d2dd17b2d5d0c688'
RETRY_REGISTRATION = HERE/'precision_v9/rounds/R03M/NUMERIC_FAILURE_REVIEW_AND_RETRY_REGISTRATION.md'
RETRY_REGISTRATION_SHA = '008709af95dc6e8ee556eb6f5b6fca21d23c4a112d9664eae4b55fdcfbac9db5'
RETRY_ADOPTION = HERE/'precision_v9/rounds/R03M/NUMERIC_RETRY_ADOPTION.md'
RETRY_ADOPTION_SHA = '382b393cdf1665f439cc3213822a9d113672000bf037932449399589f54a32d1'
DIAGNOSTIC = HERE/'runs/precision_v9_20260927/R03M_certificate_batch_diagnostic_v1/diagnostic.json'
DIAGNOSTIC_SHA = 'f91859f5be6ac22e7669e642f6323d91cf0856520390e26664a0c127b2719145'
ROUTES = ('B_group', 'B_no_group', 'C_group', 'C_no_group', 'C_no_daily')
FOLDS = ('dev1', 'dev2', 'selected_3566')
EXPECTED_ROWS = 14424
TOLERANCE = 1e-7
ARRAYS = ('x5', 'x60', 'xday', 'group_seq', 'y')
REPRESENTATIVES = (('AMD', '2026-03-02'), ('AMD', '2026-06-01'),
                   ('AMD', '2026-07-10'), ('AMD', '2026-07-13'),
                   ('AMD', '2026-09-17'), ('COHR', '2026-06-01'))
SOURCE_CODE = (HERE / 'focus_v8/run.py', HERE / 'focus_v8/core.py',
               HERE / 'focus_v8/support.py', HERE / 'focus_v8/prepare.py',
               HERE / 'focus_v8/test_core.py',
               HERE / 'focus_v8/protocol.json', HERE / 'train_multiscale_v6.py',
               HERE / 'v6_data.py', HERE / 'precision_v9/data.py',
               HERE / 'timeaxis.py',
               HERE / 'iterations_v7/b_group/experiment.py',
               HERE / 'iterations_v7/c_no_daily/experiment.py',
               HERE / 'precision_v9/forward/features.py',
               HERE / 'precision_v9/forward/inference.py',
               HERE / 'precision_v9/forward/ledger.py',
               HERE / 'precision_v9/forward/numeric_equivalence.py',
               HERE / 'precision_v9/forward/test_numeric_equivalence.py',
               HERE / 'precision_v9/forward/test_features.py')


def _sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _write_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _progress(output, event, **details):
    row = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
           'event': event, **details}
    with (Path(output) / 'progress.jsonl').open('a') as stream:
        stream.write(canonical(row) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def registered_bundles(registration=REGISTRATION, root=V8):
    """Parse the adopted frozen table; no inferred incumbents or fold selection."""
    if file_sha(registration) != REGISTRATION_SHA:
        raise ValueError('adopted registration bytes changed')
    text = Path(registration).read_text()
    found = re.findall(r'^\| (B_group|B_no_group|C_group|C_no_group|C_no_daily) '
                       r'(dev1|dev2|selected_3566) \| ([^|]+) \| `([0-9a-f]{64})` \|',
                       text, re.M)
    if [(r, f) for r, f, _, _ in found] != [(r, f) for r in ROUTES for f in FOLDS]:
        raise ValueError('registration omits/reorders a frozen route or fold')
    records = []
    for route, fold, relative, expected in found:
        path = (Path(root) / relative.strip()).resolve(strict=True)
        names = ([f'target_{j}.txt' for j in range(9)] if route.startswith('B')
                 else ['model.pt'])
        names += ['calibration.json', 'registration.json', 'result.json', 'eval.parquet']
        files = [{'file': name, 'path': str(path/name), 'sha256': file_sha(path/name)}
                 for name in names]
        actual = _sha([{'file': item['file'], 'sha256': item['sha256']}
                       for item in files])
        if actual != expected:
            raise ValueError(f'{route}/{fold}: registered bundle hash mismatch')
        saved = json.loads((path/'registration.json').read_text())
        if saved['route'] != route or saved['fold']['id'] != ('reserved' if fold ==
                'selected_3566' else fold) or saved['seed'] != 3566:
            raise ValueError(f'{route}/{fold}: checkpoint registration differs')
        records.append({'route': route, 'fold': fold, 'path': str(path),
                        'bundle_sha256': actual, 'files': files,
                        'recipe': saved['recipe'], 'fold_contract': saved['fold'],
                        'split_rows': saved['split_rows']})
    return records


def assert_complete_inventory(bundles):
    if ([(x['route'], x['fold']) for x in bundles] != [
            (route, fold) for route in ROUTES for fold in FOLDS] or
            bundles != registered_bundles()):
        raise ValueError('inventory differs from complete adopted 15-bundle registration')


def source_paths(bundles):
    paths = [REGISTRATION, RETRY_REGISTRATION, RETRY_ADOPTION, DIAGNOSTIC,
             V8/'round_10/summary.json',
             V8/'config.json',
             V8/'dataset/features.npz', V8/'dataset/rows.parquet',
             V8/'dataset/manifest.json', R03M/'dataset/features.npz',
             R03M/'dataset/rows.parquet', R03M/'dataset/manifest.json',
             R03M/'group_metadata/versions.json',
             R03M/'group_metadata/memberships.json', R03M/'result.json',
             V8/'path_summary.npy', V8/'path_summary_identity.json', *SOURCE_CODE]
    paths.extend(Path(item['path']) for bundle in bundles for item in bundle['files'])
    paths.extend(_representative_source_paths())
    frozen = json.loads((R03M/'inputs_before_sha256.json').read_text())['manifest_sources']
    paths.extend(V8/'inputs'/relative for relative in frozen)
    paths.append(R03M/'inputs_before_sha256.json')
    return sorted(set(p.resolve(strict=True) for p in paths), key=str)


def _representative_source_paths():
    candidates, universe_file, config = _candidate_universe()
    versions_path = R03M/'group_metadata/versions.json'
    members_path = R03M/'group_metadata/memberships.json'
    versions = json.loads(versions_path.read_text())
    memberships = json.loads(members_path.read_text())
    calendar_file = V8/'inputs/market_data/calendars/nasdaq_sessions_2026_v1.json'
    calendar = json.loads(calendar_file.read_text())['sessions']
    market_root = V8/'inputs'/config['source_dir']
    paths = {universe_file, calendar_file}
    for symbol, day in REPRESENTATIVES:
        sessions = [s for s in calendar if '2026-01-01' <= s['session_date'] <= day]
        if not sessions or sessions[-1]['session_date'] != day:
            raise ValueError('representative day outside frozen calendar')
        deadline = pd.Timestamp(sessions[-1]['open_at']) + pd.Timedelta(hours=2, seconds=30)
        dependency, _, _ = _representative_dependency(
            symbol, sessions, versions, memberships, candidates, deadline)
        paths.update(market_root/dep/'2026.parquet' for dep in dependency)
        paths.add(market_root.parent/'corporate_actions'/f'{symbol}.parquet')
    return paths


def hashes(paths):
    return [{'path': str(path), 'sha256': file_sha(path), 'size': path.stat().st_size}
            for path in paths]


def verify_unchanged(before):
    after = hashes([Path(x['path']) for x in before])
    changed = [a['path'] for a, b in zip(before, after) if a != b]
    return after, changed


def inventory(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for path, digest in ((RETRY_REGISTRATION, RETRY_REGISTRATION_SHA),
                             (RETRY_ADOPTION, RETRY_ADOPTION_SHA),
                             (DIAGNOSTIC, DIAGNOSTIC_SHA)):
            if file_sha(path) != digest:
                raise ValueError(f'{path}: adopted retry evidence changed')
        bundles = registered_bundles()
        summary_path = V8/'round_10/summary.json'
        if file_sha(summary_path) != 'ed1b7801e31c2aa5e3b616e521ae0a44e4d70445c2f2170798f8ebfa84639c61':
            raise ValueError('frozen round-10 recipe summary changed')
        summary = json.loads(summary_path.read_text())
        for bundle in bundles:
            if bundle['recipe'] != summary[bundle['route']]['incumbent_recipe']:
                raise ValueError(f"{bundle['route']}: recipe differs from incumbent")
        expected = {'features.npz': '6bce8471011f1c2c54ce2551ba7584f18196fb5b056580e528d95532d5c0e744',
                    'rows.parquet': 'b6220c3645ff6477ff59763d75deaff26b6f60c7b1ad34f6d0b2b913a4fed2fe'}
        for directory in (V8/'dataset', R03M/'dataset'):
            for name, digest in expected.items():
                if file_sha(directory/name) != digest:
                    raise ValueError(f'{directory/name}: frozen dataset changed')
        for name, digest in (('versions.json','1dd617f925d74024838f9cde19e58ca4296b0c5d74f64b74fec421eed9b46788'),
                             ('memberships.json','f1fc75227d83ab9d27caf96c47399279ca5457d6b8d547d703ef41ea8af5a740')):
            if file_sha(R03M/'group_metadata'/name) != digest:
                raise ValueError(f'new {name} differs from R03M adoption')
        for path, digest in ((HERE/'precision_v9/forward/features.py',
                               '528344d95bab167fb520584e854c0223e2ccba2b592b3fcaa5a2a9818229e646'),
                              (HERE/'precision_v9/forward/inference.py',
                               'e2a0667c5d44cc62c5760a10fa6d5571f7c236e78488d4b2f5d5cf349bb9ded3'),
                              (R03M/'result.json',
                               'd00e83c18a31ca26f65e40bc75d6c9a0f66ab401e96f97ddde49c3f50a8d88a5')):
            if file_sha(path) != digest:
                raise ValueError(f'{path}: adopted source hash differs')
        frozen = json.loads((R03M/'inputs_before_sha256.json').read_text())['manifest_sources']
        for relative, digest in frozen.items():
            if file_sha(V8/'inputs'/relative) != digest:
                raise ValueError(f'{relative}: original frozen source hash differs')
        _write_new(output/'inventory.json', bundles)
        declarations = {route: {'legacy': schema(route, group_peer_semantics=LEGACY_GROUP_PEERS),
                                'target_causal': schema(route, group_peer_semantics=CAUSAL_GROUP_PEERS),
                                'source': 'forward/test_features.py derived replay declaration; no native v8 schema file',
                                'source_sha256': file_sha(HERE/'precision_v9/forward/test_features.py')}
                        for route in ROUTES}
        _write_new(output/'schema_declarations.json', declarations)
        registered = output/'registration_snapshots'
        registered.mkdir()
        for source in (REGISTRATION, RETRY_REGISTRATION, RETRY_ADOPTION, DIAGNOSTIC):
            destination = registered/source.name
            shutil.copyfile(source, destination)
            if file_sha(destination) != file_sha(source):
                raise ValueError('adopted registration/diagnostic snapshot differs')
        schema_dir = output/'schema_snapshots'
        schema_dir.mkdir()
        for route, versions in declarations.items():
            for variant in ('legacy', 'target_causal'):
                _write_new(schema_dir/f'{route}_{variant}.json', versions[variant])
        code_dir = output/'source_code'
        code_dir.mkdir()
        for source in SOURCE_CODE:
            destination = code_dir/source.relative_to(HERE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        origin_dir = output/'source_origin_snapshots'
        origin_dir.mkdir()
        origin = []
        for relative, digest in (
                ('research/after_open_3d5pct/focus_v8/run.py',
                 '7f6189949f792b043c6c4a0dc4d0eedb6462717d9b9be3e4e6da5476627f8b11'),
                ('research/after_open_3d5pct/focus_v8/test_core.py',
                 '974ad8aea54e868ef3a64387982fc8f751b55c71d6eeb3e10f6c6395b3ab55b1')):
            completed = subprocess.run(['git','show',f'101d357:{relative}'],
                                       cwd=HERE.parents[1], capture_output=True, check=True)
            content = completed.stdout
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('historical Git source differs from old registration')
            dest = origin_dir/Path(relative).name
            with dest.open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            origin.append({'commit':'101d357','path':relative,'sha256':digest,
                           'snapshot_path':str(dest)})
        _write_new(origin_dir/'manifest.json', origin)
        model_dir = output/'model_snapshots'
        for bundle in bundles:
            destination = model_dir/bundle['route']/bundle['fold']
            destination.mkdir(parents=True)
            for item in bundle['files']:
                if item['file'].startswith('target_') or item['file'] in ('model.pt','calibration.json'):
                    shutil.copyfile(item['path'], destination/item['file'])
                    if file_sha(destination/item['file']) != item['sha256']:
                        raise ValueError('model snapshot bytes differ from original')
        before_paths = source_paths(bundles)
        before_paths.extend((output/'inventory.json', output/'schema_declarations.json'))
        before_paths.extend((output/'source_code').rglob('*'))
        before_paths.extend((output/'schema_snapshots').glob('*.json'))
        before_paths.extend((output/'model_snapshots').glob('*/*/*'))
        before_paths.extend((output/'source_origin_snapshots').glob('*'))
        before_paths.extend((output/'registration_snapshots').glob('*'))
        before = hashes(sorted((p for p in before_paths if p.is_file()), key=str))
        _write_new(output/'input_hashes_before.json', before)
        _write_new(output/'environment.json', {'python': platform.python_version(),
            'lightgbm': lgb.__version__, 'torch': torch.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__,
            'threads': {'torch': 1, 'lightgbm': 1, 'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS')}})
        _progress(output, 'inventory_pass', bundle_count=len(bundles),
                  source_count=len(before))
        return bundles, before
    except Exception as exc:
        _progress(output, 'inventory_failure', error_type=type(exc).__name__, error=str(exc))
        raise


def _data(directory):
    rows = pd.read_parquet(Path(directory)/'rows.parquet')
    with np.load(Path(directory)/'features.npz') as archive:
        arrays = {name: archive[name] for name in ARRAYS}
    if len(rows) != EXPECTED_ROWS or any(len(value) != EXPECTED_ROWS for value in arrays.values()):
        raise ValueError('dataset lacks complete 14424 old-key population')
    keys = rows['sample_id'].astype(str).tolist()
    if len(set(keys)) != EXPECTED_ROWS:
        raise ValueError('dataset has duplicate/missing sample IDs')
    return rows, arrays


def _diff(left, right, keys):
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        return {'shape_equal': False, 'old_shape': list(a.shape), 'new_shape': list(b.shape),
                'pass': False}
    mask = np.isnan(a) | np.isnan(b)
    mask_equal = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    delta = np.abs(np.where(mask, 0, a-b))
    maximum = float(np.max(delta)) if delta.size else 0.
    location = np.unravel_index(int(np.argmax(delta)), delta.shape) if delta.size else (0,)
    return {'shape_equal': True, 'shape': list(a.shape), 'nan_mask_equal': mask_equal,
            'max_abs_error': maximum, 'worst_sample_id': keys[location[0]],
            'worst_index': [int(i) for i in location],
            'pass': mask_equal and np.isfinite(maximum) and maximum <= TOLERANCE}


def _probability_array(value, expected_rows):
    array = np.asarray(value)
    if array.shape != (expected_rows, 9) or not np.isfinite(array).all() or \
            np.any((array < 0) | (array > 1)):
        raise ValueError('nine-target probability output is missing, nonfinite, or out of range')
    return array


def _transforms(data, rows):
    y = data['y'].reshape(-1, 9)
    prior, counts = mature_prior(rows, y)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result = {'prior': prior, 'prior_counts': counts,
                  'xbase_group': tabular(data, group=True).to_numpy(np.float32),
                  'xbase_no_group': tabular(data, group=False).to_numpy(np.float32),
                  'paths': path_summary(data),
                  'curve': curve_features(data).to_numpy(np.float32),
                  'relative': group_relative_features(data).to_numpy(np.float32)}
    return result


def compare_inputs(output):
    old_rows, old_data = _data(V8/'dataset')
    new_rows, new_data = _data(R03M/'dataset')
    if not old_rows.equals(new_rows):
        raise ValueError('old/new rows differ before numeric prediction')
    keys = old_rows.sample_id.astype(str).tolist()
    differences = {name: _diff(old_data[name], new_data[name], keys) for name in ARRAYS}
    if not all(item['pass'] for item in differences.values()):
        raise ValueError('old/new raw input arrays differ')
    old_transforms = _transforms(old_data, old_rows)
    new_transforms = _transforms(new_data, new_rows)
    differences.update({name: _diff(old_transforms[name], new_transforms[name], keys)
                        for name in old_transforms})
    _write_new(Path(output)/'input_differences.json', differences)
    if not all(item['pass'] for item in differences.values()):
        raise ValueError('old/new derived model inputs differ')
    _progress(output, 'all_inputs_pass', count=len(keys),
              key_order_sha256=_sha(keys), names=list(differences))
    return old_rows, old_data, new_data, old_transforms, new_transforms, keys


def _predict_offline(bundle, data, transformed, ids=None):
    """Original checkpoint math, without touching the forward loader/lineage."""
    route, path, recipe = bundle['route'], Path(bundle['path']), bundle['recipe']
    ids = np.arange(len(transformed['prior'])) if ids is None else np.asarray(ids)
    n = len(ids)
    if route.startswith('B'):
        x = (transformed['xbase_group'] if route == 'B_group'
             else transformed['xbase_no_group'])
        if recipe.get('representation'):
            x = np.column_stack((x, transformed['paths']))
        if recipe.get('curves'):
            x = np.column_stack((x, transformed['curve']))
            if route == 'B_group':
                x = np.column_stack((x, transformed['relative']))
        x = x[ids]
        raw = np.empty((n, 9), np.float64)
        for j in range(9):
            model = lgb.Booster(model_file=str(path/f'target_{j}.txt'))
            z = model.predict(x, raw_score=True, num_threads=1)
            if recipe.get('prior'):
                z += scipy.special.logit(np.clip(transformed['prior'][ids, j], .01, .99))
            raw[:, j] = scipy.special.expit(z)
    else:
        torch.set_num_threads(1)
        saved = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
        if saved.get('route') != route or saved.get('recipe') != recipe:
            raise ValueError(f'{route}: checkpoint route/recipe differs')
        model = build_c(route, recipe)
        model.load_state_dict(saved['state_dict'], strict=True)
        arrays = _c_input_arrays(route, data, transformed['prior'])
        raw = original.predict_c(model, arrays, ids)
    calibration = json.loads((path/'calibration.json').read_text())
    if calibration.get('method') != 'shrunken_nonnegative_platt' or \
            np.asarray(calibration.get('parameters')).shape != (9, 2):
        raise ValueError(f'{route}: frozen calibration contract differs')
    processed = project_monotone(original.apply_cal(raw, calibration['parameters'])).reshape(-1, 9)
    return raw, processed


def _c_input_arrays(route, data, prior):
    values = []
    for name in ('x5', 'x60', 'xday', 'group_seq'):
        if (name == 'xday' and route == 'C_no_daily') or \
                (name == 'group_seq' and route == 'C_no_group'):
            values.append(torch.empty(0))
        else:
            values.append(torch.from_numpy(np.asarray(data[name], np.float32)))
    values.append(torch.from_numpy(np.asarray(prior, np.float32)))
    return tuple(values)


def _bundle_reference(bundle, rows, keys, old_data, old_transforms,
                      raw_old, processed_old):
    path = Path(bundle['path'])
    reference = pd.read_parquet(path/'eval.parquet')
    expected = splits(rows, bundle['fold_contract'])['eval']
    expected_keys = [keys[i] for i in expected]
    actual_keys = reference.sample_id.astype(str).tolist()
    if actual_keys != expected_keys or len(actual_keys) != bundle['split_rows']['eval']:
        raise ValueError(f"{bundle['route']}/{bundle['fold']}: eval key/order mismatch")
    actual_raw = _probability_array(
        reference[[f'raw_{t}' for t in TARGETS]].to_numpy(np.float64), len(expected))
    actual_processed = _probability_array(
        reference[[f'p_{t}' for t in TARGETS]].to_numpy(np.float64), len(expected))
    replay_raw, replay_processed = _predict_offline(bundle, old_data,
                                                    old_transforms, expected)
    _probability_array(replay_raw, len(expected))
    _probability_array(replay_processed, len(expected))
    return {'eval_rows': len(expected),
            'raw': _diff(replay_raw, actual_raw, expected_keys),
            'processed': _diff(replay_processed, actual_processed, expected_keys),
            'full_batch_eval_slice_raw': _diff(raw_old[expected], actual_raw, expected_keys),
            'full_batch_eval_slice_processed': _diff(processed_old[expected], actual_processed,
                                                    expected_keys),
            'replay_batch_count': int(np.ceil(len(expected)/256)),
            'eval_keys_sha256': _sha(expected_keys)}


def compare_bundle(output, bundle, rows, old_data, new_data,
                   old_transforms, new_transforms, keys):
    started = time.perf_counter()
    raw_old, processed_old = _predict_offline(bundle, old_data, old_transforms)
    raw_new, processed_new = _predict_offline(bundle, new_data, new_transforms)
    for item in (raw_old, processed_old, raw_new, processed_new):
        _probability_array(item, len(keys))
    raw_diff = _diff(raw_old, raw_new, keys)
    processed_diff = _diff(processed_old, processed_new, keys)
    reference = _bundle_reference(bundle, rows, keys, old_data, old_transforms,
                                  raw_old, processed_old)
    frame = pd.DataFrame({'sample_id': keys, 'symbol': rows.symbol.to_numpy(),
                          'session_date': rows.session_date.to_numpy()})
    for j, target in enumerate(TARGETS):
        frame[f'raw_old_{target}'] = raw_old[:, j]
        frame[f'raw_new_{target}'] = raw_new[:, j]
        frame[f'raw_abs_error_{target}'] = np.abs(raw_old[:, j]-raw_new[:, j])
        frame[f'p_old_{target}'] = processed_old[:, j]
        frame[f'p_new_{target}'] = processed_new[:, j]
        frame[f'p_abs_error_{target}'] = np.abs(processed_old[:, j]-processed_new[:, j])
    destination = Path(output)/f"{bundle['route']}_{bundle['fold']}_all_keys.parquet"
    frame.to_parquet(destination, index=False)
    result = {'route': bundle['route'], 'fold': bundle['fold'], 'n': len(keys),
              'targets': list(TARGETS), 'key_order_sha256': _sha(keys),
              'old_new_raw': raw_diff, 'old_new_processed': processed_diff,
              'frozen_eval_reference': reference,
              'row_output_path': destination.name,
              'row_output_sha256': file_sha(destination),
              'seconds': time.perf_counter()-started}
    result['pass'] = (raw_diff['pass'] and processed_diff['pass'] and
                      reference['raw']['pass'] and reference['processed']['pass'])
    _write_new(Path(output)/f"{bundle['route']}_{bundle['fold']}_summary.json", result)
    _progress(output, 'bundle_complete', route=bundle['route'], fold=bundle['fold'],
              passed=result['pass'], seconds=result['seconds'],
              max_raw=raw_diff['max_abs_error'], max_processed=processed_diff['max_abs_error'],
              eval_max_raw=reference['raw']['max_abs_error'],
              eval_max_processed=reference['processed']['max_abs_error'])
    return result


def run_all_keys(output):
    output = Path(output)
    before = json.loads((output/'input_hashes_before.json').read_text())
    try:
        bundles = json.loads((output/'inventory.json').read_text())
        assert_complete_inventory(bundles)
        _, changed = verify_unchanged(before)
        if changed:
            raise ValueError(f'inventory sources changed before run: {changed[:5]}')
        rows, old_data, new_data, old_transforms, new_transforms, keys = compare_inputs(output)
        results = []
        for bundle in bundles:
            try:
                results.append(compare_bundle(output, bundle, rows, old_data, new_data,
                                              old_transforms, new_transforms, keys))
            except Exception as exc:
                failure = {'route': bundle['route'], 'fold': bundle['fold'],
                           'error_type': type(exc).__name__, 'error': str(exc)}
                _progress(output, 'bundle_failure', **failure)
                results.append({'route': bundle['route'], 'fold': bundle['fold'],
                                'pass': False, 'failure': failure})
        after, changed = verify_unchanged(before)
        _write_new(output/'all_keys_summary.json', {'bundles': results,
                   'bundle_count': len(results), 'passed': sum(bool(x['pass']) for x in results),
                   'changed_sources': changed,
                   'all_key_pass': len(results) == 15 and all(x['pass'] for x in results)
                                   and not changed})
        _progress(output, 'all_keys_complete', passed=sum(bool(x['pass']) for x in results),
                  changed_sources=len(changed))
        return results
    except Exception as exc:
        _write_new(output/'all_keys_fatal.json',
                   {'error_type':type(exc).__name__, 'error':str(exc)})
        _progress(output, 'all_keys_fatal', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        after, changed = verify_unchanged(before)
        _write_new(output/'input_hashes_after.json', after)
        if changed:
            _progress(output, 'source_drift', stage='all_keys', paths=changed[:16],
                      count=len(changed))


def _candidate_universe():
    config = json.loads((V8/'config.json').read_text())
    universe_file = V8/'inputs'/config['universe']
    document = json.loads(universe_file.read_text())
    candidates = {m['symbol'] for m in document['members'] if m['role'] == 'candidate'}
    if len(candidates) != 108 or 'QQQ' in candidates:
        raise ValueError('frozen 108 candidate roles or separate QQQ benchmark changed')
    return candidates, universe_file, config


def _representative_dependency(symbol, sessions, versions, memberships, candidates, deadline):
    representatives = {}
    for ix in range(len(sessions)-1, -1, -1):
        iso = pd.Timestamp(sessions[ix]['session_date']).isocalendar()
        week = f'{iso.year}-W{iso.week:02d}'
        representatives.setdefault(week, ix)
        if len(representatives) == 6:
            break
    if len(representatives) != 6:
        raise ValueError('six actual representative weeks unavailable')
    allowed = {v['version_id'] for v in versions
               if v['strategy_id'] in v9data.STRATEGIES and
               pd.Timestamp(v['feature_cutoff_at']) <= deadline and
               pd.Timestamp(v['effective_from']) <= deadline}
    own_groups = {(m['week_id'], m['strategy_id'], m['group_ids'][0])
                  for m in memberships if m['symbol'] == symbol and
                  m['week_id'] in representatives and
                  m['strategy_id'] in v9data.STRATEGIES and
                  len(m['group_ids']) == 1 and m['version_id'] in allowed}
    peers = {m['symbol'] for m in memberships if m['symbol'] in candidates and
             m['strategy_id'] in v9data.STRATEGIES and len(m['group_ids']) == 1 and
             (m['week_id'], m['strategy_id'], m['group_ids'][0]) in own_groups and
             m['version_id'] in allowed}
    dependency = sorted(peers | {symbol, 'QQQ'})
    first = min(representatives.values())
    peer_start = sessions[max(0, first-20)]['session_date']
    return dependency, peer_start, sorted(representatives)


def _split_days(path):
    actions = pd.read_parquet(path)
    ratio = pd.to_numeric(actions.get('split_ratio', pd.Series(np.nan, index=actions.index)),
                          errors='coerce')
    base = pd.to_numeric(actions.get('split_base', pd.Series(np.nan, index=actions.index)),
                         errors='coerce')
    return sorted(actions.loc[((ratio.notna()) & (ratio != 0) & (ratio != 1)) |
                              ((base.notna()) & (ratio.notna()) & (base != ratio)),
                              'ex_div_date'].astype(str))


def _mature_store(symbol, index, rows, labels, deadline):
    own = np.flatnonzero(rows.symbol.to_numpy() == symbol)
    eligible = [i for i in own if i != index and
                pd.Timestamp(rows.iloc[i].decision_at) < deadline and
                pd.Timestamp(rows.iloc[i].label_end_at) < deadline and
                pd.Timestamp(rows.iloc[i].label_available_at) < deadline]
    store = [{'sample_id': rows.iloc[i].sample_id, 'symbol': symbol,
              'decision_at': rows.iloc[i].decision_at,
              'label_end_at': rows.iloc[i].label_end_at,
              'label_available_at': rows.iloc[i].label_available_at,
              'y': labels[i].tolist()} for i in eligible]
    if any('entry_price' in item for item in store):
        raise ValueError('future entry leaked into mature prior store')
    return store


def _snapshot(symbol, day, candidates, config, versions, memberships):
    calendar_file = V8/'inputs/market_data/calendars/nasdaq_sessions_2026_v1.json'
    calendar = json.loads(calendar_file.read_text())['sessions']
    sessions = [s for s in calendar if '2026-01-01' <= s['session_date'] <= day]
    if not sessions or sessions[-1]['session_date'] != day:
        raise ValueError('representative day absent from frozen calendar')
    opening = pd.Timestamp(sessions[-1]['open_at'])
    cutoff = opening+pd.Timedelta(hours=2)
    deadline = cutoff+pd.Timedelta(seconds=30)
    dependency, peer_start, weeks = _representative_dependency(
        symbol, sessions, versions, memberships, candidates, deadline)
    root = V8/'inputs'/config['source_dir']
    columns = sorted(BAR_COLUMNS)
    bars, paths = {}, [calendar_file]
    for dep in dependency:
        path = root/dep/'2026.parquet'
        start = '2026-01-01' if dep == symbol else peer_start
        frame = pd.read_parquet(path, columns=columns,
                                filters=[('session_date','>=',start),
                                         ('session_date','<=',day)])
        # Preserve every already-known prior-session slot for daily/hourly
        # denominators. Only the representative day's future is removed.
        visible = ((pd.to_datetime(frame.end_at, utc=True) <= cutoff.tz_convert('UTC')) &
                   (pd.to_datetime(frame.available_at, utc=True) <= deadline.tz_convert('UTC')))
        bars[dep] = frame.loc[visible].copy()
        paths.append(path)
    action_path = root.parent/'corporate_actions'/f'{symbol}.parquet'
    paths.append(action_path)
    decision = {'symbol': symbol, 'session_date': day,
                'cutoff_at': cutoff.isoformat(), 'decision_at': cutoff.isoformat(),
                'information_deadline_at': deadline.isoformat()}
    metadata = {'split_days': {symbol: _split_days(action_path)},
                'versions': versions, 'memberships': memberships,
                'universe_symbols': sorted(candidates),
                'group_peer_semantics': CAUSAL_GROUP_PEERS,
                'membership_snapshot_sha256': file_sha(R03M/'group_metadata/memberships.json'),
                'mature_prior_scope': 'all_candidates'}
    snapshot = {'bars': bars, 'sessions': sessions, 'mode': 'historical_fixture'}
    detail = {'dependency_symbols': dependency, 'peer_prefix_start': peer_start,
              'representative_weeks': weeks,
              'source_paths': sorted(set(str(p.resolve(strict=True)) for p in paths))}
    return snapshot, decision, metadata, detail


def _one_feature_prediction(bundle, built):
    route = bundle['route']
    X = built['X']
    empty = np.empty((1, 0), np.float32)
    data = {name: np.asarray(X[name], np.float32)[None] if name in X else empty
            for name in ('x5','x60','xday','group_seq')}
    transformed = {'prior': np.asarray(X['prior'], np.float32)[None],
                   'xbase_group': np.asarray(X['xbase'], np.float32)[None],
                   'xbase_no_group': np.asarray(X['xbase'], np.float32)[None],
                   'paths': np.asarray(X['paths'], np.float32)[None],
                   'curve': np.asarray(X['curve'], np.float32)[None],
                   'relative': np.asarray(X['relative'], np.float32)[None]}
    return _predict_offline(bundle, data, transformed)


def _run_representatives_inner(output):
    output = Path(output)
    before_global = json.loads((output/'input_hashes_before.json').read_text())
    _, changed_global = verify_unchanged(before_global)
    if changed_global:
        raise ValueError(f'representative inputs changed since inventory: {changed_global[:5]}')
    inventory_rows = json.loads((output/'inventory.json').read_text())
    assert_complete_inventory(inventory_rows)
    all_keys = json.loads((output/'all_keys_summary.json').read_text())
    bundles = all_keys['bundles']
    if (not all_keys['all_key_pass'] or
            [(x['route'],x['fold']) for x in bundles] != [
                (r,f) for r in ROUTES for f in FOLDS] or
            any(not x['pass'] or x['n'] != EXPECTED_ROWS or
                file_sha(output/x['row_output_path']) != x['row_output_sha256']
                for x in bundles)):
        raise ValueError('all-key numeric stage did not pass; retain failure')
    bundles = inventory_rows
    candidates, universe_file, config = _candidate_universe()
    versions_path = R03M/'group_metadata/versions.json'
    members_path = R03M/'group_metadata/memberships.json'
    versions = json.loads(versions_path.read_text())
    memberships = json.loads(members_path.read_text())
    rows, data = _data(V8/'dataset')
    labels = data['y'].reshape(-1, 9)
    prior, _ = mature_prior(rows, labels)
    paths = {universe_file, versions_path, members_path}
    frames = {}
    for symbol, day in REPRESENTATIVES:
        snapshot, decision, metadata, detail = _snapshot(
            symbol, day, candidates, config, versions, memberships)
        frames[(symbol, day)] = (snapshot, decision, metadata, detail)
        paths.update(Path(name) for name in detail['source_paths'])
    before = hashes(sorted((p.resolve(strict=True) for p in paths), key=str))
    _write_new(output/'representative_sources_before.json', before)
    records = []
    for symbol, day in REPRESENTATIVES:
        snapshot, decision, metadata, detail = frames[(symbol, day)]
        matches = np.flatnonzero((rows.symbol == symbol).to_numpy() &
                                 (rows.session_date == day).to_numpy())
        if len(matches) != 1:
            raise ValueError(f'{symbol}/{day}: frozen key missing or duplicated')
        ix = int(matches[0])
        key = str(rows.iloc[ix].sample_id)
        deadline = pd.Timestamp(decision['information_deadline_at'])
        store = _mature_store(symbol, ix, rows, labels, deadline)
        sample_record = {'symbol': symbol, 'session_date': day, 'sample_id': key,
                         'row_index': ix, 'mature_prior_rows': len(store),
                         'mature_prior_last_63_ids': [item['sample_id'] for item in store[-63:]],
                         'dependency': detail, 'routes': []}
        if symbol == 'AMD' and day == '2026-06-01' and len(store) != 58:
            raise ValueError('AMD June 1 mature-prior count differs from frozen 58')
        for route in ROUTES:
            route_bundles = [item for item in bundles if item['route'] == route]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                built = build_features_asof(snapshot, decision,
                    schema(route, group_peer_semantics=CAUSAL_GROUP_PEERS), metadata, store)
            route_record = {'route': route, 'rejections': built['rejections'],
                            'lineage': built.get('lineage', {}), 'inputs': {},
                            'folds': []}
            sample_record['routes'].append(route_record)
            if built['rejections'] or built['X'] is None:
                _progress(output, 'representative_rejected', symbol=symbol,
                          session_date=day, route=route, reasons=built['rejections'])
                continue
            X = built['X']
            if route in ('B_group','C_group','C_no_daily') and \
                    X.get('group_seq') is None:
                raise ValueError('group route omitted group_seq')
            if sorted(built['lineage'].get('used_market_symbols', [])) != \
                    detail['dependency_symbols'] and route in ('B_group','C_group','C_no_daily'):
                raise ValueError(f'{symbol}/{day}/{route}: peer dependency changed')
            expected = {name: data[name][ix] for name in ('x5','x60','xday','group_seq')}
            expected['prior'] = prior[ix]
            singleton = {name: data[name][ix:ix+1]
                         for name in ('x5','x60','xday','group_seq')}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                expected['xbase'] = tabular(singleton,
                    group=route in ('B_group','C_group','C_no_daily')).to_numpy(np.float32)[0]
                expected['paths'] = path_summary(singleton)[0]
                expected['curve'] = curve_features(singleton).to_numpy(np.float32)[0]
                expected['relative'] = (group_relative_features(singleton).to_numpy(np.float32)[0]
                                        if route == 'B_group' else np.empty(0, np.float32))
            for name, observed in X.items():
                route_record['inputs'][name] = _diff(np.asarray(observed)[None],
                                                     np.asarray(expected[name])[None], [key])
            if not all(d['pass'] for d in route_record['inputs'].values()):
                _progress(output, 'representative_input_difference', symbol=symbol,
                          session_date=day, route=route)
            old_singleton = {'X': {name: expected[name] for name in X},
                             'lineage': {'mode': 'historical_fixture'}, 'rejections': []}
            for bundle in route_bundles:
                try:
                    old_raw, old_processed = _one_feature_prediction(bundle, old_singleton)
                    raw, processed = _one_feature_prediction(bundle, built)
                    _probability_array(old_raw, 1)
                    _probability_array(old_processed, 1)
                    _probability_array(raw, 1)
                    _probability_array(processed, 1)
                    full = pd.read_parquet(output/f"{route}_{bundle['fold']}_all_keys.parquet",
                                           filters=[('sample_id','==',key)])
                    if len(full) != 1:
                        raise ValueError('full-key reference row absent or duplicated')
                    full_raw = full[[f'raw_old_{t}' for t in TARGETS]].to_numpy()
                    full_processed = full[[f'p_old_{t}' for t in TARGETS]].to_numpy()
                    comparisons = {'raw': _diff(raw, old_raw, [key]),
                                   'processed': _diff(processed, old_processed, [key])}
                    route_record['folds'].append({'fold': bundle['fold'],
                        'comparison': comparisons,
                        'cross_batch_diagnostic': {
                            'raw': _diff(old_raw, full_raw, [key]),
                            'processed': _diff(old_processed, full_processed, [key])},
                        'pass': all(d['pass'] for d in route_record['inputs'].values()) and
                                all(d['pass'] for d in comparisons.values()),
                        'raw_old_singleton': old_raw[0].tolist(),
                        'processed_old_singleton': old_processed[0].tolist(),
                        'raw_feature_only': raw[0].tolist(),
                        'processed_feature_only': processed[0].tolist()})
                except Exception as exc:
                    route_record['folds'].append({'fold': bundle['fold'], 'pass': False,
                        'error_type': type(exc).__name__, 'error': str(exc)})
        records.append(sample_record)
        _progress(output, 'representative_complete', symbol=symbol, session_date=day,
                  passing_folds=sum(x['pass'] for r in sample_record['routes']
                                    for x in r['folds']))
    after, changed = verify_unchanged(before)
    _, final_global_drift = verify_unchanged(before_global)
    _write_new(output/'representative_sources_after.json', after)
    _write_new(output/'representatives.json', records)
    total = sum(len(route['folds']) for sample in records for route in sample['routes'])
    passing = sum(fold['pass'] for sample in records for route in sample['routes']
                  for fold in route['folds'])
    _write_new(output/'representative_summary.json', {'keys': len(records),
        'fold_predictions': total, 'passing_fold_predictions': passing,
        'changed_sources': changed, 'global_changed_sources': final_global_drift,
        'representatives_pass': len(records)==6 and
        total == 90 and passing == 90 and not changed and not final_global_drift})
    return records


def run_representatives(output):
    output = Path(output)
    before = json.loads((output/'input_hashes_before.json').read_text())
    result, failure = None, None
    try:
        result = _run_representatives_inner(output)
    except Exception as exc:
        failure = exc
    after, changed = verify_unchanged(before)
    _write_new(output/'representative_global_sources_after.json', after)
    if changed and failure is None:
        failure = ValueError('representative global model/source bytes changed')
    local_pass = (json.loads((output/'representative_summary.json').read_text())[
        'representatives_pass'] if (output/'representative_summary.json').exists() else False)
    _write_new(output/'representative_final_status.json',
               {'pass': failure is None and not changed and local_pass,
                'global_changed_sources': changed,
                'fatal': None if failure is None else
                    {'error_type': type(failure).__name__, 'error': str(failure)}})
    if failure is not None:
        _write_new(output/'representatives_fatal.json',
                   {'error_type': type(failure).__name__, 'error': str(failure),
                    'global_changed_sources': changed})
        _progress(output, 'representatives_fatal', error_type=type(failure).__name__,
                  error=str(failure), changed_sources=len(changed))
        raise failure
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Isolated R03M numeric evidence only')
    parser.add_argument('stage', choices=('inventory','all_keys','representatives'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    if args.stage == 'inventory':
        inventory(args.output)
    elif args.stage == 'all_keys':
        run_all_keys(args.output)
    else:
        run_representatives(args.output)


if __name__ == '__main__':
    main()
