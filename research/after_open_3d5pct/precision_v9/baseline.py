"""Registered R03 incumbent replay. Reads immutable inputs; never selects a new recipe."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from ..contracts import Session
from ..focus_v8 import run as old
from ..focus_v8.core import TARGETS, GROUPS, focus_weights, historical_baseline, metrics, score
from ..train_multiscale_v6 import project_monotone, tabular
from . import evaluate as ev
from . import native_artifact as native

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ('old_common', 'new_common', 'new_expanded_fit')
SOURCE_CODE = (
    HERE/'baseline.py', HERE/'evaluate.py', HERE/'prepare.py', HERE/'data.py',
    HERE.parent/'focus_v8/run.py', HERE.parent/'focus_v8/core.py',
    HERE.parent/'focus_v8/prepare.py',
    HERE.parent/'focus_v8/support.py', HERE.parent/'focus_v8/protocol.json',
    HERE.parent/'train_multiscale_v6.py', HERE.parent/'v6_data.py',
    HERE.parent/'iterations_v7/b_group/experiment.py',
    HERE.parent/'iterations_v7/c_no_daily/experiment.py',
    HERE.parent/'contracts.py', HERE.parent/'timeaxis.py',
    HERE/'native_artifact.py', HERE/'forward/schema.py',
    HERE/'forward/features.py', HERE/'forward/inference.py',
)
LABEL_FIELDS = ('cutoff_at', 'decision_at', 'entry_at', 'entry_price',
                'label_end_at', 'label_available_at',
                'terminal_1d', 'terminal_3d', 'terminal_5d')


def _path(plan, name):
    if name not in plan:
        raise ValueError(f'missing plan field: {name}')
    return Path(plan[name]).expanduser().resolve(strict=True)


def _dataset_files(root):
    return [root/'dataset'/name for name in ('rows.parquet', 'features.npz', 'manifest.json')]


def _row_keys(rows):
    if rows.duplicated(['symbol', 'session_date']).any() or rows.sample_id.duplicated().any():
        raise ValueError('duplicate symbol/date or sample_id')
    return {(s, d): i for i, (s, d) in
            enumerate(zip(rows.symbol.astype(str), rows.session_date.astype(str)))}


def _allowed_keys(path):
    if path.suffix == '.parquet':
        frame = pd.read_parquet(path)
    elif path.suffix == '.csv':
        frame = pd.read_csv(path, dtype=str)
    else:
        raise ValueError('allowed_keys must be a parquet or CSV table')
    if not {'symbol', 'session_date'} <= set(frame.columns):
        raise ValueError('allowed_keys needs symbol and session_date')
    if frame.duplicated(['symbol', 'session_date']).any():
        raise ValueError('duplicate allowed key')
    return set(zip(frame.symbol.astype(str), frame.session_date.astype(str)))


def validate_common_keys(old_root, new_root, allowed_path):
    """Require the complete common key set and exact old/new outcome parity."""
    old_rows = pd.read_parquet(old_root/'dataset/rows.parquet')
    new_rows = pd.read_parquet(new_root/'dataset/rows.parquet')
    oi, ni = _row_keys(old_rows), _row_keys(new_rows)
    allowed = _allowed_keys(allowed_path)
    common = oi.keys() & ni.keys()
    if allowed != common:
        raise ValueError(f'allowed keys differ from full common set: missing={len(common-allowed)}, '
                         f'extra={len(allowed-common)}')
    if not allowed:
        raise ValueError('no common rows')
    with np.load(old_root/'dataset/features.npz') as z:
        old_y = z['y']
    with np.load(new_root/'dataset/features.npz') as z:
        new_y = z['y']
    if len(old_y) != len(old_rows) or len(new_y) != len(new_rows):
        raise ValueError('feature/row count mismatch')
    for key in sorted(allowed):
        a, b = oi[key], ni[key]
        if any(old_rows.at[a, f] != new_rows.at[b, f] for f in LABEL_FIELDS):
            raise ValueError(f'old/new entry or terminal mismatch: {key}')
        if not np.array_equal(old_y[a], new_y[b], equal_nan=True):
            raise ValueError(f'old/new label mismatch: {key}')
    return allowed, old_rows, new_rows


def _calendar(path):
    raw = json.loads(path.read_text())['sessions']
    dates = [s['session_date'] for s in raw]
    if dates != sorted(set(dates)):
        raise ValueError('calendar sessions are duplicate or unordered')
    sessions = [Session(datetime.fromisoformat(s['open_at']),
                        datetime.fromisoformat(s['close_at'])) for s in raw]
    return dates, sessions


def official_dates(dates, start, end):
    if not dates or dates[0] > start or dates[-1] < end:
        raise ValueError('calendar does not span the full fold')
    selected = [d for d in dates if start <= d < end]
    if not selected:
        raise ValueError('no official sessions in fold segment')
    return selected


def _fold_config(fold_id):
    if fold_id == 'diagnostic':
        fold_id = 'reserved'
    for fold in old.PROTOCOL['folds']:
        if fold['id'] == fold_id:
            return fold
    raise ValueError('unknown frozen fold')


def split_indices(rows, fold, allowed, arm, fit_date_floor):
    """Restrict train/evaluation eligibility, leaving full rows for mature_prior."""
    if arm not in ARMS:
        raise ValueError('unknown plan arm')
    base = old.splits(rows, fold)
    keys = np.fromiter(((s, d) in allowed for s, d in
                        zip(rows.symbol.astype(str), rows.session_date.astype(str))), bool, len(rows))
    lower = min(d for _, d in allowed)
    dates = rows.session_date.to_numpy()
    if ((arm == 'new_expanded_fit' and fit_date_floor >= lower) or
        (arm != 'new_expanded_fit' and fit_date_floor != lower)):
        raise ValueError('fit floor is incompatible with common key start')
    fit_eligible = keys.copy()
    if arm == 'new_expanded_fit':
        fit_eligible |= dates < lower
    fit_eligible &= dates >= fit_date_floor
    result = {name: ids[(fit_eligible if name == 'fit' else keys)[ids]]
              for name, ids in base.items()}
    result['eval_full'] = base['eval']
    if any(len(result[name]) == 0 for name in ('fit', 'tune', 'cal', 'eval')):
        raise ValueError('empty selected fit/tune/cal/eval segment')
    return result


def _source_hashes(paths):
    return {str(path): ev.sha(path) for path in sorted(set(paths))}


def _validate_actual_manifest(path, new_root, calendar_path, paired_path, purpose):
    manifest = json.loads(path.read_text())
    expected_status = ('engineering_validation_only' if purpose == 'engineering_validation_only'
                       else 'accepted_for_fit')
    if manifest.get('status') != expected_status:
        raise ValueError('actual manifest has not accepted the requested execution purpose')
    required_limits = {'current_snapshot_retrospective', 'historical_availability_assumed',
                       'corporate_actions_historical_coverage_uncertified'}
    if (manifest.get('evidence_scope') != 'exposed_development' or
        not required_limits <= set(manifest.get('limitations', []))):
        raise ValueError('actual manifest must retain exposed-development limitations')
    expected = {name:ev.sha(new_root/'dataset'/name)
                for name in ('rows.parquet','features.npz','manifest.json')}
    if manifest.get('dataset_sha256') != expected:
        raise ValueError('actual manifest dataset hashes differ from selected snapshot')
    if (manifest.get('calendar_sha256') != ev.sha(calendar_path) or
        manifest.get('paired_audit_sha256') != ev.sha(paired_path)):
        raise ValueError('actual manifest calendar or paired audit hash differs')


def verify_sources(registration):
    for path, expected in registration['source_hashes'].items():
        if ev.sha(Path(path)) != expected:
            raise ValueError(f'registered source changed: {path}')


def _support(rows, fold, recipe, seed):
    out = {}
    weights = old.training_weights(rows, fold['fit'], recipe, seed)
    effective = fold['fit'][weights > 0]
    for name, ids in fold.items():
        selection = rows.iloc[ids]
        out[name] = {'rows': len(ids), 'dates': int(selection.session_date.nunique()),
                     'full_126_prior_days': int(selection.full_126_prior_days.sum()),
                     'full_126_fraction': float(selection.full_126_prior_days.mean()),
                     'available_daily_days_median': float(selection.available_daily_days.median()),
                     'available_daily_days_min': int(selection.available_daily_days.min()),
                     'available_daily_days_max': int(selection.available_daily_days.max()),
                     'group_valid_count_histogram': {str(k):int(v) for k,v in
                         selection.group_valid_count.value_counts().sort_index().items()}}
    selected = rows.iloc[effective]
    out['effective_fit'] = {'rows': len(effective), 'dates': int(selected.session_date.nunique()),
                            'full_126_prior_days': int(selected.full_126_prior_days.sum()),
                            'full_126_fraction': float(selected.full_126_prior_days.mean()),
                            'group_valid_count_histogram': {str(k):int(v) for k,v in
                                selected.group_valid_count.value_counts().sort_index().items()}}
    if not len(effective):
        raise ValueError('no positive-weight fit rows')
    return out


def prepare_run(plan_path, output, *, allow_diagnostic=False):
    """Validate all identities and register before any fit or cache write."""
    plan_path = Path(plan_path).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    purpose = plan.get('purpose', 'r03_baseline')
    if purpose not in ('r03_baseline', 'engineering_validation_only'):
        raise ValueError('unknown execution purpose')
    if plan.get('arm') not in ARMS or plan.get('route') not in old.PROTOCOL['routes']:
        raise ValueError('invalid route or arm')
    if plan.get('seed') != 3566:
        raise ValueError('incumbent replay requires seed 3566')
    fold = _fold_config(plan['fold'])
    if fold['id'] == 'reserved' and not allow_diagnostic:
        raise ValueError('exposed diagnostic requires --allow-diagnostic')
    old_root, new_root = _path(plan, 'old_dataset_root'), _path(plan, 'new_dataset_root')
    if purpose == 'engineering_validation_only':
        if old_root != new_root or plan['arm'] != 'old_common' or fold['id'] != 'dev1':
            raise ValueError('engineering replay is limited to same-snapshot old_common dev1')
    elif old_root == new_root:
        raise ValueError('R03 baseline needs distinct old and new immutable snapshots')
    dataset_root = _path(plan, 'dataset_root')
    expected = old_root if plan['arm'] == 'old_common' else new_root
    if dataset_root != expected:
        raise ValueError('arm uses the wrong dataset snapshot')
    calendar_path = _path(plan, 'calendar')
    allowed_path = _path(plan, 'allowed_keys')
    summary_path = _path(plan, 'incumbent_summary')
    docs = [_path(plan, name) for name in ('actual_manifest', 'paired_audit', 'backlog', 'matrix')]
    if purpose == 'r03_baseline':
        docs.append(_path(plan, 'native_fit_adoption'))
    paired = json.loads(_path(plan, 'paired_audit').read_text())
    if (paired.get('paired_rows', 0) <= 0 or
        paired.get('arrays', {}).get('y', {}).get('changed_rows', -1) != 0 or
        any(paired.get('fields', {}).get(field, {}).get('changed_rows', -1) != 0
            for field in LABEL_FIELDS)):
        raise ValueError('paired builder audit has changed or unverified outcomes')
    if not (dataset_root/'dataset/manifest.json').exists():
        raise ValueError('missing dataset manifest')
    _validate_actual_manifest(_path(plan,'actual_manifest'),new_root,calendar_path,
                              _path(plan,'paired_audit'),purpose)
    allowed, old_rows, new_rows = validate_common_keys(old_root, new_root, allowed_path)
    if paired['paired_rows'] != len(allowed):
        raise ValueError('paired audit and registered common keys disagree')
    rows = old_rows if plan['arm'] == 'old_common' else new_rows
    _row_keys(rows)
    fit_floor = plan['fit_date_floor']
    fold_ids = split_indices(rows, fold, allowed, plan['arm'], fit_floor)
    dates, sessions = _calendar(calendar_path)
    official_dates(dates, fold['tune'], fold['end'])
    official_dates(dates, fold['eval'], fold['end'])
    summary = json.loads(summary_path.read_text())
    recipe = summary[plan['route']]['incumbent_recipe']
    if not isinstance(recipe, dict) or not all(v is True for v in recipe.values()):
        raise ValueError('invalid frozen incumbent recipe')
    if {'within_stock_rank', 'no_group_categories', 'group_dropout'} & recipe.keys():
        raise ValueError('R01 failed mechanism entered incumbent recipe')
    if fold['id'] == 'reserved':
        results = [_path(plan, name) for name in ('dev1_result', 'dev2_result')]
        for name, path in zip(('dev1', 'dev2'), results):
            earlier = json.loads(path.read_text())
            if (earlier.get('route'), earlier.get('arm'), earlier.get('fold')) != (plan['route'], plan['arm'], name):
                raise ValueError('diagnostic lacks matching completed development result')
            if not earlier.get('source_unchanged'):
                raise ValueError('development result lacks verified immutable sources')
        docs.extend(results + [path.parent/'registration.json' for path in results])
    sources = [plan_path, calendar_path, allowed_path, summary_path, *docs,
               HERE/'PROTOCOL.md', HERE.parent/'focus_v8/PROTOCOL.md',
               HERE/'rounds/R03/native_artifact/REGISTRATION_DRAFT.md',
               HERE/'rounds/R03/native_artifact/ADOPTION.md',
               *_dataset_files(old_root), *_dataset_files(new_root), *SOURCE_CODE]
    source_hashes = _source_hashes(sources)
    if fold['id'] == 'reserved':
        shared = [calendar_path, allowed_path, summary_path,
                  *_dataset_files(old_root), *_dataset_files(new_root)]
        for path in results:
            previous = json.loads((path.parent/'registration.json').read_text())
            if any(previous['source_hashes'].get(str(p)) != source_hashes[str(p)] for p in shared):
                raise ValueError('development result used different data or calendar sources')
    support = _support(rows, fold_ids, recipe, plan['seed'])
    registration = {'status': 'registered_before_fit', 'created_at': datetime.now().astimezone().isoformat(),
                    'purpose': purpose,
                    'arm': plan['arm'], 'route': plan['route'], 'fold': fold,
                    'recipe': recipe, 'seed': plan['seed'], 'fit_date_floor': fit_floor,
                    'allowed_keys_count': len(allowed), 'split_support': support,
                    'source_hashes': source_hashes,
                    'prior_scope': 'all_rows_in_selected_immutable_dataset_before_allowed_key_filter',
                    'calibration': 'frozen_v8_shrunken_platt', 'projection': 'frozen_v8_monotone',
                    'thresholds': ev.THRESHOLDS, 'diagnostic_exposed': fold['id'] == 'reserved',
                    'independent_validation': False, 'goal_complete': False}
    output.mkdir(parents=True, exist_ok=False)
    if purpose == 'r03_baseline':
        try:
            registration.update(native.register_prefit(
                plan_path, plan, registration, rows, fold_ids, dataset_root, output, SOURCE_CODE))
        except Exception as exc:
            ev.write(output/'native_registration_failure.json',
                     {'error_type': type(exc).__name__, 'error': str(exc),
                      'stage': 'before_fit'})
            raise
    ev.write(output/'registration.json', registration)
    snapshot = output/'source_snapshot'
    for path in SOURCE_CODE:
        dest = snapshot/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    verify_sources(registration)
    return {'plan': plan, 'registration': registration, 'rows': rows, 'fold': fold,
            'fold_ids': fold_ids, 'dates': dates, 'sessions': sessions,
            'dataset_root': dataset_root, 'output': output}


def _load_registered(context):
    cache = context['output']/'cache'
    cache.mkdir()
    (cache/'dataset').symlink_to(context['dataset_root']/'dataset', target_is_directory=True)
    loaded = old.load_data(cache)
    if not loaded[0].equals(context['rows']):
        raise ValueError('loaded rows differ from preregistered rows')
    return loaded


def _assert_b_no_group_schema(loaded, recipe):
    _, data, _, _, _, paths, xbase, _, _, _, _ = loaded
    own = tabular(data, group=False).to_numpy(np.float32)
    if own.shape[1] != 585 or xbase.shape[1] != 765 or paths.shape[1] != 84:
        raise ValueError('B_no_group incumbent feature widths changed')
    if not np.array_equal(own, xbase[:, :585], equal_nan=True):
        raise ValueError('B_no_group base columns include group information')
    if recipe.get('representation') is not True or recipe.get('curves') or recipe.get('prior'):
        raise ValueError('B_no_group is not the frozen 585+84 incumbent')
    return {'base': {'count': 585, 'source': 'x5_x60_xday_only',
                     'names': [f'f_{i:03d}' for i in range(585)]},
            'path': {'count': 84, 'source': 'x5_x60_xday_only',
                     'names': [f'path_{i:03d}' for i in range(84)]}}


def _write_supervision_and_prior(context, loaded, recipe):
    rows, _, _, prior, counts, *_ = loaded
    weights = old.training_weights(rows, context['fold_ids']['fit'], recipe,
                                   context['registration']['seed'])
    fit_ids = context['fold_ids']['fit'][weights > 0]
    rows.iloc[fit_ids][['sample_id','symbol','session_date']].to_parquet(
        context['output']/'fit_supervision_ids.parquet',index=False)
    if not recipe.get('prior'):
        return
    available = pd.to_datetime(rows.label_available_at, utc=True).astype('int64').to_numpy()
    end = pd.to_datetime(rows.label_end_at, utc=True).astype('int64').to_numpy()
    decisions = pd.to_datetime(rows.decision_at, utc=True).astype('int64').to_numpy()
    history = [None]*len(rows)
    for _, ids0 in rows.groupby('symbol').indices.items():
        ids = np.array(sorted(ids0, key=lambda i: decisions[i]))
        for i in ids:
            earlier = ids[(decisions[ids] < decisions[i]) &
                          (available[ids] < decisions[i]) & (end[ids] < decisions[i])][-63:]
            if len(earlier) != counts[i]:
                raise ValueError('mature prior provenance differs from frozen old implementation')
            history[i] = rows.iloc[earlier].sample_id.astype(str).tolist()
    pd.DataFrame({'sample_id':rows.sample_id,'prior_history_ids':history,
                  'mature_prior_count':counts}).to_parquet(
                      context['output']/'prior_history_ids.parquet',index=False)


def _frames_and_metrics(context, loaded, raw, params):
    rows, _, y, prior, counts, _, _, _, _, _, _ = loaded
    frames, results = {}, {}
    for name, ids in context['fold_ids'].items():
        p = project_monotone(old.apply_cal(raw[name], params)).reshape(-1, 9)
        history = historical_baseline(rows, y, context['fold_ids']['fit'], ids)
        frame = rows.iloc[ids][['sample_id', 'symbol', 'session_date', 'decision_at']].reset_index(drop=True).copy()
        for j, target in enumerate(TARGETS):
            frame['y_'+target] = y[ids, j]
            frame['raw_'+target] = raw[name][:, j]
            frame['p_'+target] = p[:, j]
            frame['history_'+target] = history[:, j]
            frame['prior_'+target] = prior[ids, j]
        frame['mature_prior_count'] = counts[ids]
        frame.to_parquet(context['output']/f'{name}.parquet', index=False)
        frames[name] = frame
        selection = rows.iloc[ids]
        results[name] = {'rows': len(ids), 'dates': int(selection.session_date.nunique()),
                         'score': score(selection, y[ids], p),
                         'groups': metrics(selection.reset_index(drop=True), y[ids], p),
                         'raw_groups': metrics(selection.reset_index(drop=True), y[ids],
                                               project_monotone(raw[name]).reshape(-1, 9)),
                         'mature_prior_count_median': float(np.median(counts[ids]))}
    return frames, results


def _selection(context, frames):
    rows = context['rows']
    cal = ev.enrich(frames['cal'], rows, context['sessions'])
    cal_dates = [d for d in context['dates'] if cal.session_date.min() <= d <= cal.session_date.max()]
    thresholds, curves = ev.choose_thresholds(cal, cal_dates)
    joint_cal = ev.select_signals(cal, thresholds)
    cal_joint = {group: ev.summarize(cal, joint_cal, group, cal_dates) for group in GROUPS}
    ev.write(context['output']/'selection.json', {'thresholds': thresholds,
             'cal_curves': curves, 'cal_joint_selected': cal_joint,
             'cal_official_dates': cal_dates, 'selection_source': 'cal_only'})
    eval_dates = official_dates(context['dates'], context['fold']['eval'], context['fold']['end'])
    all_metrics = {}
    for segment in ('eval', 'eval_full'):
        frame = ev.enrich(frames[segment], rows, context['sessions'])
        methods = [('fixed_raw_90', {g: .9 for g in GROUPS}, 'raw_3d_5pct'),
                   ('fixed_processed_90', {g: .9 for g in GROUPS}, 'p_3d_5pct'),
                   ('cal_selected', thresholds, 'raw_3d_5pct')]
        for method, limits, score_column in methods:
            signals = ev.select_signals(frame, limits, score_column)
            signals.to_parquet(context['output']/f'{segment}_{method}_signals.parquet', index=False)
            all_metrics[f'{segment}:{method}'] = {
                g: {**ev.summarize(frame, signals, g, eval_dates, score_column),
                    'threshold': limits[g], 'intervals': ev.block_intervals(signals, g, eval_dates)}
                for g in GROUPS}
    ev.write(context['output']/'precision.json', all_metrics)
    return thresholds, all_metrics


def run(plan_path, output, *, allow_diagnostic=False):
    context = prepare_run(plan_path, output, allow_diagnostic=allow_diagnostic)
    started = time.monotonic()
    registration = context['registration']
    if registration['purpose'] == 'r03_baseline':
        native.verify_prefit(context['output'], registration['native_contract_file'])
    loaded = _load_registered(context)
    rows, _, y, prior, counts, paths, xbase, curve, relative, arrays, _ = loaded
    recipe, route, seed = registration['recipe'], registration['route'], registration['seed']
    if route == 'B_no_group':
        ev.write(context['output']/'feature_schema.json', _assert_b_no_group_schema(loaded, recipe))
    _write_supervision_and_prior(context, loaded, recipe)
    fold = context['fold_ids']
    if route.startswith('C'):
        raw, fit_info = old.fit_c(route, recipe, rows, y, arrays, fold, context['output'], seed)
    else:
        raw, fit_info = old.fit_b(route, recipe, rows, y, prior, xbase, paths, curve, relative,
                                  fold, context['output'], seed)
    if route == 'B_no_group' and fit_info['features'] != 669:
        raise ValueError('B_no_group fitted feature count differs from frozen 669')
    params = old.calibrate(raw['cal'], y[fold['cal']], focus_weights(rows.iloc[fold['cal']])) \
             if recipe.get('calibration') else [[1., 0.]]*9
    ev.write(context['output']/'calibration.json', {
        'method': 'shrunken_nonnegative_platt' if recipe.get('calibration') else 'identity_raw_estimate',
        'parameters': params, 'fitted_on': 'cal_only'})
    frames, scores = _frames_and_metrics(context, loaded, raw, params)
    thresholds, precision = _selection(context, frames)
    verify_sources(registration)
    artifacts = [p for p in context['output'].rglob('*') if p.is_file()]
    result = {'route': route, 'arm': registration['arm'], 'fold': context['fold']['id'],
              'purpose': registration['purpose'],
              'recipe': recipe, 'seed': seed, 'fit': fit_info, 'scores': scores,
              'thresholds': thresholds, 'precision': precision,
              'split_support': registration['split_support'],
              'prior_scope': registration['prior_scope'],
              'cost': {'seconds': time.monotonic()-started,
                       'artifact_bytes_before_result': sum(p.stat().st_size for p in artifacts)},
              'source_unchanged': True, 'diagnostic_exposed': context['fold']['id'] == 'reserved',
              'native_contract_sha256': registration.get('native_contract_file', {}).get('sha256'),
              'native_model_version': registration.get('native_model_version'),
              'independent_validation': False, 'goal_complete': False}
    ev.write(context['output']/'result.json', result)
    ev.write(context['output']/'summary.json', {'route': route, 'arm': registration['arm'],
             'purpose': registration['purpose'],
             'fold': context['fold']['id'], 'results': str(context['output']/'result.json'),
             'source_unchanged': True, 'development_only': True,
             'formal_passing_routes': [], 'goal_complete': False})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-diagnostic', action='store_true')
    args = parser.parse_args()
    run(args.plan, args.output, allow_diagnostic=args.allow_diagnostic)
