"""Finite, inner-only choice of existing probability transforms for v5 models."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .quality_eval_v5 import (
    _block_ci, action_comparison, choose_quality_recommendations,
    outcome_summary, probability_comparison, selection_status,
)
from .terminal_risk_v4 import REPO, mixed_expected_net, sha256

PROJECT = Path(__file__).resolve().parent


def probability_variant(frame: pd.DataFrame, method: str, config: dict) -> pd.DataFrame:
    """Recompute every probability-dependent decision on a separate view."""
    if method not in ('identity', 'sigmoid'):
        raise ValueError('unregistered probability transform')
    f = frame.reset_index(drop=True).copy()
    if method == 'identity':
        f['p_touch'] = f.p_raw
    f['probability_method'] = method
    f['probability_estimate_kind'] = ('raw_uncalibrated_model_estimate' if method == 'identity'
                                      else 'sigmoid_quality_calibrated_estimate')
    f['expected_net'] = mixed_expected_net(
        f.p_touch.to_numpy(float), f.failure_mean_gross.to_numpy(float),
        touch_gross=config['touch_gross_return'], buy_cost=config['buy_cost_rate'],
        sell_cost=config['sell_cost_rate'])
    f['selected_quality_scenario'] = choose_quality_recommendations(f, config['development_scenario'])
    return f


def choose_inner_plan(frame: pd.DataFrame, candidates: list[str]) -> tuple[dict, list[dict]]:
    """This function never reads an outer label or outcome for selection."""
    inner = frame[(frame.phase == 'selection') & frame.quality_eligible]
    trials = []
    expected = None
    for candidate in candidates:
        f = inner[inner.candidate == candidate].sort_values('sample_id')
        if f.empty or f.sample_id.duplicated().any():
            raise ValueError('missing or duplicate same-cohort inner predictions')
        signature = tuple(f.sample_id)
        if expected is None:
            expected = signature
        if signature != expected:
            raise ValueError('inner candidates have different evaluation populations')
        for method, col in [('identity', 'p_raw'), ('sigmoid', 'p_touch')]:
            p = f[col].to_numpy(float)
            if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                raise ValueError('invalid inner probability estimate')
            trials.append({'candidate': candidate, 'method': method, 'quality_n': len(f),
                           'brier': float(np.mean((f.target.to_numpy(float) - p) ** 2))})
    if not trials:
        raise ValueError('no completed inner candidate')
    winner = min(trials, key=lambda x: (x['brier'], 0 if x['method'] == 'sigmoid' else 1,
                                       x['candidate']))
    return dict(winner), trials


def compare_frozen_pipelines(new: pd.DataFrame, old: pd.DataFrame,
                             config: dict, date_axis: list[str]) -> dict:
    a = new[new.quality_eligible].sort_values('sample_id').reset_index(drop=True)
    b = old[old.quality_eligible].sort_values('sample_id').reset_index(drop=True)
    if not a.sample_id.equals(b.sample_id) or not a.target.equals(b.target):
        raise ValueError('pipeline comparison population differs')
    a['brier_improvement'] = ((b.target.to_numpy(float) - b.p_touch.to_numpy(float)) ** 2 -
                               (a.target.to_numpy(float) - a.p_touch.to_numpy(float)) ** 2)
    settings = config['bootstrap']
    return {'quality_n': len(a), 'brier_improvement_over_v5_selected': float(a.brier_improvement.mean()),
            'date_block_ci': {str(w): _block_ci(a, 'brier_improvement', date_axis, w,
                                              settings['replicates'], settings['seed'])
                              for w in settings['block_sessions']}}


def evaluate_view(f: pd.DataFrame, config: dict, axis: list[str]) -> dict:
    p = probability_comparison(f, config, axis)
    a = action_comparison(f, f.selected_quality_scenario.to_numpy(bool), config, axis)
    q = f[f.quality_eligible]
    return {'probability': p, 'action': a, 'v5_targets': selection_status(p, a, config),
            'all_pool': outcome_summary(f, config['tail_fraction']),
            'quality_pool': outcome_summary(q, config['tail_fraction']),
            'by_symbol_quality': {s: outcome_summary(g, config['tail_fraction'])
                                  for s, g in q.groupby('symbol')},
            'by_date': {day: {'quality': outcome_summary(q[q.session_date == day], config['tail_fraction']),
                              'selected': outcome_summary(f[(f.session_date == day) & f.selected_quality_scenario],
                                                          config['tail_fraction'])}
                        for day in axis}}


def _clean(x):
    if isinstance(x, dict): return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list): return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def run(config_path: Path, output: Path) -> dict:
    if output.exists(): raise FileExistsError(output)
    cfg = json.loads(config_path.read_text())
    if (cfg['protocol'] != 'causal_quality_calibration_selection_v51' or
            cfg['probability_methods'] != ['identity', 'sigmoid'] or
            cfg['selector'] != 'inner_quality_Brier_then_sigmoid_then_candidate_name'):
        raise ValueError('unregistered v5.1 protocol or search space')
    source = REPO / cfg['source_run']
    manifest_path = source / 'manifest.json'
    if sha256(manifest_path) != cfg['source_manifest_sha256']:
        raise ValueError('v5 manifest drift')
    m = json.loads(manifest_path.read_text())
    for group, root in [('result_artifacts_sha256', source), ('model_artifacts_sha256', source/'models')]:
        for name, expected in m[group].items():
            if sha256(root/name) != expected: raise ValueError(f'source artifact drift: {name}')
    c = json.loads((source/'config.json').read_text())
    provenance = json.loads((source/'provenance_at_launch.json').read_text())
    if sha256(source/'provenance_at_launch.json') != m['provenance_at_launch_sha256']:
        raise ValueError('v5 provenance drift')
    registered_config = PROJECT/'configs/quality_training_v5.json'
    if (sha256(registered_config) != m['config_sha256'] or
            provenance['files_sha256'].get(str(registered_config.relative_to(REPO))) != m['config_sha256'] or
            json.loads(registered_config.read_text()) != c):
        raise ValueError('v5 original config or reformatted copied config drift')
    calendar_path = REPO / c['calendar']
    for path in [calendar_path, PROJECT/'quality_eval_v5.py', PROJECT/'terminal_risk_v4.py']:
        expected = provenance['files_sha256'].get(str(path.relative_to(REPO)))
        if path.name == 'terminal_risk_v4.py' and expected is None:
            expected = cfg['valuation_code_sha256']
        if not expected or sha256(path) != expected:
            raise ValueError(f'upstream calendar or metric code drift: {path}')
    calendar = json.loads(calendar_path.read_text())['sessions']
    predictions = pd.read_parquet(source/'predictions.parquet')
    base_trials = [json.loads(line) for line in (source/'trials.jsonl').read_text().splitlines()]
    output.mkdir(parents=True)
    files = [config_path, Path(__file__).resolve(), PROJECT/'docs/21_quality_calibration_v51_registration.md',
             PROJECT/'quality_eval_v5.py', PROJECT/'terminal_risk_v4.py', manifest_path,
             source/'predictions.parquet', source/'selection.json', source/'trials.jsonl']
    launch = {'created_at': datetime.now(timezone.utc).isoformat(), 'timing': 'before_transform_selection',
              'files_sha256': {str(p.relative_to(REPO)): sha256(p) for p in files}}
    (output/'provenance_at_launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    selections, trials = {}, {}
    # Complete every inner-only decision and persist it before any outer scoring.
    for fold in c['folds']:
        name = fold['name']
        candidates = [t['candidate'] for t in base_trials if t['fold'] == name and t['status'] == 'completed']
        selected, log = choose_inner_plan(predictions[predictions.fold == name], candidates)
        selected['previous_v5_candidate'] = m['folds'][name]['best_inner_probability_candidate']
        selected['previous_v5_method'] = 'sigmoid'
        selections[name], trials[name] = selected, log
    (output/'selection.json').write_text(json.dumps(selections, indent=2)+'\n')
    (output/'inner_trials.json').write_text(json.dumps(trials, indent=2)+'\n')
    frozen_selection_hash = sha256(output/'selection.json')
    evaluation, frames = {}, []
    for fold in c['folds']:
        name, chosen = fold['name'], selections[fold['name']]
        evaluation[name] = {'selected': chosen, 'phases': {}}
        for phase in ['selection', 'outer']:
            pool = predictions[(predictions.fold == name) & (predictions.phase == phase)]
            new = probability_variant(pool[pool.candidate == chosen['candidate']], chosen['method'], c)
            old = probability_variant(pool[pool.candidate == chosen['previous_v5_candidate']], 'sigmoid', c)
            start = fold['selection_start'] if phase == 'selection' else fold['outer_start']
            stop = fold['outer_start'] if phase == 'selection' else fold['outer_end_exclusive']
            axis = [s['session_date'] for s in calendar if start <= s['session_date'] < stop]
            pair = compare_frozen_pipelines(new, old, c, axis)
            evaluation[name]['phases'][phase] = {'new': evaluate_view(new, c, axis),
                'previous_v5': evaluate_view(old, c, axis), 'paired_change': pair}
            if phase == 'selection':
                lo = pair['date_block_ci']['5']['lower']
                evaluation[name]['inner_iteration_goal_met'] = bool(
                    pair['brier_improvement_over_v5_selected'] >= cfg['min_inner_brier_improvement'] and
                    lo is not None and lo > 0)
            for label, f in [('v51_inner_selected', new), ('v5_inner_selected_reference', old)]:
                f['pipeline_role'] = label
                frames.append(f)
    if sha256(output/'selection.json') != frozen_selection_hash:
        raise ValueError('outer scoring changed frozen selection')
    pd.concat(frames, ignore_index=True).to_parquet(output/'predictions.parquet', index=False)
    (output/'evaluation.json').write_text(json.dumps(_clean(evaluation), ensure_ascii=False, indent=2)+'\n')
    result = {'protocol': cfg['protocol'], 'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'exposed_development_probability_transform_iteration', 'new_base_model_fits': 0,
              'source_model_fits_reused': m['completed_trials'], 'config_sha256': sha256(config_path),
              'source_manifest_sha256': sha256(manifest_path), 'selection_frozen_before_outer_sha256': frozen_selection_hash,
              'provenance_sha256': sha256(output/'provenance_at_launch.json'),
              'artifacts_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()},
              'formal_action_enabled': False}
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.config.resolve(), args.output.resolve()), indent=2))
