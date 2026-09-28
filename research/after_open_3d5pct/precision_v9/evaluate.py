"""Offline selective-prediction audit; never tunes on evaluation labels."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..contracts import Session
from ..timeaxis import add_regular_minutes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V8 = json.loads((HERE.parent / 'focus_v8/protocol.json').read_text())
GROUPS = V8['groups']
ROUTES = dict(zip(V8['routes'], ['B有群', 'B无群', 'C有群', 'C无群', 'C无日级']))
THRESHOLDS = [round(.30 + .05 * i, 2) for i in range(14)]
PRIMARY = '3d_5pct'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def calendar():
    raw = json.loads((ROOT / 'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())
    return raw['sessions'], [Session(datetime.fromisoformat(s['open_at']),
                                    datetime.fromisoformat(s['close_at'])) for s in raw['sessions']]


def enrich(frame, rows, sessions):
    if frame.sample_id.duplicated().any() or rows.sample_id.duplicated().any():
        raise ValueError('duplicate sample IDs')
    out = frame.merge(rows[['sample_id', 'entry_at', 'terminal_3d']], on='sample_id', validate='one_to_one')
    if len(out) != len(frame):
        raise ValueError('missing entry metadata')
    ends = {s: add_regular_minutes(datetime.fromisoformat(s), 1170, sessions).isoformat()
            for s in out.entry_at.unique()}
    out['signal_end_at'] = out.entry_at.map(ends)
    return out


def select_signals(frame, thresholds, score_column='raw_3d_5pct'):
    """Per-model own-symbol cooldown; outcome fields are never consulted.

    One row can trigger several groups, but is one union opportunity. Only groups
    whose threshold triggered receive the opportunity, so cross-membership does
    not silently give a group a signal below its frozen threshold.
    """
    if frame.sample_id.duplicated().any():
        raise ValueError('duplicate sample IDs')
    if not np.isfinite(frame[score_column]).all():
        raise ValueError('nonfinite scores')
    busy, selected = {}, []
    for row in frame.sort_values(['decision_at', 'symbol']).to_dict('records'):
        groups = [g for g, t in thresholds.items() if t is not None
                  and row['symbol'] in GROUPS[g] and row[score_column] >= t]
        now = pd.Timestamp(row['decision_at'])
        if not groups or (row['symbol'] in busy and now < busy[row['symbol']]):
            continue
        busy[row['symbol']] = pd.Timestamp(row['signal_end_at'])
        row['trigger_groups'] = groups
        selected.append(row)
    return pd.DataFrame(selected, columns=[*frame.columns, 'trigger_groups'])


def summarize(frame, signals, group, session_dates, score_column='raw_3d_5pct'):
    eligible = frame[frame.symbol.isin(GROUPS[group])]
    selected = signals[signals.trigger_groups.map(lambda gs: group in gs).astype(bool)]
    if not eligible.session_date.isin(session_dates).all():
        raise ValueError('evaluation rows outside official window')
    n = len(selected)
    y = selected['y_' + PRIMARY].to_numpy(float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('only complete mature binary outcomes can be scored')
    tp = int(y.sum())
    positives = int(eligible['y_' + PRIMARY].sum())
    dates = selected.session_date.nunique()
    return {'n': n, 'tp': tp, 'fp': n - tp, 'precision': tp/n if n else None,
            'recall': tp/positives if positives else None,
            'signals_per_5_sessions': n * 5 / len(session_dates) if session_dates else 0,
            'signal_dates': int(dates), 'evaluation_sessions': len(session_dates),
            'eligible_rows': len(eligible), 'eligible_positive_rows': positives,
            'rejected_or_cooldown_rows': len(eligible) - n,
            'selected_stocks': int(selected.symbol.nunique()),
            'mean_score': float(selected[score_column].mean()) if n else None,
            'mean_terminal_3d': float(selected.terminal_3d.mean()) if n else None,
            'worst_terminal_3d': float(selected.terminal_3d.min()) if n else None,
            'point_and_supply_pass': bool(n and tp/n >= .9 and n*5 >= len(session_dates))}


def choose_thresholds(cal, dates):
    curves, chosen = {}, {}
    for g in GROUPS:
        curves[g] = []
        for threshold in THRESHOLDS:
            signals = select_signals(cal, {g: threshold})
            curves[g].append({'threshold': threshold, **summarize(cal, signals, g, dates)})
        eligible = [m for m in curves[g] if m['point_and_supply_pass']
                    and m['n'] >= 5 and m['signal_dates'] >= 3]
        best = max(eligible, key=lambda m: (m['n'], m['precision'], m['threshold'])) if eligible else None
        chosen[g] = best['threshold'] if best else None
    return chosen, curves


def block_intervals(signals, group, dates, repeats=4000):
    """Development uncertainty only; zero-signal resamples are not perfect rates."""
    s = signals[signals.trigger_groups.map(lambda gs: group in gs).astype(bool)]
    z = s.groupby('session_date')['y_' + PRIMARY].agg(['sum', 'count']).reindex(dates, fill_value=0)
    a = z.to_numpy()
    result = {}
    for block in (5, 10):
        if len(dates) < block or not a[:, 1].sum():
            result[str(block)] = {'ci95': None, 'simultaneous_15': None, 'valid_draws': 0}
            continue
        rng = np.random.default_rng(9566)
        starts = rng.integers(0, len(dates)-block+1, (repeats, int(np.ceil(len(dates)/block))))
        ids = (starts[:, :, None] + np.arange(block)).reshape(repeats, -1)[:, :len(dates)]
        hit = a[ids, 0].sum(1)
        n = a[ids, 1].sum(1)
        values = hit[n > 0]/n[n > 0]
        result[str(block)] = {'ci95': np.quantile(values, [.025, .975]).tolist() if len(values) else None,
                             'simultaneous_15': np.quantile(values, [.05/30, 1-.05/30]).tolist() if len(values) else None,
                             'valid_draws': len(values), 'zero_signal_draws': int((n == 0).sum())}
    return result


def audit(source, output):
    output.mkdir(parents=True, exist_ok=False)
    selected = json.loads((source/'round_10/summary.json').read_text())
    raw_calendar, sessions = calendar()
    rows = pd.read_parquet(source/'dataset/rows.parquet')
    files = [source/'dataset/rows.parquet', source/'dataset/features.npz',
             source/'round_10/summary.json', source/'final_selection.json',
             ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json']
    cases = []
    for route in ROUTES:
        paths = [Path(p) for p in selected[route]['incumbent_paths']]
        paths += [source/'final'/route/'selected_3566']
        for path, fold in zip(paths, V8['folds']):
            files.extend(path/n for n in ['cal.parquet', 'eval.parquet', 'registration.json', 'result.json'])
            cases.append((route, path, fold))
    before = {str(p): sha(p) for p in files}
    write(output/'registration.json', {'status': 'registered_before_computation',
          'created_at': datetime.now().astimezone().isoformat(), 'input_hashes': before,
          'code_sha256': sha(Path(__file__)), 'protocol_sha256': sha(HERE/'PROTOCOL.md'),
          'backlog_sha256': sha(HERE/'rounds/R00/BACKLOG.md'), 'matrix_sha256': sha(HERE/'rounds/R00/MATRIX.md'),
          'thresholds': THRESHOLDS, 'all_periods_exposed': True})
    records, selections, union_signals = [], [], []
    for route, path, fold in cases:
        cal = enrich(pd.read_parquet(path/'cal.parquet'), rows, sessions)
        ev = enrich(pd.read_parquet(path/'eval.parquet'), rows, sessions)
        # Only the fully matured cal dates preserved by the old purged split;
        # no internal missing day may disappear from the denominator.
        cdates = [s['session_date'] for s in raw_calendar if cal.session_date.min() <= s['session_date'] <= cal.session_date.max()]
        edates = [s['session_date'] for s in raw_calendar if fold['eval'] <= s['session_date'] < fold['end']]
        thresholds, curves = choose_thresholds(cal, cdates)
        selections.append({'route': route, 'fold': fold['id'], 'source': str(path),
                           'thresholds': thresholds, 'cal_curves': curves})
        methods = [('fixed_raw_90', {g: .9 for g in GROUPS}, 'raw_3d_5pct'),
                   ('fixed_processed_90', {g: .9 for g in GROUPS}, 'p_3d_5pct'),
                   ('cal_selected', thresholds, 'raw_3d_5pct')]
        for method, limits, column in methods:
            signals = select_signals(ev, limits, column)
            for g in GROUPS:
                records.append({'route': route, 'fold': fold['id'], 'method': method, 'group': g,
                                'threshold': limits[g], **summarize(ev, signals, g, edates, column),
                                'intervals': block_intervals(signals, g, edates)})
            for row in signals.to_dict('records'):
                union_signals.append({'route': route, 'fold': fold['id'], 'method': method, **row})
    unchanged = all(sha(Path(p)) == h for p, h in before.items())
    if not unchanged:
        raise ValueError('source changed during audit')
    write(output/'metrics.json', records)
    write(output/'selection.json', selections)
    pd.DataFrame([{k:v for k,v in x.items() if k != 'intervals'} for x in records]).to_csv(output/'metrics.csv', index=False)
    if union_signals:
        pd.DataFrame(union_signals).to_parquet(output/'signals.parquet', index=False)
    passing = [r for r in ROUTES if all(x['point_and_supply_pass'] for x in records
                                        if x['route'] == r and x['method'] == 'cal_selected')]
    write(output/'audit.json', {'source_unchanged': unchanged, 'development_passing_routes': passing,
          'formal_passing_routes': [], 'independent_validation': False, 'goal_complete': False,
          'n_model_group_fold_method_cells': len(records)})
    print(json.dumps({'output': str(output), 'development_passing_routes': passing, 'source_unchanged': unchanged}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=HERE.parent/'runs/focus_v8_20260927')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.source.resolve(), args.output.resolve())
