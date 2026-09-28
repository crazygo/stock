"""Frozen-cohort evaluation from all issued BUY events and official dates."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np

from .ledger import (Ledger, TARGETS, _artifact_valid, _prediction_hash,
                     _reject_registration_wrapper, REQUIRED_ARTIFACTS,
                     canonical, digest, file_sha, timestamp)


def _cell(buys, outcomes, target, group, official_count, violations):
    chosen = [b for b in buys if group in b['payload']['trigger_groups']]
    statuses = defaultdict(int)
    tp = fp = 0
    per_date = defaultdict(lambda: [0, 0])
    for buy in chosen:
        result = outcomes.get((buy['event_id'], target))
        status = result['outcome']['status'] if result else 'pending'
        if status == 'tp':
            tp += 1
            per_date[buy['session_date']][0] += 1
        elif status == 'fp':
            fp += 1
            per_date[buy['session_date']][1] += 1
        else:
            statuses[status] += 1
    n = len(chosen)
    u = n - tp - fp
    if n != tp + fp + u:
        raise AssertionError('BUY denominator lost')
    known = tp / (tp + fp) if tp + fp else None
    possible = [tp / n, (tp + u) / n] if n else None
    return {'issued': n, 'tp': tp, 'fp': fp, 'unresolved': u,
            'unresolved_by_reason': dict(statuses), 'known_precision': known,
            'possible_precision': possible, 'signal_dates': len({b['session_date'] for b in chosen}),
            'evaluation_sessions': official_count,
            'signals_per_5_sessions': n * 5 / official_count if official_count else None,
            'forward_violations': violations,
            '_date_counts': dict(per_date)}


def _bootstrap(cells: dict, dates: list[str], *, block: int, repeats: int, seed: int):
    """Reuse one date-index sample across all 15 cells; zero-buy draws are null."""
    rng = np.random.default_rng(seed + block)
    n = len(dates)
    ids = list(cells)
    draws = {key: [] for key in ids}
    simultaneous = []
    points = {key: c['known_precision'] for key, c in cells.items()}
    for _ in range(repeats):
        sampled = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            sampled.extend(dates[(start + j) % n] for j in range(block))
        sampled = sampled[:n]
        deviations = []
        valid_all = True
        for key in ids:
            counts = cells[key]['_date_counts']
            tp = sum(counts.get(day, [0, 0])[0] for day in sampled)
            fp = sum(counts.get(day, [0, 0])[1] for day in sampled)
            value = tp / (tp + fp) if tp + fp else None
            draws[key].append(value)
            if value is None or points[key] is None:
                valid_all = False
            else:
                deviations.append(abs(value - points[key]))
        if valid_all:
            simultaneous.append(max(deviations))
    intervals = {}
    for key, values in draws.items():
        valid = [v for v in values if v is not None]
        intervals[key] = {'interval_95': [float(np.quantile(valid, .025)),
                                           float(np.quantile(valid, .975))] if valid else None,
                          'valid_draws': len(valid), 'zero_signal_draws': repeats - len(valid)}
    radius = float(np.quantile(simultaneous, .95)) if simultaneous else None
    return {'block_sessions': block, 'repeats': repeats,
            'common_date_draws': True, 'cells': intervals,
            'simultaneous_15_cell_radius_95': radius,
            'simultaneous_valid_draws': len(simultaneous)}


def evaluate_cohort(cohort_id: str, ledger_cut: int, preregistration: dict, *,
                    ledger: Ledger) -> dict:
    """A point gate, never a claim that a 90% lower confidence bound was met."""
    _reject_registration_wrapper(preregistration)
    if not isinstance(ledger_cut, int) or ledger_cut < 1:
        raise ValueError('explicit ledger cut required')
    if ledger_cut > ledger.verify_chain():
        raise ValueError('ledger cut is beyond the current immutable chain')
    events = ledger.events(cut=ledger_cut)
    registered = next((e for e in events if e['event_type'] == 'cohort_registered'
                       and e['cohort_id'] == cohort_id), None)
    if not registered or registered['payload']['registration_sha256'] != digest(preregistration):
        raise ValueError('evaluation preregistration differs from immutable ledger')
    dates = preregistration['session_dates']
    if len(dates) < 60 or dates != sorted(set(dates)):
        raise ValueError('invalid frozen official denominator')
    routes = preregistration['routes']
    groups = preregistration['groups']
    contracts = preregistration.get('route_contracts') or {}
    if len(routes) != 5 or len(groups) != 3 or len(set(routes)) != 5 or \
            set(contracts) != set(routes):
        raise ValueError('expected five routes and three groups')
    buys = [e for e in events if e['event_type'] == 'buy' and e['cohort_id'] == cohort_id]
    session_rows = {(e['route_id'], e['session_date']): e['payload']['status'] for e in events
                    if e['event_type'] == 'session_status' and e['cohort_id'] == cohort_id}
    predictions = {(e['route_id'], e['cohort_id'], e['cutoff_at'], e['symbol']): e
                   for e in events if e['event_type'] == 'prediction'}
    violations = defaultdict(int)
    violated_signals = {e['payload']['signal_id'] for e in events
                        if e['event_type'] == 'protocol_violation' and e['cohort_id'] == cohort_id}
    for buy in buys:
        if buy['session_date'] not in dates or buy['route_id'] not in routes:
            raise ValueError('BUY outside frozen cohort')
        key = (buy['route_id'], buy['cohort_id'], buy['cutoff_at'], buy['symbol'])
        pred_event = predictions.get(key)
        if pred_event is None:
            raise ValueError('BUY has no immutable prediction')
        pred = pred_event['payload']['prediction']
        contract = contracts[buy['route_id']]
        artifacts = pred['quality']['artifacts']
        if buy['model_version'] != contract['model_version'] or \
                pred['model_version'] != contract['model_version'] or \
                pred['score_column'] != contract['score_column'] or \
                buy['payload']['thresholds'] != contract['thresholds'] or \
                any(artifacts.get(name) != contract['artifacts'].get(name)
                    for name in REQUIRED_ARTIFACTS):
            raise ValueError('issued BUY differs from preregistered route contract')
        if _prediction_hash(pred) != pred_event['payload']['prediction_sha256'] or \
                pred_event['payload']['prediction_sha256'] != buy['payload']['prediction_sha256']:
            raise ValueError('prediction JSON hash mismatch')
        if not _artifact_valid(pred['quality'], {'artifacts': pred['quality']['artifacts']},
                               pred['lineage'].get('snapshot_sha256')):
            raise ValueError('source or model bytes changed since issue')
        if timestamp(buy['payload']['published_at']) > timestamp(buy['payload']['deadline_at']) or \
                buy['payload'].get('protocol_violation') or buy['event_id'] in violated_signals:
            violations[buy['route_id']] += 1
    outcomes = {}
    for event in events:
        if event['event_type'] != 'outcome':
            continue
        payload = event['payload']
        outcome = payload['outcome']
        if outcome.get('source_path'):
            if file_sha(Path(outcome['source_path'])) != outcome.get('source_sha256'):
                raise ValueError('outcome source bytes changed after label')
        elif ledger.mode == 'live':
            raise ValueError('live outcome lacks verifiable source path')
        if outcome.get('corporate_action_source_path'):
            if file_sha(Path(outcome['corporate_action_source_path'])) != \
                    outcome.get('corporate_action_source_sha256'):
                raise ValueError('corporate-action source bytes changed after label')
        elif ledger.mode == 'live':
            raise ValueError('live outcome lacks corporate-action source path')
        key = (payload['signal_id'], payload['target'])
        if key in outcomes and payload.get('supersedes') != outcomes[key]['_event_id']:
            raise ValueError('outcome revision does not supersede previous event')
        outcomes[key] = dict(payload, _event_id=event['event_id'])
    cells = {}
    main = {}
    period_end = timestamp(preregistration['last_session_close_at'])
    period_ended = ledger.now() >= period_end
    for route in routes:
        subset = [b for b in buys if b['route_id'] == route]
        accounted = sum((route, day) in session_rows for day in dates)
        missing_runs = sum(session_rows.get((route, day)) == 'missing_run' for day in dates)
        for group in groups:
            key = f'{route}:{group}'
            all_targets = {target: _cell(subset, outcomes, target, group, len(dates),
                                         violations[route]) for target in TARGETS}
            cell = all_targets['3d_5pct']
            midpoint = len(dates) // 2
            halves = []
            for half_dates in (dates[:midpoint], dates[midpoint:]):
                half_buys = [b for b in subset if b['session_date'] in half_dates]
                half = _cell(half_buys, outcomes, '3d_5pct', group,
                             len(half_dates), violations[route])
                half.pop('_date_counts')
                halves.append({'start': half_dates[0], 'end': half_dates[-1], **half})
            cell['halves'] = halves
            cell['gate'] = {'period_ended': period_ended,
                            'full_session_accounting': accounted == len(dates),
                            'all_outcomes_resolved': cell['unresolved'] == 0,
                            'minimum_mature_signals': cell['tp'] + cell['fp'] >= 30,
                            'minimum_signal_dates': cell['signal_dates'] >= 20,
                            'weekly_supply': cell['signals_per_5_sessions'] >= 1,
                            'point_precision_90': cell['known_precision'] is not None
                            and cell['known_precision'] >= .9,
                            'no_forward_violation': violations[route] == 0,
                            'live_evidence': ledger.mode == 'live'
                            and preregistration.get('evidence_mode') == 'live'}
            cell['pass'] = all(cell['gate'].values())
            cell['accounted_sessions'] = accounted
            cell['missing_run_sessions'] = missing_runs
            main[key] = cell
            cells[key] = {'main': cell, 'targets': all_targets}
    bootstrap = {str(block): _bootstrap(main, dates, block=block,
                                         repeats=int(preregistration.get('bootstrap_repeats', 2000)),
                                         seed=int(preregistration.get('bootstrap_seed', 3566)))
                 for block in (5, 10)}
    for cell in main.values():
        cell.pop('_date_counts')
    for cell in cells.values():
        for target in cell['targets'].values():
            target.pop('_date_counts', None)
    passing_routes = [route for route in routes
                      if all(main[f'{route}:{group}']['pass'] for group in groups)]
    return {'cohort_id': cohort_id, 'ledger_cut': ledger_cut,
            'registered_session_dates': dates, 'official_sessions': len(dates),
            'period_ended': period_ended, 'issued_union_buys': len(buys),
            'cells': cells, 'bootstrap': bootstrap,
            'passing_routes': passing_routes,
            'final_target_pass': len(passing_routes) >= 3,
            'evidence_mode': ledger.mode,
            'point_90_is_not_confidence_lower_bound': True}


def seal_report(report: dict, path: Path, *, ledger: Ledger) -> dict:
    """Export once and anchor exact report bytes in the append-only ledger."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(report) + '\n'
    with path.open('x') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    with ledger.transaction():
        return ledger.append({'route_id': '_system', 'model_version': '',
            'cohort_id': report['cohort_id'], 'run_id': '', 'sample_id': '',
            'symbol': '', 'session_date': '', 'cutoff_at': '', 'decision_at': '',
            'event_type': 'report_sealed'},
            {'path': str(path), 'sha256': file_sha(path),
             'report_sha256': digest(report), 'ledger_cut': report['ledger_cut']})


def verify_report(path: Path, *, ledger: Ledger) -> dict:
    path = Path(path).resolve()
    matches = [e for e in ledger.events(event_type='report_sealed')
               if e['payload']['path'] == str(path)]
    if len(matches) != 1 or file_sha(path) != matches[0]['payload']['sha256']:
        raise ValueError('report absent, duplicated, or changed after seal')
    parsed = json.loads(path.read_text())
    if digest(parsed) != matches[0]['payload']['report_sha256']:
        raise ValueError('report JSON no longer matches sealed content')
    return parsed
