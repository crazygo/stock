"""Pure full-window nine-target labels and append-only BUY outcome wrapper."""
from __future__ import annotations

import json
import math
from pathlib import Path

from ...timeaxis import add_regular_minutes, intervals
from .ledger import ET, Ledger, TARGETS, calendar_sessions, digest, file_sha, timestamp

HORIZONS = {'1d': 78, '3d': 234, '5d': 390}
PCTS = {'3pct': .03, '5pct': .05, '8pct': .08}


def _snapshot_for(outcome_snapshot, symbol):
    if 'bars' in outcome_snapshot:
        return outcome_snapshot
    return outcome_snapshot[symbol]


def evaluate_candidate(candidate: dict, outcome_snapshot: dict, calendar,
                       evaluation_asof) -> dict:
    """Evaluate any candidate, including non-BUY history, without shortening windows.

    This function does not write the BUY ledger or a historical-prior store.
    Caller must persist *all* eligible candidates separately for mature priors.
    """
    symbol = candidate['symbol']
    snapshot = _snapshot_for(outcome_snapshot, symbol)
    source_hash = snapshot.get('source_sha256')
    if not source_hash:
        raise ValueError('outcome snapshot lacks source SHA')
    if snapshot.get('path') and file_sha(Path(snapshot['path'])) != source_hash:
        raise ValueError('outcome source bytes changed')
    sessions = calendar_sessions({'calendar_sessions': calendar})
    entry_at = timestamp(candidate['planned_entry_at'])
    asof = timestamp(evaluation_asof)
    bars = list(snapshot.get('bars', []))
    if snapshot.get('bars_sha256') != digest(bars):
        raise ValueError('outcome bars differ from snapshot hash')
    actions = snapshot.get('corporate_actions') or {}
    action_fields = ('coverage_start', 'coverage_end', 'split_dates', 'halt_dates',
                     'coverage_verified')
    action_record = {key: actions.get(key) for key in action_fields}
    action_sha = actions.get('source_sha256')
    action_proof = bool(actions.get('coverage_verified') is True and
                        action_sha and all(key in actions for key in action_fields) and
                        isinstance(actions.get('coverage_start'), str) and
                        isinstance(actions.get('coverage_end'), str) and
                        isinstance(actions.get('split_dates'), list) and
                        isinstance(actions.get('halt_dates'), list))
    if action_proof and actions.get('path'):
        path = Path(actions['path'])
        try:
            action_proof = file_sha(path) == action_sha and \
                json.loads(path.read_text()) == action_record
        except (OSError, ValueError):
            action_proof = False
    elif action_proof:
        action_proof = action_sha == digest(action_record)
    by_start = {}
    duplicate = set()
    for bar in bars:
        if bar.get('symbol', symbol) != symbol:
            continue
        start = timestamp(bar['start_at'])
        if start in by_start:
            duplicate.add(start)
        by_start[start] = bar
    split_days = set(actions.get('split_dates') or [])
    halt_days = set(actions.get('halt_dates') or [])
    outcomes = {}
    for target in TARGETS:
        horizon, pct = target.split('_', 1)
        count = HORIZONS[horizon]
        payload = {'target': target, 'symbol': symbol,
                   'entry_at': entry_at.isoformat(), 'source_sha256': source_hash,
                   'source_path': snapshot.get('path'),
                   'corporate_action_source_sha256': action_sha,
                   'corporate_action_source_path': actions.get('path'),
                   'expected_rth_bars': count, 'evaluated_at': asof.isoformat(),
                   'price_basis': snapshot.get('price_basis'),
                   'status': 'pending', 'y': None}
        try:
            end = add_regular_minutes(entry_at, count * 5, sessions)
            grid = intervals(entry_at, end, sessions, 5)
        except ValueError:
            payload.update(status='pending_calendar', reason='calendar_unavailable')
            outcomes[target] = payload
            continue
        if len(grid) != count:
            raise ValueError('official calendar produced wrong target grid')
        payload['label_end_at'] = end.isoformat()
        payload['expected_grid_sha256'] = digest([[a.isoformat(), b.isoformat()] for a, b in grid])
        if asof < end:
            payload.update(status='pending', reason='full_window_not_finished')
            outcomes[target] = payload
            continue
        days = {next(s.open_at.astimezone(ET).date().isoformat()
                     for s in sessions if s.open_at <= a < s.close_at)
                for a, _ in grid}
        if not action_proof or not actions.get('coverage_start') <= min(days) or \
                not max(days) <= actions.get('coverage_end'):
            payload.update(status='evidence_unknown', reason='corporate_action_coverage_unknown')
            outcomes[target] = payload
            continue
        if days & split_days:
            payload.update(status='invalid', reason='split_price_units_unresolved')
            outcomes[target] = payload
            continue
        if days & halt_days:
            payload.update(status='halts', reason='halt_in_window')
            outcomes[target] = payload
            continue
        selected = []
        bad = None
        for start, stop in grid:
            if start in duplicate:
                bad = 'duplicate_source_bar'
                break
            bar = by_start.get(start)
            if bar is None:
                bad = 'missing_bar'
                break
            try:
                actual_end = timestamp(bar['end_at'])
                available = timestamp(bar['available_at'])
                received = timestamp(bar['received_at'])
                prices = [float(bar[name]) for name in ('open', 'high', 'low', 'close')]
                if actual_end != stop or available < stop or received < stop or \
                        not all(math.isfinite(v) and v > 0 for v in prices) or \
                        prices[1] < max(prices[0], prices[2], prices[3]) or \
                        prices[2] > min(prices[0], prices[1], prices[3]) or \
                        bar.get('price_basis') != snapshot.get('price_basis'):
                    bad = 'invalid_source_bar'
                    break
                selected.append((bar, start, stop, max(available, received)))
            except (KeyError, TypeError, ValueError):
                bad = 'source_evidence_unknown'
                break
        if bad:
            status = ('pending' if asof < end and bad == 'missing_bar' else
                      'missing' if bad == 'missing_bar' else 'invalid')
            payload.update(status=status, reason=bad, observed_rth_bars=len(selected))
            outcomes[target] = payload
            continue
        available_at = max(x[3] for x in selected)
        if asof < available_at:
            payload.update({'status': 'pending', 'reason': 'full_window_not_available',
                            'label_available_at': available_at.isoformat()})
            outcomes[target] = payload
            continue
        entry = float(selected[0][0]['open'])
        if not math.isfinite(entry) or entry <= 0:
            payload.update(status='invalid', reason='entry_open_invalid')
            outcomes[target] = payload
            continue
        payload.update({'entry_price': entry,
                        'entry_bar': {'start_at': selected[0][1].isoformat(),
                                      'end_at': selected[0][2].isoformat(),
                                      'source_id': selected[0][0].get('source_id')},
                        'threshold_price': entry * (1 + PCTS[pct]),
                        'label_available_at': available_at.isoformat(),
                        'observed_rth_bars': count})
        first = next(((a, b, bar) for bar, a, b, _ in selected
                      if float(bar['high']) >= payload['threshold_price']), None)
        if first:
            a, b, bar = first
            payload.update(status='tp', y=1, first_touch={
                'start_at': a.isoformat(), 'end_at': b.isoformat(),
                'source_id': bar.get('source_id')})
        else:
            payload.update(status='fp', y=0, first_touch=None)
        outcomes[target] = payload
    mature = all(outcomes[t]['status'] in ('tp', 'fp') for t in TARGETS)
    latest_available = max((timestamp(o['label_available_at']) for o in outcomes.values()
                            if o.get('label_available_at')), default=None)
    latest_end = max((timestamp(o['label_end_at']) for o in outcomes.values()
                      if o.get('label_end_at')), default=None)
    return {'sample_id': candidate.get('sample_id'), 'symbol': symbol,
            'decision_at': candidate.get('decision_at'),
            'label_end_at': latest_end.isoformat() if latest_end else None,
            'label_available_at': latest_available.isoformat() if latest_available else None,
            'y': [outcomes[t]['y'] for t in TARGETS] if mature else None,
            'all_nine_mature': mature, 'targets': outcomes}


def mature_signals(signal_ids, outcome_snapshot, calendar, evaluation_asof, *, ledger: Ledger):
    """Append revisions for existing BUY ids; never creates or removes BUY rows."""
    if ledger.mode == 'live' and abs((timestamp(evaluation_asof) - ledger.now()).total_seconds()) > 5:
        raise ValueError('live labels require actual current evaluation time')
    ids = set(signal_ids)
    buys = {e['event_id']: e for e in ledger.events(event_type='buy')}
    if ids - buys.keys():
        raise ValueError('label request includes a non-BUY event')
    appended = []
    with ledger.transaction():
        for signal_id in sorted(ids):
            buy = buys[signal_id]
            if ledger.mode == 'live':
                source = _snapshot_for(outcome_snapshot, buy['symbol'])
                if not source.get('path') or not (source.get('corporate_actions') or {}).get('path'):
                    raise ValueError('live outcome lacks price/action source paths')
            candidate = {'sample_id': buy['sample_id'], 'symbol': buy['symbol'],
                         'decision_at': buy['decision_at'],
                         'planned_entry_at': buy['payload']['planned_entry_at']}
            evaluated = evaluate_candidate(candidate, outcome_snapshot, calendar, evaluation_asof)
            for target, outcome in evaluated['targets'].items():
                old = ledger.db.execute('''SELECT * FROM events WHERE event_type='outcome'
                    AND sample_id=? AND route_id=? ORDER BY seq DESC''',
                    (signal_id, buy['route_id'])).fetchall()
                previous = next((x for x in old if json_target(x) == target), None)
                content = {'signal_id': signal_id, 'target': target, 'outcome': outcome,
                           'supersedes': previous['event_id'] if previous else None}
                if previous:
                    prior = json.loads(previous['payload_json'])
                    if prior['outcome'] == outcome:
                        continue
                fields = {key: buy[key] for key in ('route_id', 'model_version', 'cohort_id',
                    'run_id', 'symbol', 'session_date', 'cutoff_at', 'decision_at')}
                fields['sample_id'] = signal_id
                fields['event_type'] = 'outcome'
                appended.append(ledger.append(fields, content))
    return appended


def json_target(row) -> str:
    return json.loads(row['payload_json'])['target']
