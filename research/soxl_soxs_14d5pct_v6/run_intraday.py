"""Registered intraday entry study with daily fitting and prior-shadow-equity routing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from path_engine import (core, RULES, EXITS, EXPERTS, build_context, feature_at, rule_action,
                         exit_label, day_cutoff, digest)
from learning import fit_daily, predict, select_shadow
from walkforward import label_tables
from daily_study import score as v4_score
from risk_ledger import replay
from study import save_replay

SOURCE = Path(__file__).resolve().parent


def score(book, months):
    stats = v4_score(book, months)
    t = book[1]
    if len(t):
        cutoff = pd.Timestamp(str(book[0].index[-1].date()) + ' 23:59:59', tz=core.ET)
        t = t[t.exit_at.notna() & (pd.to_datetime(t.exit_at, utc=True) <= cutoff)]
    stats['risk_stop_exits'] = int(t.reason.eq('risk_stop').sum()) if len(t) else 0
    stats['trailing_stop_exits'] = int(t.reason.eq('trailing_stop').sum()) if len(t) else 0
    stats['trend_exits'] = int(t.reason.eq('trend_exit').sum()) if len(t) else 0
    stats['session_close_exits'] = int(t.reason.eq('session_close').sum()) if len(t) else 0
    return stats


def run(output, data_root=core.DATA_ROOT):
    output = Path(output)
    if output.exists():
        raise ValueError('Output must be new; keep prior attempts immutable')
    output.mkdir(parents=True); (output / 'training').mkdir()
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    prereg = json.loads((SOURCE / 'evidence/preregistration.json').read_text())
    assert prereg['protocol_sha256'] == hashlib.sha256((SOURCE / 'protocol.json').read_bytes()).hexdigest()
    files = [*SOURCE.glob('*.py'), SOURCE / 'protocol.json', SOURCE.parent / 'soxl_soxs_14d5pct_v3/risk_ledger.py',
             SOURCE.parent / 'soxl_soxs_14d5pct_v2/core.py']
    core.dump(output / 'registration.json', {'started_at': datetime.now(timezone.utc).isoformat(), 'preregistration': prereg,
              'source_sha256': {str(p.relative_to(core.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    (output / 'executed_sources').mkdir()
    for p in SOURCE.glob('*.py'):
        (output / 'executed_sources' / p.name).write_bytes(p.read_bytes())
    market = core.load_market(data_root); context = build_context(market)
    rows = []; coverage = []
    for date in market.daily['SOXL'].index:
        for clock in protocol['analysis_clocks_et']:
            at = pd.Timestamp(str(date.date()) + ' ' + clock, tz=core.ET).tz_convert('UTC')
            if at >= pd.Timestamp(market.calendar[date]['close_at']):
                continue
            row, audit = feature_at(market, context, at)
            coverage.append({'decision_at': at, 'ready': row is not None, 'audit': audit})
            if row is not None:
                rows.append({'decision_at': at, 'date': date, 'clock': clock, **row})
    features = pd.DataFrame(rows).set_index('decision_at').sort_index()
    # Missing regular context in the explicit premarket WAIT row is represented
    # by zero plus the context-ready flag; these rows never train or buy.
    context_columns = [c for c in features if c.startswith(('x_', 'q_'))]
    premarket = features.regular_context_ready.eq(0.)
    features.loc[premarket, context_columns] = features.loc[premarket, context_columns].fillna(0.)
    if features.isna().any().any():
        raise ValueError('Unexpected missing feature; do not convert it to a zero observation')
    features.to_parquet(output / 'features.parquet', compression='zstd', compression_level=7)
    core.dump(output / 'coverage.json', coverage)
    core.dump(output / 'input_manifest.json', {'market': market.manifest, 'feature_rows': len(features),
              'feature_sha256': hashlib.sha256((output / 'features.parquet').read_bytes()).hexdigest()})
    regular = features[features.regular_context_ready.eq(1.)]
    labels = {}; tables = {}
    for policy in EXITS:
        labels[policy] = [exit_label(market, context, symbol, at, policy) for at in regular.index for symbol in core.ETFS]
        core.dump(output / (policy + '_labels.json'), labels[policy])
        tables[policy] = label_tables(labels[policy])
        print('labels', policy, len(labels[policy]), flush=True)
    histories = {e: pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC')) for e in EXPERTS}
    train_registry = []; started = time.monotonic()
    for i, (date, day) in enumerate(features.groupby('date')):
        at = day.index[0]
        registrations = {'date': date, 'training_cutoff': day_cutoff(at), 'exit_policies': {}}
        for policy in EXITS:
            fitted, record, eligible = fit_daily(regular, tables[policy], at, protocol)
            registrations['exit_policies'][policy] = record
            for when, row in day.iterrows():
                for rule in RULES:
                    expert = rule + '__' + policy
                    event = {'clock': row.clock, 'action': 'CASH', 'SOXL_raw': np.nan, 'SOXS_raw': np.nan,
                             'ready': False, 'training_cutoff': day_cutoff(when)}
                    if not row.regular_context_ready:
                        event['reason'] = 'WAIT_FOR_REGULAR_INFORMATION'
                    elif rule not in ['ridge_log', 'boosted_log']:
                        event.update({'action': rule_action(row, rule), 'ready': True})
                    elif fitted is not None:
                        event.update(predict(fitted, rule, row, histories[expert], tables[policy], eligible, protocol))
                    new = pd.DataFrame([event], index=pd.DatetimeIndex([when]))
                    histories[expert] = new if histories[expert].empty else pd.concat([histories[expert], new])
        registrations['daily_id'] = digest(registrations)[:24]
        core.dump(output / 'training' / (str(date.date()) + '.json'), registrations)
        train_registry.append({'date': date, 'daily_id': registrations['daily_id'],
              'training_dates': {p: registrations['exit_policies'][p]['training_dates'] for p in EXITS}})
        if (i + 1) % 25 == 0:
            print('trained through', date.date(), 'elapsed', round(time.monotonic() - started), 's', flush=True)
    core.dump(output / 'training_registry.json', train_registry)
    for expert, history in histories.items():
        history.to_json(output / (expert + '_predictions.json'), orient='table', date_format='iso', index=True)
    # Shadow equity is generated from saved as-of predictions. The selector below
    # may inspect only dates strictly earlier than its decision date.
    books = {}; history_start = str(features.date.min().date())
    end = protocol['evaluation_end']; start = protocol['evaluation_start']
    for expert in EXPERTS:
        policy = expert.split('__')[1]
        books[expert] = replay(market, features, labels[policy], histories[expert].action.to_dict(), history_start, end)
        save_replay(output, 'shadow_' + expert, books[expert])
    chosen = {}; actions = {}; registry = []
    for date, day in features[(features.date >= start) & (features.date <= end)].groupby('date'):
        expert, choices = select_shadow(books, date, protocol)
        policy = expert.split('__')[1] if expert else 'flat_close'
        for at in day.index:
            chosen[at] = policy; actions[at] = histories[expert].loc[at, 'action'] if expert else 'CASH'
        registry.append({'date': date, 'selected_expert': expert, 'selected_exit': policy, 'selector': choices})
    core.dump(output / 'selector_registry.json', registry)
    core.dump(output / 'decisions_frozen.json', {'frozen_at': datetime.now(timezone.utc).isoformat(),
              'primary_candidate': 'previous_shadow_capital_selector', 'case_results_seen_before_rules_frozen': False,
              'actions': [{'decision_at': at, 'action': a, 'exit_policy': chosen[at]} for at, a in actions.items()]})
    test = features[(features.date >= start) & (features.date <= end)]
    test_labels = [x for p in EXITS for x in labels[p] if x['decision_at'] in chosen and chosen[x['decision_at']] == p]
    # Premarket actions are always CASH, so no hypothetical premarket order label is required.
    book = replay(market, test, test_labels, actions, start, end)
    save_replay(output, 'primary', book)
    results = {'primary': score(book, protocol['regression_months']), 'diagnostic_experts': {}}
    for expert in EXPERTS:
        policy = expert.split('__')[1]
        diagnostic = replay(market, test, labels[policy], histories[expert].action.to_dict(), start, end)
        save_replay(output, expert, diagnostic)
        results['diagnostic_experts'][expert] = score(diagnostic, protocol['regression_months'])
    double_labels = [exit_label(market, context, s, at, chosen[at], 2) for at in test[test.regular_context_ready.eq(1.)].index for s in core.ETFS]
    stress = replay(market, test, double_labels, actions, start, end, cost_multiple=2)
    save_replay(output, 'double_cost', stress)
    results['double_cost'] = score(stress, protocol['regression_months'])
    expected_days = market.sessions[(market.sessions >= start) & (market.sessions <= end)]
    coverage_complete = test.date.nunique() == len(expected_days) and all((test[test.date == d].clock == '09:25:30').any() for d in expected_days)
    passed = results['primary']['primary_pass'] and coverage_complete
    results['acceptance'] = {'passed': bool(passed), 'new_buy_eligible': bool(passed),
        'primary_policy': 'previous_shadow_capital_selector', 'expert_count': 18,
        'diagnostic_pass_count': sum(x['primary_pass'] for x in results['diagnostic_experts'].values()),
        'daily_and_premarket_coverage_complete': coverage_complete, 'only_primary_can_grant_admission': True,
        'independent_forward_validation_passed': False, 'future_nonnegative_month_guarantee': False}
    core.dump(output / 'results.json', results)
    core.dump(output / 'status.json', {'stage': 'complete', 'acceptance': results['acceptance']})
    print(json.dumps({'primary': results['primary'], 'stress': results['double_cost'], 'acceptance': results['acceptance']}, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.data_root)
