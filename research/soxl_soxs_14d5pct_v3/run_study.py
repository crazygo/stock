"""Choose from pre-May data once, then run sealed May/July/September regressions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from risk import core, direction, risk_label
from risk_ledger import replay, statistics
from study import save_replay

SOURCE = Path(__file__).resolve().parent
CACHE = core.ROOT / '.cache/soxl_soxs_14d5pct_v3'
OLD_RUN = core.ROOT / '.cache/soxl_soxs_14d5pct_v2/runs/phase3_clock_20261012'
CASES = ['2026-05', '2026-07', '2026-09']


def score(result, requested=CASES):
    s = statistics(result[0], result[1], requested)
    monthly = s['monthly']
    values = [m['return'] for m in monthly]
    required = [m for m in monthly if m['month'] in requested]
    s['arithmetic_gate_from_v2'] = s.pop('primary_pass')
    s['all_calendar_months_nonnegative'] = bool(values and all(v >= -1e-12 for v in values))
    s['requested_months_nonnegative'] = bool(len(required) == len(requested) and all(m['return'] >= -1e-12 for m in required))
    s['negative_months'] = [m for m in monthly if m['return'] < -1e-12]
    s['minimum_month_return'] = float(min(values))
    s['all_months_geometric_return'] = float(np.prod([1 + v for v in values]) ** (1 / len(values)) - 1)
    s['regression_cases_pass'] = bool(s['arithmetic_gate_from_v2'] and s['requested_months_nonnegative'])
    s['primary_pass'] = bool(s['regression_cases_pass'] and s['all_calendar_months_nonnegative'] and all(m['mark_complete'] for m in monthly))
    s['primary_metric'] = 'nonnegative_months_and_previously_confirmed_requested_arithmetic_mean_10pct'
    t = result[1]
    if len(t):
        done = t[t.exit_at.notna() & (pd.to_datetime(t.exit_at, utc=True) <= pd.Timestamp(str(result[0].index[-1].date()) + ' 23:59:59', tz=core.ET))]
        s['risk_stop_exits'] = int(done.reason.isin(['risk_stop', 'profit_protection']).sum())
        s['adverse_gap_exits'] = int(done.gap_below_stop.fillna(False).sum()) if 'gap_below_stop' in done else 0
        s['ambiguous_stop_first_exits'] = int(done.same_bar_both_barriers.fillna(False).sum()) if 'same_bar_both_barriers' in done else 0
    return s


def choose(trials):
    if not trials or any(pd.Timestamp(t['selection_end']) >= pd.Timestamp('2026-05-01') for t in trials):
        raise ValueError('Selection must end strictly before the regression cases')
    return sorted(trials, key=lambda t: (-t['score']['minimum_month_return'],
                                        -t['score']['all_months_geometric_return'],
                                        -t['score']['close_max_drawdown'], t['id']))[0]


def run(data_root, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Choose a new run; previous evidence is immutable')
    output.mkdir(parents=True)
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir() if p.suffix in {'.py', '.json'}}
    core.dump(output / 'registration.json', {'registered_at': datetime.now(timezone.utc).isoformat(), 'protocol': protocol, 'source_hashes': code})
    market = core.load_market(data_root)
    previous = json.loads((OLD_RUN / 'input_manifest.json').read_text())
    assert all(market.manifest[s]['sha256'] == previous[s]['sha256'] for s in core.SYMBOLS)
    feature_path = OLD_RUN / 'features.parquet'
    core.dump(output / 'input_manifest.json', {'market': market.manifest, 'feature_sha256': hashlib.sha256(feature_path.read_bytes()).hexdigest(), 'feature_source': str(feature_path)})
    features = pd.read_parquet(feature_path)
    features = features[features.clock == protocol['analysis_clock_et']]
    # Do not construct or score any regression risk label until choice is on disk.
    training = features[(features.date >= pd.Timestamp(protocol['selection_start'])) & (features.date <= pd.Timestamp(protocol['selection_end']))]
    training = training[[core.expiry_day(market, d) < pd.Timestamp('2026-05-01') for d in training.date]]
    risk_maps = {}
    for rule in protocol['risk_rules']:
        risk_maps[rule['id']] = [risk_label(market, symbol, at, rule) for at in training.index for symbol in core.ETFS]
        print('pre-May labels', rule['id'], len(risk_maps[rule['id']]), flush=True)
    selection_months = [str(m) for m in pd.period_range(protocol['selection_start'], protocol['selection_end'], freq='M')]
    trials = []
    definitions = itertools.product(protocol['direction_rules'], protocol['risk_rules'], protocol['capital_fractions'], protocol['cooldown_sessions_after_loss'])
    for name, rule, allocation, cooldown in definitions:
        actions = {at: direction(row, name) for at, row in training.iterrows()}
        result = replay(market, training, risk_maps[rule['id']], actions,
                        protocol['selection_start'], protocol['selection_end'], allocation=allocation, cooldown_sessions=cooldown)
        stats = score(result, selection_months)
        admission = (len(stats['monthly']) == 12 and stats['all_calendar_months_nonnegative']
                     and stats['completed_trades'] >= 12 and stats['pending_trades'] == 0
                     and all(m['mark_complete'] for m in stats['monthly']))
        policy = {'id': f'{name}_{rule["id"]}_size{allocation}_cool{cooldown}', 'direction': name, 'risk': rule,
                  'allocation': allocation, 'cooldown_sessions': cooldown}
        trials.append({**policy, 'selection_start': protocol['selection_start'], 'selection_end': protocol['selection_end'],
                       'last_entry': training.date.max(), 'last_contractual_expiry': max(core.expiry_day(market, d) for d in training.date),
                       'selection_admission': bool(admission), 'score': stats})
    assert len(trials) == protocol['maximum_policy_trials']
    core.dump(output / 'selection_trials.json', trials)
    chosen = choose(trials)
    frozen = {'frozen_at': datetime.now(timezone.utc).isoformat(), 'chosen': chosen,
              'selection_uses_regression_months': False, 'regression_replays_before_freeze': 0,
              'source_hashes': code, 'candidate_count': len(trials),
              'admissible_candidates': sum(t['selection_admission'] for t in trials)}
    core.dump(output / 'frozen_policy.json', frozen)
    print('FROZEN', chosen['id'], 'selection_admission', chosen['selection_admission'], flush=True)
    test = features[(features.date >= pd.Timestamp('2026-05-01')) & (features.date <= pd.Timestamp('2026-09-30'))]
    actions = {at: direction(row, chosen['direction']) for at, row in test.iterrows()}
    labels = [risk_label(market, symbol, at, chosen['risk']) for at in test.index for symbol in core.ETFS]
    core.dump(output / 'regression_labels.json', labels)
    test.to_parquet(output / 'features.parquet', compression='zstd', compression_level=7)
    result = replay(market, test, labels, actions, '2026-05-01', '2026-09-30',
                    allocation=chosen['allocation'], cooldown_sessions=chosen['cooldown_sessions'])
    save_replay(output, 'frozen_candidate', result)
    candidate = score(result)
    cash_result = replay(market, test, labels, {}, '2026-05-01', '2026-09-30')
    cash = score(cash_result)
    live = chosen['selection_admission'] and candidate['primary_pass']
    # Historical admission uses April information only. Current admission may
    # also use the later regression, but cannot erase earlier historical trades.
    admitted_result = result if chosen['selection_admission'] else cash_result
    save_replay(output, 'admitted_policy', admitted_result)
    doubled = [risk_label(market, symbol, at, chosen['risk'], 2) for at in test.index for symbol in core.ETFS]
    stress_result = replay(market, test, doubled, actions, '2026-05-01', '2026-09-30', cost_multiple=2,
                           allocation=chosen['allocation'], cooldown_sessions=chosen['cooldown_sessions'])
    save_replay(output, 'double_cost', stress_result)
    results = {'frozen_candidate': candidate, 'admitted_policy': score(admitted_result), 'double_cost': score(stress_result),
               'selected_policy': chosen['id'], 'selection_admission': chosen['selection_admission'],
               'acceptance': {'passed': bool(live), 'new_buy_eligible': bool(live),
                              'selection_admissible_count': frozen['admissible_candidates'],
                              'only_one_frozen_policy_tested_on_regression': True,
                              'rule_changed_after_regression': False,
                              'cash_is_risk_abstention_not_return_success': True,
                              'future_nonnegative_month_guarantee': False,
                              'regression_months_already_exposed': True}}
    core.dump(output / 'results.json', results)
    core.dump(output / 'status.json', {'stage': 'complete', 'acceptance': results['acceptance']})
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data_root, args.output)
