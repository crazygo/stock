"""One preregistered causal router over twelve models and two structural experts."""
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

from experts import (core, EXPERTS, RISKS, ALGORITHMS, extend_features, outcome, label_tables,
                     eligible_index, day_cutoff, fit_predict, corrected_choice, select_expert, digest)
from daily_study import load_features, score as legacy_score
from risk_ledger import replay
from study import save_replay

SOURCE = Path(__file__).resolve().parent
PREREGISTRATION = SOURCE / 'evidence/preregistration.json'


def score(book, months):
    result = legacy_score(book, months)
    trades = book[1]
    done = trades
    if len(trades):
        end = pd.Timestamp(str(book[0].index[-1].date()) + ' 23:59:59', tz=core.ET)
        done = trades[trades.exit_at.notna() & (pd.to_datetime(trades.exit_at, utc=True) <= end)]
    result['risk_stop_exits'] = int(done.reason.eq('risk_stop').sum()) if len(done) else 0
    result['profit_protection_exits'] = int(done.reason.eq('profit_protection').sum()) if len(done) else 0
    return result


def append(history, at, row):
    new = pd.DataFrame([row], index=pd.DatetimeIndex([at]))
    return new if history.empty else pd.concat([history, new])


def run(output, feature_cache, data_root=core.DATA_ROOT):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new run; never overwrite an earlier experiment')
    output.mkdir(parents=True)
    (output / 'daily').mkdir()
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    pre = json.loads(PREREGISTRATION.read_text())
    assert pre['protocol_sha256'] == hashlib.sha256((SOURCE / 'protocol.json').read_bytes()).hexdigest()
    source_paths = [*SOURCE.glob('*.py'), SOURCE / 'protocol.json', SOURCE.parent / 'soxl_soxs_14d5pct_v4/walkforward.py',
                    SOURCE.parent / 'soxl_soxs_14d5pct_v3/risk.py', SOURCE.parent / 'soxl_soxs_14d5pct_v3/risk_ledger.py',
                    SOURCE.parent / 'soxl_soxs_14d5pct_v2/core.py']
    core.dump(output / 'registration.json', {'run_at': datetime.now(timezone.utc).isoformat(), 'preregistration': pre,
             'source_sha256': {str(p.relative_to(core.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}})
    market = core.load_market(data_root)
    base = load_features(market, protocol, feature_cache)
    features = extend_features(market, base)
    features.to_parquet(output / 'features.parquet', compression='zstd', compression_level=7)
    core.dump(output / 'input_manifest.json', {'market': market.manifest, 'feature_dates': len(features),
              'features_sha256': hashlib.sha256((output / 'features.parquet').read_bytes()).hexdigest()})
    labels = {}; tables = {}
    for risk in RISKS:
        labels[risk] = [outcome(market, row, s, at, risk) for at, row in features.iterrows() for s in core.ETFS]
        core.dump(output / (risk + '_labels.json'), labels[risk])
        tables[risk] = label_tables(labels[risk])
        print('labels', risk, len(labels[risk]), flush=True)
    histories = {e: pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC')) for e in EXPERTS}
    actions = {}; selected_risks = {}; registrations = []
    started = time.monotonic()
    for i, (at, row) in enumerate(features.iterrows()):
        cutoff = day_cutoff(at)
        # Select only from older forecasts before adding today's predictions.
        selected, choices = select_expert(histories, tables, features, cutoff, protocol)
        record = {'decision_at': at, 'training_cutoff': cutoff, 'selected_expert': selected, 'selector': choices, 'training': {}}
        today = {}
        for risk in RISKS:
            eligible = eligible_index(features, tables[risk], cutoff)
            train = eligible[-protocol['training_window_dates']:]
            train_ready = len(train) >= protocol['minimum_training_dates']
            raw, diagnostics = fit_predict(features, train, tables[risk], row) if train_ready else ({}, {})
            record['training'][risk] = {'dates': len(train), 'first': train[0] if len(train) else None,
                'last': train[-1] if len(train) else None,
                'last_label_available': max(tables[risk][s].loc[train, 'label_available_at'].max() for s in core.ETFS) if len(train) else None,
                'members': list(train), 'input_sha256': digest({'dates': list(train),
                      'X': features.loc[train].to_dict('records'),
                      'y': {s: tables[risk][s].loc[train, 'net_return'].tolist() for s in core.ETFS}})}
            for algorithm in ALGORITHMS:
                expert = algorithm + '__' + risk
                prediction = {'decision_at': at, 'training_cutoff': cutoff, 'action': 'CASH',
                              'SOXL_raw': np.nan, 'SOXS_raw': np.nan, 'ready': False}
                if train_ready:
                    action, calibration = corrected_choice(raw[algorithm], histories[expert], tables[risk], eligible, protocol)
                    prediction.update({'action': action, 'SOXL_raw': float(raw[algorithm][0]), 'SOXS_raw': float(raw[algorithm][1]),
                                       **calibration})
                today[expert] = prediction
            if risk in ['expiry14', 'atr_protect']:
                today['trend20__' + risk] = {'decision_at': at, 'training_cutoff': cutoff,
                     'action': 'SOXL' if row.SOXX_ret20 > 0 else 'SOXS', 'ready': True,
                     'SOXL_raw': np.nan, 'SOXS_raw': np.nan}
            record.setdefault('model_diagnostics', {})[risk] = diagnostics
        for expert, prediction in today.items():
            histories[expert] = append(histories[expert], at, prediction)
        action = today[selected]['action'] if selected else 'CASH'
        risk = selected.split('__')[1] if selected else 'expiry14'
        actions[at] = action; selected_risks[at] = risk
        record.update({'action': action, 'risk': risk, 'predictions': today})
        record['model_id'] = digest(record)[:24]
        core.dump(output / 'daily' / (str(row.date.date()) + '.json'), record)
        registrations.append({k: v for k, v in record.items() if k not in ['selector', 'predictions', 'model_diagnostics', 'training']})
        if (i + 1) % 20 == 0:
            print('day', str(row.date.date()), 'train', len(train), 'selected', selected, 'action', action,
                  'elapsed', round(time.monotonic() - started), 's', flush=True)
    core.dump(output / 'daily_registration.json', registrations)
    # Freeze the entire decision history before computing any v5 case capital score.
    core.dump(output / 'decisions_frozen.json', {'frozen_at': datetime.now(timezone.utc).isoformat(),
              'primary_policy': protocol['primary_policy'], 'recipe_changed_after_case_results': False,
              'decision_history_sha256': hashlib.sha256((output / 'daily_registration.json').read_bytes()).hexdigest()})
    for expert, history in histories.items():
        history.drop(columns=['calibration_members'], errors='ignore').to_csv(output / (expert + '_predictions.csv'), index_label='decision_at')
    start, end = protocol['evaluation_start'], protocol['evaluation_end']
    test = features[(features.date >= start) & (features.date <= end)]
    chosen_labels = []
    for risk in RISKS:
        chosen_labels += [x for x in labels[risk] if x['decision_at'] in test.index and selected_risks[x['decision_at']] == risk]
    kwargs = {'allocation': protocol['allocation'], 'cooldown_sessions': protocol['cooldown_sessions']}
    book = replay(market, test, chosen_labels, actions, start, end, **kwargs)
    save_replay(output, 'router', book)
    results = {'router': score(book, protocol['regression_months']), 'diagnostic_experts': {}}
    for expert in EXPERTS:
        risk = expert.split('__')[1]
        expert_book = replay(market, test, labels[risk], histories[expert].action.to_dict(), start, end, **kwargs)
        save_replay(output, expert, expert_book)
        results['diagnostic_experts'][expert] = score(expert_book, protocol['regression_months'])
    double = [outcome(market, row, symbol, at, selected_risks[at], 2) for at, row in test.iterrows() for symbol in core.ETFS]
    double_book = replay(market, test, double, actions, start, end, cost_multiple=2, **kwargs)
    save_replay(output, 'double_cost', double_book)
    results['double_cost'] = score(double_book, protocol['regression_months'])
    results['acceptance'] = {'passed': results['router']['primary_pass'], 'new_buy_eligible': results['router']['primary_pass'],
        'primary_policy': protocol['primary_policy'], 'expert_count': len(EXPERTS),
        'diagnostic_expert_pass_count': sum(v['primary_pass'] for v in results['diagnostic_experts'].values()),
        'diagnostic_winner_does_not_override_router_admission': True, 'regression_cases_already_exposed': True,
        'independent_forward_validation_passed': False, 'future_nonnegative_guaranteed': False}
    core.dump(output / 'results.json', results)
    core.dump(output / 'status.json', {'stage': 'complete', 'acceptance': results['acceptance']})
    print(json.dumps({'router': results['router'], 'double_cost': results['double_cost'], 'acceptance': results['acceptance']}, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-cache', type=Path, default=core.ROOT / '.cache/soxl_soxs_14d5pct_v2/runs/phase3_clock_20261012/features.parquet')
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.feature_cache, args.data_root)
