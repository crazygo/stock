"""Registered chronological model selection and full capital-ledger research."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import core
from ledger import replay, statistics
import model

SOURCE = Path(__file__).resolve().parent
CACHE = core.ROOT / '.cache/soxl_soxs_14d5pct_v2'


def save_replay(output, name, result):
    equity, trades, decisions, orders = result
    equity.to_csv(output / f'{name}_equity.csv')
    for kind, frame in [('trades', trades), ('decisions', decisions), ('orders', orders)]:
        writable = frame.drop(columns=['dividends', 'attached_order'], errors='ignore')
        writable.to_csv(output / f'{name}_{kind}.csv', index=False)


def study(data_root, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new run directory; existing evidence is never overwritten')
    output.mkdir(parents=True)
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in SOURCE.iterdir() if p.suffix in {'.py', '.json'}}
    core.dump(output / 'registration.json', {'created_at': datetime.now(timezone.utc).isoformat(),
              'protocol': protocol, 'source_hashes': source_hashes,
              'evaluation_status': 'historical_development_already_exposed'})
    market = core.load_market(data_root)
    core.dump(output / 'input_manifest.json', market.manifest)
    print('Building causal candidate features and 14-day labels', flush=True)
    candidates, labels, metadata = core.candidate_table(market, protocol['decision_clocks_et'])
    candidates.to_parquet(output / 'features.parquet', compression='zstd', compression_level=7)
    core.dump(output / 'candidate_coverage.json', metadata)
    core.dump(output / 'labels.json', labels)
    outcomes = {s: pd.DataFrame([x for x in labels if x['symbol'] == s]).set_index('decision_at') for s in core.ETFS}
    columns = model.feature_columns(candidates)
    eligible = candidates.index.copy()
    for symbol in core.ETFS:
        table = outcomes[symbol]
        good = table.complete_window.fillna(False) & table.reason.isin(['target', 'expiry']) & table.net_return.notna()
        eligible = eligible.intersection(table.index[good])
    all_actions = {}; predictions = []; trials = []; monthly = []; frozen = {}
    for period in pd.period_range('2026-05', '2026-10', freq='M'):
        boundary = period.start_time
        allowed = eligible[outcomes['SOXL'].loc[eligible, 'expiry_date'] < boundary]
        allowed_days = pd.DatetimeIndex(candidates.loc[allowed, 'date'].unique()).sort_values()
        selection_days = allowed_days[-40:]
        if len(selection_days) < 40:
            raise ValueError(f'{period}: fewer than 40 eligible selection dates')
        selection = allowed[candidates.loc[allowed, 'date'].isin(selection_days)]
        train = allowed[outcomes['SOXL'].loc[allowed, 'expiry_date'] < selection_days[0]]
        training_days = int(candidates.loc[train, 'date'].nunique())
        if training_days < 100:
            raise ValueError(f'{period}: fewer than 100 training dates')
        models = model.fit(candidates.loc[train, columns], outcomes)
        selection_prediction = model.predict(models, candidates.loc[selection, columns])
        block = candidates.loc[selection]
        replay_end = outcomes['SOXL'].loc[selection, 'expiry_date'].max()
        possible = []
        trial_defs = [('trend20', None, '08:55:30')]
        trial_defs += [(route, threshold, clocks) for route in ['conditional', 'direct']
                       for threshold in protocol['ev_thresholds'] for clocks in protocol['policy_clocks']]
        for route, threshold, clocks in trial_defs:
            actions = model.actions(block, selection_prediction, route, threshold, clocks)
            result = replay(market, block, labels, actions, selection_days[0], replay_end)
            score = statistics(result[0], result[1], [])
            value = {'month': str(period), 'route': route, 'threshold': threshold, 'clocks': clocks,
                     'selection_return': score['continuous_return'], 'selection_completed': score['completed_trades'],
                     'selection_pending': score['pending_trades'], 'selection_mdd': score['close_max_drawdown']}
            trials.append(value)
            if score['completed_trades'] >= 3 and score['pending_trades'] == 0 and score['continuous_return'] > 0:
                possible.append(value)
        chosen = sorted(possible, key=lambda x: (x['selection_return'], -x['selection_completed'], x['threshold'] or 0), reverse=True)[0] if possible else {'route': 'cash', 'threshold': None, 'clocks': 'all_registered_clocks'}
        test = candidates[candidates.date.dt.to_period('M') == period]
        if len(test):
            pred = model.predict(models, test[columns])
            pred['month'] = str(period); predictions.append(pred)
            all_actions.update(model.actions(test, pred, chosen['route'], chosen['threshold'], chosen['clocks']))
        record = {'month': str(period), 'training_distinct_dates': training_days, 'training_rows_each_etf': len(train),
                  'train_start': candidates.loc[train, 'date'].min(),
                  'train_last_contractual_deadline': outcomes['SOXL'].loc[train, 'expiry_date'].max(),
                  'selection_start': selection_days[0], 'selection_last_contractual_deadline': replay_end,
                  'chosen': chosen, 'trials': len(trial_defs), 'feature_columns': columns}
        assert record['train_last_contractual_deadline'] < record['selection_start']
        assert record['selection_last_contractual_deadline'] < boundary
        monthly.append(record); frozen[str(period)] = models
        print(f'{period}: train_dates={training_days} selection={chosen}', flush=True)
    prediction = pd.concat(predictions)
    prediction.to_csv(output / 'predictions.csv', index_label='decision_at')
    core.dump(output / 'all_policy_trials.json', trials)
    core.dump(output / 'monthly_registration.json', monthly)
    with (output / 'models.pkl').open('wb') as file:
        pickle.dump(frozen, file)
    core.dump(output / 'model_manifest.json', {'sha256': hashlib.sha256((output / 'models.pkl').read_bytes()).hexdigest(),
                                              'source': 'generated_by_this_registered_run', 'months': list(frozen)})
    evaluation = candidates[(candidates.date >= pd.Timestamp('2026-05-01')) & (candidates.date <= pd.Timestamp('2026-09-30'))]
    eight55 = evaluation.clock == '08:55:30'
    variants = {'adaptive_model': all_actions, 'cash': {},
                'always_SOXL': {at: ('SOXL' if row.clock == '08:55:30' else 'CASH') for at, row in evaluation.iterrows()},
                'always_SOXS': {at: ('SOXS' if row.clock == '08:55:30' else 'CASH') for at, row in evaluation.iterrows()},
                'trend20_premarket': model.actions(evaluation, None, 'trend20', None, '08:55:30')}
    results = {}
    for name, actions in variants.items():
        result = replay(market, evaluation, labels, actions, '2026-05-01', '2026-09-30')
        save_replay(output, name, result)
        results[name] = statistics(result[0], result[1], protocol['requested_months'])
    baseline = results['trend20_premarket']; adaptive = results['adaptive_model']
    results['acceptance'] = {'primary_metric': protocol['primary_metric'],
                              'adaptive_model_pass': adaptive['primary_pass'],
                              'registered_trend_baseline_pass': baseline['primary_pass'],
                              'any_registered_strategy_pass': any(x['primary_pass'] for x in results.values()),
                              'no_every_month_requirement': True,
                              'prospective_effectiveness_established': False,
                              'all_wins_losses_and_pending_retained': True}
    calibration = []
    for symbol in core.ETFS:
        covered = prediction.index.intersection(eligible)
        covered = covered[prediction.loc[covered, 'month'] <= '2026-09']
        yy = outcomes[symbol].loc[covered, 'target_hit'].astype(int)
        pp = prediction.loc[covered, f'{symbol}_p']
        calibration.append({'symbol': symbol, 'candidate_rows': len(covered),
                            'distinct_days': candidates.loc[covered, 'date'].nunique(),
                            'brier': brier_score_loss(yy, pp),
                            'baseline_brier': brier_score_loss(yy, prediction.loc[covered, f'{symbol}_base_p']),
                            'warning': 'Multiple clocks and overlapping 14-day windows are dependent observations'})
    results['probability_diagnostics'] = calibration
    core.dump(output / 'results.json', results)
    core.dump(output / 'status.json', {'stage': 'replay_complete', 'acceptance': results['acceptance']})
    print(json.dumps(results['acceptance'], indent=2), flush=True)
    print({name: value.get('primary_average') for name, value in results.items() if isinstance(value, dict)}, flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    parser.add_argument('--output', type=Path, default=CACHE / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args(); study(args.data_root, args.output)
