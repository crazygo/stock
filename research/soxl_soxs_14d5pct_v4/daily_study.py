"""Run one registered daily-learning recipe and its frozen-April diagnostic control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from walkforward import core, make_labels, label_tables, fit_day, forecast
from risk_ledger import replay, statistics
from study import save_replay

SOURCE = Path(__file__).resolve().parent


def score(result, months):
    value = statistics(result[0], result[1], months)
    value['arithmetic_gate'] = value.pop('primary_pass')
    value['negative_months'] = [m for m in value['monthly'] if m['return'] < -1e-12]
    value['all_months_nonnegative'] = not value['negative_months']
    value['primary_pass'] = bool(value['arithmetic_gate'] and value['all_months_nonnegative'] and
                                 all(m['mark_complete'] for m in value['monthly']))
    value['risk_stop_exits'] = int((result[1].reason == 'risk_stop').sum()) if len(result[1]) else 0
    value['primary_metric'] = 'requested_arithmetic_mean_10pct_and_all_months_nonnegative'
    return value


def load_features(market, protocol, cache=None):
    if cache:
        cache = Path(cache)
        manifest = json.loads((cache.parent / 'input_manifest.json').read_text())
        manifest = manifest.get('market', manifest)
        if any(manifest[s]['sha256'] != market.manifest[s]['sha256'] for s in core.SYMBOLS):
            raise ValueError('Feature cache market hashes differ; rebuild features from updated local data')
        features = pd.read_parquet(cache)
        return features[features.clock == protocol['analysis_clock_et']].sort_index()
    rows = []
    for date in market.daily['SOXL'].index:
        at = pd.Timestamp(str(date.date()) + ' ' + protocol['analysis_clock_et'], tz=core.ET).tz_convert('UTC')
        row, _ = core.feature_row(market, at)
        if row is not None:
            rows.append({'decision_at': at, 'date': date, 'clock': protocol['analysis_clock_et'], **row})
    if not rows:
        raise ValueError('No causal features; acquire complete local data first')
    return pd.DataFrame(rows).set_index('decision_at').sort_index()


def prediction_diagnostics(predictions, tables, start, end):
    rows = []
    maturity_cutoff = pd.Timestamp(end + ' 23:59:59', tz=core.ET)
    for at, p in predictions.iterrows():
        day = at.tz_convert(core.ET).strftime('%Y-%m-%d')
        if not start <= day <= end:
            continue
        chosen = p.action
        if chosen not in core.ETFS:
            continue
        label = tables[chosen].loc[at]
        known = pd.notna(label.label_available_at) and label.label_available_at <= maturity_cutoff
        rows.append({'decision_at': at, 'symbol': chosen, 'forecast_net_return': p[f'{chosen}_adjusted_ev'],
                     'label_mature_by_evaluation_end': bool(known),
                     'actual_net_return': float(label.net_return) if known else None})
    mature = [r for r in rows if r['label_mature_by_evaluation_end']]
    return {'candidate_signals': len(rows), 'mature_signals': len(mature), 'pending_signals': len(rows) - len(mature),
            'mean_forecast': float(np.mean([r['forecast_net_return'] for r in mature])) if mature else None,
            'mean_realized_hypothetical_trade_return': float(np.mean([r['actual_net_return'] for r in mature])) if mature else None,
            'rmse': float(np.sqrt(np.mean([(r['actual_net_return'] - r['forecast_net_return']) ** 2 for r in mature]))) if mature else None,
            'scope': 'One hypothetical top choice per day; overlapping outcomes are not independent or actual capital returns. Only labels matured by September end scored.',
            'records': rows}


def run(data_root, output, feature_cache=None):
    output = Path(output)
    if output.exists():
        raise ValueError('Run output must be new; retain old models, forecasts and results')
    output.mkdir(parents=True)
    (output / 'models').mkdir()
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    dependencies = [*SOURCE.glob('*.py'), SOURCE / 'protocol.json',
                    SOURCE.parent / 'soxl_soxs_14d5pct_v3/risk.py',
                    SOURCE.parent / 'soxl_soxs_14d5pct_v3/risk_ledger.py',
                    SOURCE.parent / 'soxl_soxs_14d5pct_v2/core.py']
    core.dump(output / 'registration.json', {'registered_at': datetime.now(timezone.utc).isoformat(), 'protocol': protocol,
              'source_hashes': {str(p.relative_to(core.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}})
    market = core.load_market(data_root)
    features = load_features(market, protocol, feature_cache)
    features.to_parquet(output / 'features.parquet', compression='zstd', compression_level=7)
    core.dump(output / 'input_manifest.json', {'market': market.manifest,
              'feature_sha256': hashlib.sha256((output / 'features.parquet').read_bytes()).hexdigest()})
    print('Building complete-window risk labels:', len(features), 'dates', flush=True)
    labels = make_labels(market, features, protocol['risk'])
    core.dump(output / 'labels.json', labels)
    tables = label_tables(labels)
    history = pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
    registry = []; controls = []; frozen = None
    started = time.monotonic()
    for i, (at, row) in enumerate(features.iterrows()):
        bundle, record = fit_day(features, tables, history, at, protocol)
        if bundle is not None:
            p = forecast(bundle, row, at, protocol)
            binary = gzip.compress(pickle.dumps(bundle, protocol=5), mtime=0)
            filename = 'models/' + record['date'] + '.pkl.gz'
            (output / filename).write_bytes(binary)
            record.update({'model_file': filename, 'model_sha256': hashlib.sha256(binary).hexdigest(),
                           'source': 'generated_by_this_registered_run'})
            history = pd.concat([history, pd.DataFrame([p], index=pd.DatetimeIndex([at]))])
            # Persist each day's prediction before processing the following day.
            core.dump(output / 'models' / (record['date'] + '.forecast.json'), p)
            date = str(row.date.date())
            if date >= protocol['evaluation_start'] and frozen is None:
                frozen = bundle
                core.dump(output / 'frozen_control.json', record)
            if frozen is not None and protocol['evaluation_start'] <= date <= protocol['evaluation_end']:
                controls.append(forecast(frozen, row, at, protocol))
        registry.append(record)
        if (i + 1) % 30 == 0:
            print('as-of', record['date'], 'train', record['training_dates'], 'calibration', record.get('calibration_dates', 0),
                  'elapsed', round(time.monotonic() - started), 's', flush=True)
    core.dump(output / 'daily_registry.json', registry)
    history.index.name = 'decision_at'
    history.to_csv(output / 'predictions.csv')
    start, end = protocol['evaluation_start'], protocol['evaluation_end']
    test = features[(features.date >= start) & (features.date <= end)]
    test_labels = [label for label in labels if label['decision_at'] in test.index]
    kwargs = {'allocation': protocol['allocation'], 'cooldown_sessions': protocol['cooldown_sessions']}
    actions = history.action.to_dict()
    control_actions = {p['forecast_at']: p['action'] for p in controls}
    control_predictions = pd.DataFrame(controls).set_index('forecast_at')
    control_predictions.to_csv(output / 'frozen_control_predictions.csv')
    results = {}
    for name, decisions in [('daily_updated', actions), ('frozen_april_control', control_actions), ('cash', {})]:
        book = replay(market, test, test_labels, decisions, start, end, **kwargs)
        save_replay(output, name, book)
        results[name] = score(book, protocol['regression_months'])
    double_labels = make_labels(market, test, protocol['risk'], 2)
    stress = replay(market, test, double_labels, actions, start, end, cost_multiple=2, **kwargs)
    save_replay(output, 'double_cost', stress)
    results['double_cost'] = score(stress, protocol['regression_months'])
    diagnostics = prediction_diagnostics(history, tables, start, end)
    core.dump(output / 'forecast_diagnostics.json', diagnostics)
    evaluation_records = [r for r in registry if start <= r['date'] <= end]
    complete_training = len(evaluation_records) == len(test) and all(r['status'] == 'ready' for r in evaluation_records)
    passed = complete_training and results['daily_updated']['primary_pass']
    results['acceptance'] = {'passed': passed, 'new_buy_eligible': passed,
                            'all_evaluation_days_model_ready': complete_training,
                            'evaluation_days': len(test), 'daily_models': len(evaluation_records),
                            'training_recipe_count': 1, 'control_is_not_a_selection_candidate': True,
                            'known_earlier_case_labels_may_train_later_models': True,
                            'old_predictions_revised_by_new_models': False,
                            'regression_history_already_exposed': True,
                            'independent_forward_validation_passed': False,
                            'future_nonnegative_month_guarantee': False}
    results['learning_examples'] = [{k: r.get(k) for k in ['date', 'model_id', 'training_dates', 'train_first', 'train_last',
                                    'train_last_label_available', 'calibration_dates']} for r in registry
                                   if r['date'] in ['2026-05-01', '2026-07-01', '2026-09-01', '2026-09-30']]
    core.dump(output / 'results.json', results)
    core.dump(output / 'status.json', {'stage': 'complete', 'acceptance': results['acceptance']})
    print(json.dumps({k: {'monthly': v['monthly'], 'mean': v['primary_average'], 'continuous': v['continuous_return'],
                          'entries': v['entries'], 'pass': v['primary_pass']} for k, v in results.items() if 'monthly' in v}, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-cache', type=Path, help='Optional v2/v4 features with matching adjacent input_manifest.json')
    args = parser.parse_args()
    run(args.data_root, args.output, args.feature_cache)
