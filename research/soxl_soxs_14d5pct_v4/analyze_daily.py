"""Any-time research analysis with that ET day's daily model; never sends orders."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import pandas as pd

from walkforward import core, day_cutoff, fit_day, forecast, label_tables
from position import manage

DEFAULT_RUN = core.ROOT / '.cache/soxl_soxs_14d5pct_v4/current'


def daily_bundle(run, at, protocol):
    """Reuse immutable daily binaries; a newly encountered day gets a new fit."""
    date = str(day_cutoff(at).tz_convert(core.ET).date())
    records = json.loads((run / 'daily_registry.json').read_text())
    record = next((r for r in records if r['date'] == date), None)
    online = run / 'online' / date
    if record is None and (online / 'registration.json').exists():
        record = json.loads((online / 'registration.json').read_text())
    if record is not None:
        if not record.get('model_file'):
            return None, record
        path = run / record['model_file']
        blob = path.read_bytes()
        if record.get('source') != 'generated_by_this_registered_run' or hashlib.sha256(blob).hexdigest() != record['model_sha256']:
            raise ValueError('Local daily model does not match its registered hash')
        # Only this research runner's local, hash-verified cache is accepted.
        return pickle.loads(gzip.decompress(blob)), record
    features = pd.read_parquet(run / 'features.parquet')
    tables = label_tables(json.loads((run / 'labels.json').read_text()))
    history = pd.read_csv(run / 'predictions.csv', index_col='decision_at')
    history.index = pd.to_datetime(history.index, utc=True)
    bundle, record = fit_day(features, tables, history, at, protocol)
    if bundle is not None:
        online.mkdir(parents=True, exist_ok=False)
        blob = gzip.compress(pickle.dumps(bundle, protocol=5), mtime=0)
        path = online / 'model.pkl.gz'
        path.write_bytes(blob)
        record.update({'model_file': str(path.relative_to(run)), 'model_sha256': hashlib.sha256(blob).hexdigest(),
                       'source': 'generated_by_this_registered_run'})
        core.dump(online / 'registration.json', record)
    return bundle, record


def analyze(run, at, data_root=core.DATA_ROOT, current=False, position=None):
    run = Path(run)
    at = core.timestamp(at)
    result = {'analysis_at': at, 'analysis_at_et': at.tz_convert(core.ET), 'action': 'NO_ACTION',
              'broker_order_submitted': False, 'any_timestamp_supported': True,
              'analysis_mode': 'current_receipt_checked' if current else 'historical_assumed_bar_end_plus_1s',
              'estimate_kind': 'expected_net_trade_return_not_hit_probability'}
    market = core.load_market(data_root)
    if position is not None:
        result.update(manage(market, position, at))
        return result
    protocol = json.loads((run / 'registration.json').read_text())['protocol']
    manifest = json.loads((run / 'input_manifest.json').read_text())
    if any(manifest['market'][s]['sha256'] != market.manifest[s]['sha256'] for s in core.SYMBOLS):
        result['reason'] = 'market_updated_rebuild_daily_history_in_a_new_run'
        return result
    bundle, record = daily_bundle(run, at, protocol)
    result['daily_model'] = {k: v for k, v in record.items() if k not in {'train_members', 'calibration_members', 'feature_columns'}}
    row, audit = core.feature_row(market, at, require_received=current)
    result['data'] = audit
    results = json.loads((run / 'results.json').read_text())
    result['acceptance'] = results['acceptance']
    result['matches_registered_premarket_clock'] = at.tz_convert(core.ET).strftime('%H:%M:%S') == protocol['analysis_clock_et']
    if bundle is None:
        result['reason'] = 'insufficient_training'
    elif row is None:
        result['reason'] = 'data_unavailable_or_stale'
    else:
        result['research_forecast'] = forecast(bundle, row, at, protocol)
        result['feature_snapshot'] = row
        if not results['acceptance']['new_buy_eligible']:
            result['reason'] = 'daily_strategy_failed_return_or_nonnegative_month_gate'
        elif not bundle['ready']:
            result['reason'] = 'insufficient_prequential_calibration'
        elif not result['matches_registered_premarket_clock']:
            result['reason'] = 'other_clock_reference_only_effectiveness_unvalidated'
        else:
            chosen = result['research_forecast']['action']
            result['action'] = 'NO_ACTION' if chosen == 'CASH' else 'RESEARCH_BUY_' + chosen
            result['reason'] = 'qualified_research_reference_manual_execution_only'
            result['execution_constraints'] = {'one_position': True, 'allocation_of_settled_cash': protocol['allocation'],
                 'cooldown_after_loss_sessions': protocol['cooldown_sessions'], 'target_fraction': .05,
                 'protective_stop_fraction': protocol['risk']['stop_fraction'], 'max_calendar_days_inclusive': 14,
                 'account_holdings_cash_and_cooldown_must_be_checked_before_manual_execution': True}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--at', help='Aware historical timestamp; omission checks current data receipt')
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    parser.add_argument('--position-json', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.at or datetime.now(timezone.utc).isoformat(), args.data_root,
                     args.at is None, json.loads(args.position_json.read_text()) if args.position_json else None)
    if args.output:
        core.dump(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
