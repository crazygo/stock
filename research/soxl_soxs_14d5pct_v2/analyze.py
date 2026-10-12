"""Run the same causal model at any explicit or current timestamp.

Output is research and a paper-order template; no brokerage mutation occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import pandas as pd

import core
import model


def analyze(data_root, run, decision_at, strategy='auto', position=None, current=False):
    at = core.timestamp(decision_at)
    local = at.tz_convert(core.ET)
    date = local.tz_localize(None).normalize()
    month = str(date.to_period('M'))
    market = core.load_market(data_root)
    row, audit = core.feature_row(market, at, require_received=current)
    report = {'analysis_at': at, 'analysis_at_et': local, 'action': 'NO_ACTION',
              'analysis_clock_is_arbitrary': True, 'data': audit,
              'broker_order_submitted': False,
              'analysis_mode': 'current_receipt_checked' if current else 'historical_with_assumed_availability',
              'historical_availability_assumption': 'bar_end+1s applies only to historical replay; current mode checks observed receipt',
              'probability_status': 'research_estimate_not_independently_calibrated'}
    if position is not None:
        from position import manage
        report.update(manage(market, position, at))
        return report
    if row is None:
        report['reason'] = 'data_unavailable_or_stale'
        return report
    run = Path(run)
    results = json.loads((run / 'results.json').read_text())
    if strategy == 'auto':
        if results.get('adaptive_model', {}).get('primary_pass', False):
            strategy = 'adaptive_model'
        elif results['trend20_premarket']['primary_pass']:
            strategy = 'trend20_premarket'
        else:
            report.update({'reason': 'no_registered_strategy_passed_user_gate',
                           'historical_results': results['acceptance']})
            return report
    if strategy not in ['adaptive_model', 'trend20_premarket']:
        raise ValueError('Only registered model or trend20 strategies may be analyzed')
    if strategy not in results or not results[strategy]['primary_pass']:
        report.update({'reason': 'chosen_strategy_failed_user_gate', 'strategy': strategy})
        return report
    # User's 2026-10-12 risk requirement supersedes the old arithmetic-only gate.
    negative_months = [m for m in results[strategy]['monthly'] if m['return'] < -1e-12]
    if negative_months:
        report.update({'reason': 'negative_months_fail_current_requirement',
                       'strategy': strategy, 'failed_months': negative_months,
                       'legacy_arithmetic_pass_does_not_grant_current_admission': True})
        return report
    report['strategy'] = strategy
    report['historical_three_month_mean'] = results[strategy]['primary_average']
    registered_clock = results[strategy].get('scheduled_analysis_clock_et')
    report['registered_replay_clock_et'] = registered_clock
    report['matches_registered_replay_clock'] = registered_clock is None or local.strftime('%H:%M:%S') == registered_clock
    report['other_timestamp_effectiveness_established'] = False
    report['return_evidence_scope'] = 'The reported monthly mean belongs to the registered replay schedule; arbitrary-time analysis is a research reference and does not inherit that return guarantee'
    report['feature_snapshot'] = row
    if strategy == 'trend20_premarket':
        chosen = 'SOXL' if row['SOXX_ret20'] > 0 else 'SOXS'
        report['algorithm'] = 'fixed SOXX 20-completed-session return sign'
        report['direction_feature'] = row['SOXX_ret20']
    else:
        registry = json.loads((run / 'monthly_registration.json').read_text())
        records = [r for r in registry if r['month'] == month]
        if not records:
            report['reason'] = 'month_has_no_registered_model'; return report
        record = records[0]
        if pd.Timestamp(record['selection_last_contractual_deadline']) >= date:
            raise ValueError('Model selection uses a label not matured before the analysis month')
        manifest = json.loads((run / 'model_manifest.json').read_text())
        binary = run / 'models.pkl'
        if hashlib.sha256(binary.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Locally generated model hash differs from registered manifest')
        if manifest['source'] != 'generated_by_this_registered_run':
            raise ValueError('Unregistered serialized models are not accepted')
        with binary.open('rb') as file:
            models = pickle.load(file)[month]
        X = pd.DataFrame([row], index=pd.DatetimeIndex([at]))[record['feature_columns']]
        prediction = model.predict(models, X)
        chosen_policy = record['chosen']
        # Replay clocks define repeat frequency. An explicit manual query may use
        # any timestamp with the same causal features and frozen monthly threshold.
        table = pd.DataFrame([{'date': date, 'clock': local.strftime('%H:%M:%S'), **row}], index=X.index)
        chosen = model.actions(table, prediction, chosen_policy['route'], chosen_policy.get('threshold'), 'all_registered_clocks')[at]
        report['model_policy'] = chosen_policy
        report['predictions'] = prediction.iloc[0].to_dict()
        report['algorithm'] = 'LogisticRegression + ExtraTrees or direct ExtraTrees, monthly selected'
    session = market.calendar.get(date)
    minute = local.hour * 60 + local.minute
    if chosen == 'CASH':
        report['reason'] = 'confidence_or_expected_payoff_insufficient'; return report
    if session is None or minute < 240 or minute >= 930 or at >= pd.Timestamp(session['close_at']):
        report.update({'reason': 'outside_registered_entry_window', 'reference_symbol': chosen})
        return report
    entry_after = at + pd.Timedelta(seconds=30)
    next_start = entry_after.ceil('5min')
    if next_start <= entry_after:
        next_start += pd.Timedelta(minutes=5)
    if next_start >= pd.Timestamp(session['close_at']) or next_start.tz_convert(core.ET).hour * 60 + next_start.tz_convert(core.ET).minute > 930:
        report['reason'] = 'no_entry_window_remaining'; return report
    expiry = core.expiry_day(market, date)
    if expiry is None:
        report['reason'] = 'future_calendar_unavailable'; return report
    report.update({'action': 'BUY_' + chosen, 'reason': 'registered_model_or_baseline_condition_met',
                   'entry_not_before': next_start,
                   'order_template': {'entry_side': 'BUY', 'symbol': chosen,
                                      'take_profit_side': 'SELL', 'take_profit_type': 'LIMIT',
                                      'take_profit_tif': 'GTC', 'take_profit_active_session': 'regular',
                                      'take_profit_price_formula': 'ceil(actual_entry_fill * 1.05 * 100) / 100',
                                      'take_profit_created': 'at actual entry fill',
                                      'force_sell_at': market.calendar[expiry]['close_at'],
                                      'cancel_take_profit_before_forced_sell': True,
                                      'force_sell_only_if_position_still_open': True,
                                      'calendar_days_inclusive': 14, 'broker_submission': False}})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--at', help='Explicit timezone-aware historical cutoff; omitted analyzes current time after refresh')
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    parser.add_argument('--run', type=Path, default=core.ROOT / '.cache/soxl_soxs_14d5pct_v2/current')
    parser.add_argument('--strategy', choices=['auto', 'adaptive_model', 'trend20_premarket'], default='auto')
    parser.add_argument('--position-json', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--offline', action='store_true', help='Skip current-data refresh')
    parser.add_argument('--data-timeout', type=float, default=30)
    args = parser.parse_args()
    position = json.loads(args.position_json.read_text()) if args.position_json else None
    refresh_status = None
    if args.at is None and not args.offline and position is None:
        from refresh import refresh
        args.data_root, refresh_status = refresh(args.data_root, args.data_timeout)
    at = args.at or datetime.now(timezone.utc).isoformat()
    result = analyze(args.data_root, args.run, at, args.strategy, position, current=args.at is None)
    if refresh_status:
        result['refresh'] = refresh_status
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); core.dump(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
