"""Independent historical routing, price-bar exits and capital-account verification."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experts import core, EXPERTS, RISKS


def verify(run):
    run = Path(run)
    registration = json.loads((run / 'registration.json').read_text())
    protocol = registration['preregistration']['protocol']
    features = pd.read_parquet(run / 'features.parquet')
    labels = {}
    for risk in RISKS:
        t = pd.DataFrame(json.loads((run / (risk + '_labels.json')).read_text()))
        t.decision_at = pd.to_datetime(t.decision_at, utc=True)
        t.label_available_at = pd.to_datetime(t.label_available_at, utc=True)
        labels[risk] = {s: t[t.symbol == s].set_index('decision_at') for s in core.ETFS}
    histories = {}
    for expert in EXPERTS:
        h = pd.read_csv(run / (expert + '_predictions.csv'), index_col='decision_at')
        h.index = pd.to_datetime(h.index, utc=True)
        histories[expert] = h
    records = {}
    for path in sorted((run / 'daily').glob('*.json')):
        record = json.loads(path.read_text())
        at = pd.Timestamp(record['decision_at']); cutoff = pd.Timestamp(record['training_cutoff'])
        records[at] = record
        assert cutoff == at.tz_convert(core.ET).normalize().tz_convert('UTC')
        independently_qualified = []
        eligible_by_risk = {}
        for risk in RISKS:
            eligible = features.index[features.index < cutoff]
            for symbol in core.ETFS:
                table = labels[risk][symbol].reindex(eligible)
                eligible = eligible[(table.label_available_at < cutoff) & table.complete_window.eq(True) & np.isfinite(table.net_return)]
            eligible_by_risk[risk] = eligible
            recorded = record['training'][risk]
            members = pd.DatetimeIndex(pd.to_datetime(recorded['members'], utc=True))
            expected = eligible[-protocol['training_window_dates']:]
            assert list(members) == list(expected)
            if len(members):
                assert pd.Timestamp(recorded['last_label_available']) < cutoff
        for candidate in record['selector']:
            expert = candidate['expert']; risk = expert.split('__')[1]
            dates = eligible_by_risk[risk].intersection(histories[expert].index)[-protocol['selector_window_dates']:]
            values = []; logs = []
            for when in dates:
                action = histories[expert].loc[when, 'action']
                if action == 'CASH':
                    values.append(0.); continue
                outcome = labels[risk][action].loc[when]
                assert outcome.label_available_at < cutoff
                assert when < outcome.label_available_at
                net_log = math.log1p(outcome.net_return)
                values.append(net_log / max(int(outcome.held_sessions), 1)); logs.append(net_log)
            mean = float(np.mean(values)) if values else 0.
            penalty = .5 * float(np.std(values, ddof=1)) / math.sqrt(max(len(values) / 5, 1)) if len(values) > 1 else 0.
            ranking_score = mean - penalty
            qualifies = len(dates) == 63 and len(logs) >= 12 and bool(logs) and np.mean(logs) > 0 and ranking_score > 0
            assert bool(qualifies) == candidate['qualified']
            assert math.isclose(ranking_score, candidate['score'], abs_tol=1e-13)
            if qualifies:
                independently_qualified.append((ranking_score, expert))
        selected = sorted(independently_qualified, key=lambda x: (-x[0], x[1]))[0][1] if independently_qualified else None
        assert selected == record['selected_expert']
        assert record['action'] == (histories[selected].loc[at, 'action'] if selected else 'CASH')
        for expert, p in record['predictions'].items():
            if expert.startswith('trend20'):
                expected = 'SOXL' if features.loc[at, 'SOXX_ret20'] > 0 else 'SOXS'
                assert p['action'] == expected
                continue
            if not np.isfinite(p['SOXL_raw']):
                assert p['action'] == 'CASH'; continue
            risk = expert.split('__')[1]
            eligible = eligible_by_risk[risk]
            h = histories[expert].loc[eligible]
            valid = np.isfinite(h[['SOXL_raw', 'SOXS_raw']].to_numpy(dtype=float)).all(axis=1)
            calibration = h.index[valid][-60:]
            assert len(calibration) == p['calibration_dates']
            adjusted = []
            for symbol in core.ETFS:
                bias = float((np.log1p(labels[risk][symbol].loc[calibration, 'net_return']) - h.loc[calibration, symbol + '_raw']).mean()) if len(calibration) else None
                i = core.ETFS.index(symbol)
                if bias is not None:
                    assert math.isclose(bias, p['correction'][i], abs_tol=1e-13)
                    adjusted.append(p[symbol + '_raw'] + bias)
            if len(calibration) < 40:
                assert p['action'] == 'CASH'
            else:
                best = int(np.argmax(adjusted))
                assert p['action'] == (core.ETFS[best] if adjusted[best] > 0 else 'CASH')
    market = core.load_market()
    manifest = json.loads((run / 'input_manifest.json').read_text())
    assert all(market.manifest[s]['sha256'] == manifest['market'][s]['sha256'] for s in core.SYMBOLS)
    results = json.loads((run / 'results.json').read_text())
    trade_checks = 0; gaps = 0; ambiguous = 0; protected = 0
    for name, multiple in [('router', 1), ('double_cost', 2)]:
        try:
            trades = pd.read_csv(run / (name + '_trades.csv'))
        except pd.errors.EmptyDataError:
            trades = pd.DataFrame()
        equity = pd.read_csv(run / (name + '_equity.csv'), parse_dates=['date']).set_index('date')
        fee = .0005 * multiple; slip = .0005 * multiple
        closed_profit = 0.; open_profit = 0.
        previous_exit = None
        for t in trades.itertuples():
            at = pd.Timestamp(t.decision_at); entry_at = pd.Timestamp(t.entry_at); expiry = pd.Timestamp(t.expiry_at)
            assert entry_at > at + pd.Timedelta(seconds=30)
            assert previous_exit is None or entry_at > previous_exit
            record = records[at]
            assert t.risk_rule == record['risk'] and t.symbol == record['action']
            bars = market.bars[t.symbol]
            initial = bars[bars.start_at == entry_at].iloc[0]
            entry_slip = (.001 if initial.session_type == 'pre_market' else .0005) * multiple
            entry = initial.open * (1 + entry_slip) * initial.share_factor
            assert math.isclose(entry, t.entry_fill_economic, abs_tol=1e-9)
            assert math.isclose(t.units_economic * entry * (1 + fee), t.capital, abs_tol=1e-9)
            target = math.ceil(initial.open * (1 + entry_slip) * 1.05 * 100 - 1e-10) / 100 * initial.share_factor
            width = max(.06, min(.18, 2 * features.loc[at, t.symbol + '_atr14']))
            stop = entry * (1 - width) if record['risk'] != 'expiry14' else -math.inf
            pending = None; raised = False; event = None
            path = bars[(bars.start_at >= entry_at) & (bars.end_at <= expiry) & bars.session_type.eq('regular')]
            for bar in path.itertuples():
                if pending is not None and bar.start_at > pending[1]:
                    stop = max(stop, pending[0]); raised = True; pending = None
                raw_stop = math.floor(stop / bar.share_factor * 100 + 1e-10) / 100 if math.isfinite(stop) else -math.inf
                raw_target = math.ceil(target / bar.share_factor * 100 - 1e-10) / 100
                if bar.low <= raw_stop:
                    event = (bar.end_at, min(bar.open, raw_stop) * (1 - slip) * bar.share_factor, 'profit_protection' if raised else 'risk_stop')
                    gaps += int(bar.open < raw_stop); ambiguous += int(bar.high >= raw_target); protected += int(raised)
                    break
                if bar.high >= raw_target:
                    event = (bar.end_at, raw_target * bar.share_factor, 'target'); break
                if record['risk'] == 'atr_protect' and not raised and pending is None and bar.high * bar.share_factor >= entry * 1.02:
                    flat = entry * (1 + fee) / ((1 - fee) * (1 - slip))
                    pending = (math.ceil(flat / bar.share_factor * 100 - 1e-10) / 100 * bar.share_factor, bar.available_at + pd.Timedelta(seconds=30))
            if event is None and len(path) and path.end_at.iloc[-1] == expiry:
                event = (expiry, path.econ_close.iloc[-1] * (1 - slip), 'expiry')
            if event is not None:
                assert event[0] == pd.Timestamp(t.exit_at)
                assert event[2] == t.reason
                assert math.isclose(event[1], t.exit_fill_economic, abs_tol=1e-9)
                previous_exit = event[0]
            else:
                assert pd.isna(t.exit_at)
            if pd.notna(t.exit_at) and pd.Timestamp(t.exit_at) <= pd.Timestamp('2026-09-30 16:00', tz=core.ET):
                net = (t.exit_fill_economic * (1 - fee) + t.dividend_per_economic_unit) / (entry * (1 + fee)) - 1
                assert math.isclose(net, t.net_return, abs_tol=1e-10)
                closed_profit += t.capital * net
            else:
                mark = market.daily[t.symbol].loc[equity.index[-1], 'close']
                open_profit += t.units_economic * mark + t.accrued_dividends - t.capital
            trade_checks += 1
        assert math.isclose(equity.equity.iloc[-1], 1 + closed_profit + open_profit, abs_tol=1e-9)
        month_ends = equity.equity.groupby(equity.index.to_period('M')).last()
        monthly = month_ends / np.r_[1., month_ends.iloc[:-1]] - 1
        for row in results[name]['monthly']:
            assert math.isclose(monthly[pd.Period(row['month'])], row['return'], abs_tol=1e-10)
        mean = float(np.mean([monthly[pd.Period(m)] for m in protocol['regression_months']]))
        assert math.isclose(mean, results[name]['primary_average'], abs_tol=1e-10)
        assert results[name]['primary_pass'] == bool(mean >= .1 and (monthly >= -1e-12).all())
    evidence = {'pass': True, 'causal_daily_router_records': len(records), 'expert_routes': len(EXPERTS),
        'training_and_calibration_membership_recomputed': True, 'no_case_winner_used_for_admission': True,
        'raw_bar_trade_checks': trade_checks, 'gap_exits': gaps, 'ambiguous_stop_first_exits': ambiguous,
        'delayed_profit_protection_exits': protected, 'terminal_equity_and_month_returns_recomputed': True,
        'historical_replay_not_actual_broker_fills': True, 'acceptance_passed': results['acceptance']['passed']}
    core.dump(run / 'verification.json', evidence)
    print(json.dumps(evidence, indent=2))
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    verify(parser.parse_args().run)
