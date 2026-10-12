"""Independent sample-boundary, shadow-equity and raw-bar execution audit."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

from path_engine import core, EXITS, EXPERTS


def verify(run):
    run = Path(run)
    protocol = json.loads((run / 'registration.json').read_text())['preregistration']['protocol']
    features = pd.read_parquet(run / 'features.parquet')
    regular = features[features.regular_context_ready.eq(1.)]
    tables = {}; date_release = {}
    for policy in EXITS:
        all_labels = pd.DataFrame(json.loads((run / (policy + '_labels.json')).read_text()))
        all_labels.decision_at = pd.to_datetime(all_labels.decision_at, utc=True)
        all_labels.label_available_at = pd.to_datetime(all_labels.label_available_at, utc=True)
        tables[policy] = {s: all_labels[all_labels.symbol == s].set_index('decision_at') for s in core.ETFS}
        releases = {}
        for date, day in regular.groupby('date'):
            part = all_labels[all_labels.decision_at.isin(day.index)]
            if len(part) == 2 * len(day) and part.complete_window.all() and part.label_available_at.notna().all() and np.isfinite(part.net_return).all():
                releases[date] = part.label_available_at.max()
        date_release[policy] = releases
    training = {}; membership_checks = 0
    for path in sorted((run / 'training').glob('*.json')):
        record = json.loads(path.read_text()); cutoff = pd.Timestamp(record['training_cutoff'])
        training[pd.Timestamp(record['date'])] = record
        for policy in EXITS:
            eligible = sorted(d for d, release in date_release[policy].items() if release < cutoff)[-252:]
            expected = regular.index[regular.date.isin(eligible)]
            actual = list(pd.to_datetime(record['exit_policies'][policy]['members'], utc=True))
            assert actual == list(expected)
            assert record['exit_policies'][policy]['training_dates'] == len(eligible)
            if record['exit_policies'][policy]['ready']:
                assert math.isclose(record['exit_policies'][policy]['sample_weight_sum'], len(eligible), abs_tol=1e-9)
            membership_checks += 1
    histories = {e: pd.read_json(run / (e + '_predictions.json'), orient='table') for e in EXPERTS}
    calibrated_checks = 0
    for expert, history in histories.items():
        assert history.loc[features[features.regular_context_ready.eq(0.)].index, 'action'].eq('CASH').all()
        if not expert.startswith(('ridge_log', 'boosted_log')):
            continue
        policy = expert.split('__')[1]
        for at, pred in history.iterrows():
            if not np.isfinite(pred.SOXL_raw):
                continue
            date = at.tz_convert(core.ET).tz_localize(None).normalize()
            train = pd.DatetimeIndex(pd.to_datetime(training[date]['exit_policies'][policy]['members'], utc=True))
            past = history.loc[train]
            finite = np.isfinite(past[['SOXL_raw', 'SOXS_raw']].to_numpy(dtype=float)).all(axis=1)
            old = past.index[finite & past.clock.eq(pred.clock).to_numpy()][-60:]
            assert len(old) == pred.calibration_dates
            if len(old) < 40:
                assert pred.action == 'CASH'; continue
            adjusted = []
            for symbol in core.ETFS:
                t = tables[policy][symbol].loc[old]
                assert (t.label_available_at < pd.Timestamp(training[date]['training_cutoff'])).all()
                assert (old < t.label_available_at).all()
                bias = float((np.log1p(t.net_return) - past.loc[old, symbol + '_raw']).mean())
                # orient=table stores ten decimal places in this run.
                assert math.isclose(bias, pred[symbol + '_correction'], abs_tol=1e-9)
                adjusted.append(pred[symbol + '_raw'] + bias)
            winner = int(np.argmax(adjusted))
            assert abs(max(adjusted)) > 1e-9, 'Rounded score too close to decision threshold'
            assert pred.action == (core.ETFS[winner] if adjusted[winner] > 0 else 'CASH')
            calibrated_checks += 1
    shadow = {}
    for expert in EXPERTS:
        eq = pd.read_csv(run / ('shadow_' + expert + '_equity.csv'), parse_dates=['date']).set_index('date')
        try:
            tr = pd.read_csv(run / ('shadow_' + expert + '_trades.csv'), parse_dates=['entry_date'])
        except pd.errors.EmptyDataError:
            tr = pd.DataFrame()
        shadow[expert] = (eq, tr)
    selector = json.loads((run / 'selector_registry.json').read_text())
    for day in selector:
        date = pd.Timestamp(day['date']); cutoff = date.tz_localize(core.ET)
        good = []
        for row in day['selector']:
            eq, tr = shadow[row['expert']]
            past = eq[eq.index < date].tail(64)
            assert len(past) == row['dates']
            if len(past) != 64 or not past.mark_complete.all():
                assert not row['qualified']; continue
            curve = past.equity.to_numpy(); changes = np.diff(np.log(curve))
            score = float(changes.mean() - .25 * changes.std(ddof=1) / 63 ** .5)
            drawdown = float((curve / np.maximum.accumulate(curve) - 1).min())
            closed = int(((tr.entry_date >= past.index[1]) & (tr.entry_date < date) & tr.exit_at.notna() &
                          (pd.to_datetime(tr.exit_at, utc=True) < cutoff)).sum()) if len(tr) else 0
            qualifies = score > 0 and curve[-1] > curve[0] and curve[-1] > curve[-22] and drawdown >= -.1 and closed >= 12
            assert bool(qualifies) == row['qualified']
            assert math.isclose(score, row['score'], abs_tol=1e-12)
            if qualifies:
                good.append((score, row['expert']))
        selected = sorted(good, key=lambda x: (-x[0], x[1]))[0][1] if good else None
        assert selected == day['selected_expert']
    market = core.load_market(); sectors = {}
    for symbol in ['SOXX', 'QQQ']:
        frames = []
        for _, f in market.bars[symbol][market.bars[symbol].session_type.eq('regular')].groupby('date'):
            typical = (f.econ_high + f.econ_low + f.econ_close) / 3
            vw = (typical * f.econ_volume).cumsum() / f.econ_volume.cumsum()
            frames.append(pd.DataFrame({'direction': f.econ_close / f.econ_open.iloc[0] - 1,
                        'vwap_direction': f.econ_close / vw - 1, 'available_at': f.available_at}, index=f.index).iloc[5:])
        sectors[symbol] = pd.concat(frames)
    selected_by_date = {pd.Timestamp(x['date']): x for x in selector}
    results = json.loads((run / 'results.json').read_text())
    trade_count = 0; stop_count = 0; trend_count = 0; close_count = 0
    verified_books = [('primary', 1), ('double_cost', 2), ('ridge_log__adaptive_overnight', 1)]
    for name, multiple in verified_books:
        equity = pd.read_csv(run / (name + '_equity.csv'), parse_dates=['date']).set_index('date')
        try:
            trades = pd.read_csv(run / (name + '_trades.csv'), parse_dates=['entry_date'])
        except pd.errors.EmptyDataError:
            trades = pd.DataFrame()
        closed_profit = 0.; open_profit = 0.; previous_exit = None
        fee = .0005 * multiple; slip = .0005 * multiple
        for t in trades.itertuples():
            at = pd.Timestamp(t.decision_at); start = pd.Timestamp(t.entry_at); policy = t.exit_policy
            chosen = selected_by_date[t.entry_date]['selected_expert'] if name in ['primary', 'double_cost'] else name
            assert chosen.endswith('__' + policy) and histories[chosen].loc[at, 'action'] == t.symbol
            assert start > at + pd.Timedelta(seconds=30)
            assert previous_exit is None or start > previous_exit
            f = market.bars[t.symbol]; first = f[f.start_at == start].iloc[0]
            entry = first.open * (1 + slip) * first.share_factor
            assert math.isclose(entry, t.entry_fill_economic, abs_tol=1e-9)
            assert math.isclose(t.units_economic * entry * (1 + fee), t.capital, abs_tol=1e-9)
            limit = math.ceil(first.open * (1 + slip) * 1.05 * 100 - 1e-10) / 100 * first.share_factor
            end = pd.Timestamp(t.expiry_at) if policy == 'adaptive_overnight' else pd.Timestamp(market.calendar[t.entry_date]['close_at'])
            path = f[(f.start_at >= start) & (f.end_at <= end) & f.session_type.eq('regular')]
            stop = entry * (1 - (.03 if policy == 'adaptive_overnight' else .015))
            updates = []; sell_after = None; peak = entry; raised = False; event = None
            for bar in path.itertuples():
                target = math.ceil(limit / bar.share_factor * 100 - 1e-10) / 100
                if sell_after is not None and bar.start_at > sell_after:
                    event = (bar.start_at, min(bar.open, target) * bar.share_factor * (1 - slip), 'trend_exit'); break
                ready = [u for u in updates if bar.start_at > u[1]]
                if ready:
                    stop = max(stop, max(u[0] for u in ready)); raised = True
                    updates = [u for u in updates if bar.start_at <= u[1]]
                raw_stop = math.floor(stop / bar.share_factor * 100 + 1e-10) / 100
                if bar.low <= raw_stop:
                    event = (bar.end_at, min(bar.open, raw_stop) * bar.share_factor * (1 - slip), 'trailing_stop' if raised else 'risk_stop'); break
                if bar.high >= target:
                    event = (bar.end_at, target * bar.share_factor, 'target'); break
                peak = max(peak, bar.econ_high)
                if policy != 'flat_close' and peak >= entry * 1.02:
                    flat = math.ceil(entry * (1 + fee) / ((1 - fee) * (1 - slip)) / bar.share_factor * 100 - 1e-10) / 100 * bar.share_factor
                    level = max(flat, peak * .985)
                    if level > max([stop] + [u[0] for u in updates]):
                        updates.append((level, bar.available_at + pd.Timedelta(seconds=30)))
                if policy == 'adaptive_overnight' and bar.end_at in sectors['SOXX'].index and bar.end_at in sectors['QQQ'].index:
                    x = sectors['SOXX'].loc[bar.end_at]; q = sectors['QQQ'].loc[bar.end_at]
                    sign = 1 if t.symbol == 'SOXL' else -1
                    if sell_after is None and sign * x.direction < 0 and sign * x.vwap_direction < 0 and sign * q.direction < 0:
                        sell_after = max(bar.available_at, x.available_at, q.available_at) + pd.Timedelta(seconds=30)
            if event is None and len(path) and path.end_at.iloc[-1] == end:
                event = (end, path.econ_close.iloc[-1] * (1 - slip), 'expiry' if policy == 'adaptive_overnight' else 'session_close')
            if event is None:
                assert pd.isna(t.exit_at)
            else:
                assert event[0] == pd.Timestamp(t.exit_at)
                assert event[2] == t.reason
                assert math.isclose(event[1], t.exit_fill_economic, abs_tol=1e-9)
                previous_exit = event[0]
                stop_count += int(event[2] == 'trailing_stop'); trend_count += int(event[2] == 'trend_exit'); close_count += int(event[2] == 'session_close')
            if pd.notna(t.exit_at) and pd.Timestamp(t.exit_at) <= pd.Timestamp('2026-09-30 16:00', tz=core.ET):
                net = (t.exit_fill_economic * (1 - fee) + t.dividend_per_economic_unit) / (entry * (1 + fee)) - 1
                assert math.isclose(net, t.net_return, abs_tol=1e-10)
                closed_profit += t.capital * net
            else:
                open_profit += t.units_economic * market.daily[t.symbol].loc[equity.index[-1], 'close'] + t.accrued_dividends - t.capital
            trade_count += 1
        assert math.isclose(equity.equity.iloc[-1], 1 + closed_profit + open_profit, abs_tol=1e-9)
        ends = equity.equity.groupby(equity.index.to_period('M')).last()
        monthly = ends / np.r_[1., ends.iloc[:-1]] - 1
        reported = results[name] if name in ['primary', 'double_cost'] else results['diagnostic_experts'][name]
        for row in reported['monthly']:
            assert math.isclose(row['return'], monthly[pd.Period(row['month'])], abs_tol=1e-10)
        average = np.mean([monthly[pd.Period(m)] for m in protocol['regression_months']])
        assert math.isclose(average, reported['primary_average'], abs_tol=1e-10)
        assert bool(average >= .1 and (monthly >= -1e-12).all()) == reported['primary_pass']
    evidence = {'pass': True, 'training_membership_checks': membership_checks,
        'calibrated_forecasts_recomputed': calibrated_checks, 'selector_days_recomputed': len(selector),
        'raw_trade_checks': trade_count, 'verified_books': [x[0] for x in verified_books],
        'trailing_exits': stop_count, 'trend_exits': trend_count,
        'planned_session_close_exits': close_count, 'premarket_rows_always_wait': True,
        'marked_capital_and_monthly_returns_recomputed': True,
        'historical_replay_not_broker_fills': True, 'acceptance_passed': results['acceptance']['passed']}
    core.dump(run / 'verification.json', evidence); print(json.dumps(evidence, indent=2))
    return evidence


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    verify(p.parse_args().run)
