"""Independent membership, calibration, raw-path and capital reconciliation."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from breadth import core


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10), (a, b)


def read_trades(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def verify(run):
    run = Path(run)
    protocol = json.loads((run / 'registration.json').read_text())['preregistration']['protocol']
    features = pd.read_parquet(run / 'features.parquet')
    regular = features[features.regular_context_ready.eq(1.)]
    labels = pd.DataFrame(json.loads((run / 'labels.json').read_text()))
    for column in ['decision_at', 'label_available_at']:
        labels[column] = pd.to_datetime(labels[column], utc=True)
    tables = {s: labels[labels.symbol == s].set_index('decision_at') for s in core.ETFS}
    releases = {}
    for day, group in regular.groupby('date'):
        t = labels[labels.decision_at.isin(group.index)]
        if len(t) == 2 * len(group) and t.complete_window.all() and t.label_available_at.notna().all() and np.isfinite(t.net_return).all():
            releases[day] = t.label_available_at.max()
    training, checks = {}, 0
    for path in sorted((run / 'training').glob('*.json')):
        record = json.loads(path.read_text()); cutoff = pd.Timestamp(record['cutoff'])
        good = sorted(d for d, release in releases.items() if release < cutoff)[-252:]
        members = regular.index[regular.date.isin(good)]
        for variant, fit in record['variants'].items():
            assert list(pd.to_datetime(fit['members'], utc=True)) == list(members)
            assert fit['training_dates'] == len(good)
            assert fit['ready'] == (len(good) >= 120)
            if fit['ready']:
                close(fit['sample_weight_sum'], len(good))
            checks += 1
        training[pd.Timestamp(record['date'])] = members
    names = [v + '__' + a for v in protocol['feature_variants'] for a in protocol['algorithms']]
    histories = {n: pd.read_json(run / (n + '_predictions.json'), orient='table') for n in names}
    calibration_checks, raw_checks = 0, 0
    for name, history in histories.items():
        assert history.index.equals(features.index)
        assert history.loc[features.regular_context_ready.eq(0.), 'action'].eq('CASH').all()
        for at, pred in history.iterrows():
            if not np.isfinite(pred.SOXL_raw):
                assert pred.action == 'CASH'; continue
            date = at.tz_convert(core.ET).tz_localize(None).normalize()
            cutoff = date.tz_localize(core.ET).tz_convert('UTC')
            assert pd.Timestamp(pred.training_cutoff) == cutoff
            past = history.loc[training[date]]
            finite = np.isfinite(past[['SOXL_raw', 'SOXS_raw']].to_numpy()).all(axis=1)
            old = past.index[finite & past.clock.eq(pred.clock).to_numpy()][-60:]
            assert len(old) == pred.calibration_dates
            assert list(pd.to_datetime(pred.calibration_members, utc=True)) == list(old)
            if len(old) < 40:
                assert pred.action == 'CASH'; continue
            adjusted = []
            for symbol in core.ETFS:
                t = tables[symbol].loc[old]
                assert (t.label_available_at < cutoff).all()
                bias = float((np.log1p(t.net_return) - past.loc[old, symbol + '_raw']).mean())
                close(bias, pred[symbol + '_correction'])
                close(pred[symbol + '_raw'] + bias, pred[symbol + '_adjusted'])
                adjusted.append(pred[symbol + '_adjusted'])
            assert pred.action == (core.ETFS[int(np.argmax(adjusted))] if max(adjusted) > 0 else 'CASH')
            calibration_checks += 1
    for variant in protocol['feature_variants']:
        a, b, c = [histories[variant + '__' + n] for n in ['ridge_log','boosted_log','ensemble_log']]
        for symbol in core.ETFS:
            valid = a[symbol + '_raw'].notna()
            np.testing.assert_allclose(c.loc[valid, symbol + '_raw'],
                (a.loc[valid, symbol + '_raw'] + b.loc[valid, symbol + '_raw']) / 2, rtol=1e-10, atol=1e-14)
            raw_checks += int(valid.sum())
    frozen = json.loads((run / 'decisions_frozen.json').read_text())
    primary = protocol['primary']
    assert frozen['primary'] == primary
    for decision in frozen['actions']:
        assert histories[primary].loc[pd.Timestamp(decision['decision_at']), 'action'] == decision['action']
    market = core.load_market(); results = json.loads((run / 'results.json').read_text())
    trade_checks, book_checks = 0, []
    for name, multiple in [(n, 1) for n in names] + [('double_cost', 2)]:
        source = primary if name == 'double_cost' else name
        equity = pd.read_csv(run / (name + '_equity.csv'), parse_dates=['date']).set_index('date')
        trades = read_trades(run / (name + '_trades.csv'))
        if len(trades):
            trades['date'] = pd.to_datetime(trades.entry_date).dt.normalize()
            assert not trades.date.duplicated().any()
        expected_equity, previous = [], 1.
        fee, slip = .0005 * multiple, .0005 * multiple
        for date, mark in equity.iterrows():
            day_trades = trades[trades.date == date] if len(trades) else trades
            if len(day_trades):
                t = day_trades.iloc[0]; at, start = pd.Timestamp(t.decision_at), pd.Timestamp(t.entry_at)
                assert histories[source].loc[at, 'action'] == t.symbol
                assert start > at + pd.Timedelta(seconds=30)
                assert start - at <= pd.Timedelta(minutes=6)
                frame = market.bars[t.symbol]
                eligible = frame[(frame.date == date) & frame.session_type.eq('regular') & (frame.start_at > at + pd.Timedelta(seconds=30))]
                first = eligible.iloc[0]; assert first.start_at == start
                entry = first.open * (1 + slip) * first.share_factor
                target = math.ceil(first.open * (1 + slip) * 1.05 * 100 - 1e-10) / 100 * first.share_factor
                stop = entry * .985; event = None
                closing = pd.Timestamp(market.calendar[date]['close_at'])
                path = frame[(frame.start_at >= start) & (frame.end_at <= closing) & frame.session_type.eq('regular')]
                for bar in path.itertuples():
                    raw_stop = math.floor(stop / bar.share_factor * 100 + 1e-10) / 100
                    raw_target = math.ceil(target / bar.share_factor * 100 - 1e-10) / 100
                    if bar.low <= raw_stop:
                        event = (bar.end_at, min(bar.open, raw_stop) * bar.share_factor * (1 - slip), 'risk_stop'); break
                    if bar.high >= raw_target:
                        event = (bar.end_at, raw_target * bar.share_factor, 'target'); break
                if event is None:
                    last = path.iloc[-1]; assert last.end_at == closing
                    event = (closing, last.econ_close * (1 - slip), 'session_close')
                assert pd.Timestamp(t.exit_at) == event[0] and t.reason == event[2]
                close(entry, t.entry_fill_economic); close(event[1], t.exit_fill_economic)
                # Intraday positions have no overnight dividend entitlement.
                close(t.dividend_per_economic_unit, 0.)
                net = event[1] * (1 - fee) / (entry * (1 + fee)) - 1
                close(net, t.net_return); close(t.capital, previous)
                close(t.units_economic, previous / (entry * (1 + fee)))
                previous *= 1 + net
                close(mark.cash_settled, 0.); close(mark.cash_unsettled, previous)
                trade_checks += 1
            else:
                close(mark.cash_settled, previous); close(mark.cash_unsettled, 0.)
            close(mark.equity, previous); close(mark.position_value, 0.); close(mark.receivables, 0.)
            assert mark.mark_complete and mark.symbol == 'CASH'
            expected_equity.append(previous)
        stats = results['double_cost'] if name == 'double_cost' else results['routes'][name]
        prior, monthly = 1., []
        for month, group in equity.groupby(equity.index.to_period('M')):
            rate = group.equity.iloc[-1] / prior - 1
            match = next(m for m in stats['monthly'] if m['month'] == str(month))
            close(rate, match['return']); monthly.append((str(month), rate)); prior = group.equity.iloc[-1]
        average = np.mean([r for m, r in monthly if m in protocol['regression_months']])
        close(average, stats['primary_average']); close(prior - 1, stats['continuous_return'])
        assert stats['primary_pass'] == (average >= .1 and all(r >= -1e-12 for _, r in monthly))
        book_checks.append({'name': name, 'trades': len(trades), 'monthly_and_settlement_reconciled': True})
    out = {'passed': True, 'membership_checks': checks, 'calibrated_predictions': calibration_checks,
        'equal_weight_raw_predictions': raw_checks, 'raw_trades': trade_checks, 'capital_books': book_checks,
        'frozen_decisions': len(frozen['actions']), 'no_effect_pass_inferred': True}
    core.dump(run / 'verification.json', out); print(json.dumps(out, indent=2))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    verify(parser.parse_args().run)
