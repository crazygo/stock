"""Verify exported trades, order intents and equity independently of replay()."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import core


def verify(run, data_root):
    run = Path(run); market = core.load_market(data_root)
    inputs = json.loads((run / 'input_manifest.json').read_text())
    for symbol in core.SYMBOLS:
        if inputs[symbol]['sha256'] != market.manifest[symbol]['sha256']:
            raise AssertionError(f'{symbol}: frozen input hash changed')
    coverage = json.loads((run / 'candidate_coverage.json').read_text())
    for item in coverage:
        if item['status'] == 'ready':
            decision = pd.Timestamp(item['decision_at'])
            for cutoff in item['data_cutoffs'].values():
                assert pd.Timestamp(cutoff) + pd.Timedelta(seconds=1) <= decision
    results = json.loads((run / 'results.json').read_text())
    reports = []; checked_orders = 0; checked_losses = 0
    for name, score in results.items():
        if not isinstance(score, dict) or 'primary_average' not in score:
            continue
        equity = pd.read_csv(run / f'{name}_equity.csv', parse_dates=['date']).set_index('date')
        path = run / f'{name}_trades.csv'
        trades = pd.read_csv(path) if path.stat().st_size > 1 else pd.DataFrame()
        order_path = run / f'{name}_orders.csv'
        orders = pd.read_csv(order_path) if order_path.stat().st_size > 1 else pd.DataFrame()
        assert len(orders) == len(trades)
        assert not len(trades) or not trades.entry_date.duplicated().any()
        checked_orders += len(orders)
        prior = 1.0; monthly = {}
        for period, group in equity.groupby(equity.index.to_period('M')):
            last = float(group.equity.iloc[-1]); monthly[str(period)] = last / prior - 1; prior = last
            assert group.mark_complete.all()
        mean = float(np.mean([monthly[m] for m in ['2026-05', '2026-07', '2026-09']]))
        assert np.isclose(mean, score['primary_average'], rtol=1e-11, atol=1e-12)
        assert score['primary_pass'] == (mean >= .1)
        total_pnl = 0.0; events = []; completed = 0; pending = 0
        end = equity.index[-1]
        for number, trade in trades.iterrows():
            symbol = trade.symbol; data = market.bars[symbol]
            entry_at = pd.Timestamp(trade.entry_at); decision = pd.Timestamp(trade.decision_at)
            assert entry_at > decision + pd.Timedelta(seconds=30)
            entry_day = entry_at.tz_convert(core.ET).tz_localize(None).normalize()
            target_date = entry_day + pd.Timedelta(days=13)
            expected_day = market.sessions[market.sessions <= target_date][-1]
            assert expected_day == pd.Timestamp(trade.expiry_date)
            assert pd.Timestamp(trade.expiry_at) == pd.Timestamp(market.calendar[expected_day]['close_at'])
            first = data[data.start_at == entry_at].iloc[0]
            cost_multiple = 2.0 if name == 'double_cost' else 1.0
            slip = (core.PRE_SLIP if first.session_type == 'pre_market' else core.SLIP) * cost_multiple
            fill = float(first.open) * (1 + slip)
            assert np.isclose(fill, trade.entry_fill_raw, rtol=1e-11)
            limit = math.ceil(fill * 1.05 * 100 - 1e-10) / 100
            assert np.isclose(limit, trade.limit_raw)
            order = orders.iloc[number]
            assert order.side == 'SELL' and order.type == 'LIMIT' and order.tif == 'GTC'
            assert np.isclose(order.limit_price, limit)
            assert pd.Timestamp(order.created_at) == entry_at
            assert not bool(order.broker_submission)
            # Reconstruct split-adjusted order changes directly from events.
            shares_ratio = 1.0; target_raw = limit; first_hit = None
            regular = data.loc[entry_at:pd.Timestamp(trade.expiry_at)]
            regular = regular[(regular.start_at >= entry_at) & (regular.session_type == 'regular')]
            for day, group in regular.groupby('date'):
                cumulative = np.prod([float(a['value']) for a in market.actions[symbol]
                                      if a['type'] == 'Splits' and entry_day < pd.Timestamp(a['date']) <= day])
                target_raw = math.ceil(limit / cumulative * 100 - 1e-10) / 100
                hit = group[group.high >= target_raw]
                if len(hit):
                    first_hit = hit.iloc[0]; break
            observed_exit = pd.Timestamp(trade.exit_at) if pd.notna(trade.exit_at) else None
            if trade.reason == 'target':
                assert first_hit is not None
                assert observed_exit == first_hit.end_at
                assert np.isclose(trade.exit_fill_economic, target_raw * first_hit.share_factor)
            elif trade.reason == 'expiry':
                assert first_hit is None
                assert observed_exit == pd.Timestamp(trade.expiry_at)
                last = regular.iloc[-1]
                assert last.end_at == observed_exit
                assert np.isclose(trade.exit_fill_economic, last.econ_close * (1 - core.SLIP * cost_multiple))
            else:
                assert trade.reason in ['pending', 'missing_prefix']
                assert observed_exit is None and pd.isna(trade.net_return)
            if observed_exit is not None:
                expected_net = ((trade.exit_fill_economic * (1 - core.FEE * cost_multiple)
                                 + trade.dividend_per_economic_unit)
                                / (trade.entry_fill_economic * (1 + core.FEE * cost_multiple)) - 1)
                assert np.isclose(expected_net, trade.net_return, rtol=1e-11, atol=1e-12)
            if observed_exit is not None and observed_exit.tz_convert(core.ET).tz_localize(None).normalize() <= end:
                completed += 1; total_pnl += trade.capital * trade.net_return
                checked_losses += int(trade.net_return < 0)
            else:
                pending += 1
                close = market.daily[symbol].loc[end, 'close']
                total_pnl += trade.units_economic * close + trade.accrued_dividends - trade.capital
            events.append({'symbol': symbol, 'entry_at': trade.entry_at, 'reason': trade.reason,
                           'exit_at': trade.exit_at, 'order_and_first_event_agree': True})
        assert completed == score['completed_trades'] and pending == score['pending_trades']
        assert np.isclose(1 + total_pnl, equity.equity.iloc[-1], rtol=1e-11, atol=1e-12)
        reports.append({'strategy': name, 'orders': len(orders), 'completed': completed,
                        'pending': pending, 'independent_primary_mean': mean,
                        'equity_equals_all_realized_and_open_position_pnl': True,
                        'trade_checks': events})
    result = {'pass': True, 'frozen_inputs_verified': len(inputs),
              'causal_cutoff_records_checked': sum(x['status'] == 'ready' for x in coverage),
              'orders_checked_across_policies': checked_orders,
              'losing_trades_retained_across_policies': checked_losses,
              'strategies': reports,
              'scope': 'Independent order/first-touch/expiry/equity arithmetic; cross-policy duplicates are not independent samples; no broker fill proof'}
    core.dump(run / 'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'strategies'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=core.DATA_ROOT)
    args = parser.parse_args(); verify(args.run, args.data_root)
