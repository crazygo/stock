"""One continuous capital ledger, shared by model and baseline policies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import ET, ETFS, FEE


def replay(market, candidates, labels, actions, start, end, cost_multiple=1.0):
    days = market.sessions[(market.sessions >= pd.Timestamp(start)) & (market.sessions <= pd.Timestamp(end))]
    label_map = {(x['decision_at'], x['symbol']): x for x in labels}
    cash = 1.0; unsettled = []; receivables = []; position = None
    trades = []; marks = []; decisions = []; orders = []; entry_days = set(); exit_days = set()
    for day in days:
        due = [x for x in unsettled if x['settle'] <= day]
        cash += sum(x['amount'] for x in due)
        unsettled = [x for x in unsettled if x['settle'] > day]
        paid = [x for x in receivables if x['pay'] <= day]
        cash += sum(x['amount'] for x in paid)
        receivables = [x for x in receivables if x['pay'] > day]
        if position is not None:
            for event in position['dividends']:
                if event['ex'] == day:
                    amount = position['units_economic'] * event['economic_amount']
                    position['accrued_dividends'] += amount
                    if event['pay'] <= day:
                        cash += amount
                    else:
                        receivables.append({'pay': event['pay'], 'amount': amount})
        day_candidates = candidates[candidates.date == day]
        for decision_at, row in day_candidates.iterrows():
            if position is not None and position['exit_at'] is not None and position['exit_at'] <= decision_at:
                proceeds = position['units_economic'] * position['exit_fill_economic'] * (1 - FEE * cost_multiple)
                next_days = market.sessions[market.sessions > day]
                unsettled.append({'settle': next_days[0], 'amount': proceeds})
                trades.append(position); position = None; exit_days.add(day)
            requested = actions.get(decision_at, 'CASH')
            status = 'no_action'
            if position is not None:
                status = 'position_held'
            elif day in entry_days or day in exit_days:
                status = 'daily_entry_limit'
            elif cash <= 1e-15:
                status = 'cash_unsettled'
            elif requested in ETFS:
                label = label_map[(decision_at, requested)]
                if 'entry_fill_economic' not in label:
                    status = label['reason']
                else:
                    units = cash / (label['entry_fill_economic'] * (1 + FEE * cost_multiple))
                    position = {**label, 'capital': cash, 'units_economic': units,
                                'accrued_dividends': 0.0}
                    cash = 0.0; entry_days.add(day); status = 'buy_' + requested
                    orders.append({'decision_at': decision_at, 'entry_at': label['entry_at'],
                                   'symbol': requested, 'raw_quantity_at_entry': units * label['entry_share_factor'],
                                   **label['attached_order']})
            decisions.append({'decision_at': decision_at, 'date': day, 'clock': row.clock,
                              'requested_action': requested, 'decision': status})
        if position is not None and position['exit_at'] is not None:
            exit_day = position['exit_at'].tz_convert(ET).tz_localize(None).normalize()
            if exit_day == day:
                proceeds = position['units_economic'] * position['exit_fill_economic'] * (1 - FEE * cost_multiple)
                next_days = market.sessions[market.sessions > day]
                unsettled.append({'settle': next_days[0], 'amount': proceeds})
                trades.append(position); position = None; exit_days.add(day)
        asset = 0.0; symbol = 'CASH'; quality = True
        if position is not None:
            symbol = position['symbol']
            daily = market.daily[symbol]
            if day not in daily.index or not daily.loc[day, 'complete']:
                quality = False
                prior = daily.loc[:day]
                price = float(prior.close.iloc[-1]) if len(prior) else np.nan
            else:
                price = float(daily.loc[day, 'close'])
            asset = position['units_economic'] * price
        receivable = sum(x['amount'] for x in receivables)
        blocked_cash = sum(x['amount'] for x in unsettled)
        marks.append({'date': day, 'equity': cash + blocked_cash + asset + receivable,
                      'cash_settled': cash, 'cash_unsettled': blocked_cash,
                      'position_value': asset, 'receivables': receivable, 'symbol': symbol,
                      'mark_complete': quality})
    if position is not None:
        trades.append({**position, 'reason_at_evaluation_end': 'pending',
                       'event_after_evaluation_end': position['exit_at']})
    return pd.DataFrame(marks).set_index('date'), pd.DataFrame(trades), pd.DataFrame(decisions), pd.DataFrame(orders)


def statistics(equity, trades, requested_months):
    if equity.empty:
        return {'primary_pass': False, 'reason': 'no_equity_records'}
    end = equity.index[-1]
    if len(trades):
        done_mask = trades.exit_at.notna() & (pd.to_datetime(trades.exit_at, utc=True).dt.tz_convert(ET).dt.tz_localize(None).dt.normalize() <= end)
        done = trades[done_mask]
    else:
        done = trades
    prior = 1.0; months = []
    for month, group in equity.groupby(equity.index.to_period('M')):
        value = float(group.equity.iloc[-1])
        months.append({'month': str(month), 'return': value / prior - 1,
                       'end_equity': value, 'mark_complete': bool(group.mark_complete.all()),
                       'end_position': group.symbol.iloc[-1]})
        prior = value
    selected = [m for m in months if m['month'] in requested_months]
    complete = bool(requested_months) and len(selected) == len(requested_months) and all(m['mark_complete'] for m in selected)
    average = float(np.mean([m['return'] for m in selected])) if selected else None
    geometric = float(np.prod([1 + m['return'] for m in selected]) ** (1 / len(selected)) - 1) if selected else None
    curve = np.r_[1.0, equity.equity.to_numpy()]
    attribution = []
    for month in requested_months:
        entries = trades[trades.entry_date.dt.to_period('M') == pd.Period(month)] if len(trades) else trades
        closed = done[done.entry_date.dt.to_period('M') == pd.Period(month)] if len(done) else done
        attribution.append({'entry_month': month, 'entries': len(entries), 'completed': len(closed),
                            'pending': len(entries) - len(closed),
                            'sum_completed_trade_net_returns': float(closed.net_return.sum()) if len(closed) else 0.0,
                            'warning': 'Sum of individual trade percentages is attribution, not the capital-ledger monthly return'})
    return {'primary_metric': 'requested_months_net_return_arithmetic_mean',
            'primary_average': average, 'primary_threshold': 0.10,
            'primary_pass': bool(complete and average >= 0.10),
            'requested_months_complete': complete, 'geometric_average': geometric,
            'continuous_return': float(equity.equity.iloc[-1] - 1),
            'close_max_drawdown': float((curve / np.maximum.accumulate(curve) - 1).min()),
            'entries': len(trades), 'completed_trades': len(done), 'pending_trades': len(trades) - len(done),
            'target_exits': int((done.reason == 'target').sum()) if len(done) else 0,
            'expiry_exits': int((done.reason == 'expiry').sum()) if len(done) else 0,
            'mean_losing_trade': float(done.loc[done.net_return < 0, 'net_return'].mean()) if len(done) and (done.net_return < 0).any() else None,
            'monthly': months, 'entry_month_attribution': attribution,
            'prospective_expected_return_established': False,
            'execution_evidence': 'OHLC limit/expiry simulation, not broker fill confirmation'}
