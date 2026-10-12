"""V25 full-cash-cycle ledger; retain the original V15 accounting and exits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from allocation import core
ET, ETFS, FEE = core.ET, core.ETFS, core.FEE


def replay(market, candidates, labels, actions, start, end, cost_multiple=1.0, allocations=None, cooldown_sessions=0, require_full_cash=True):
    days = market.sessions[(market.sessions >= pd.Timestamp(start)) & (market.sessions <= pd.Timestamp(end))]
    label_map = {(x['decision_at'], x['symbol']): x for x in labels}
    if allocations is None: allocations = {at: (1. if a in ETFS else 0.) for at,a in actions.items()}
    if cooldown_sessions < 0: raise ValueError('Invalid cooldown')
    for at,a in actions.items():
        fraction = allocations[at]
        if not np.isfinite(fraction) or not 0 <= fraction <= 1: raise ValueError('Invalid allocation')
        if (a in ETFS) != (fraction > 0): raise ValueError('Action/allocation mismatch')
    cooldown_until = pd.Timestamp.min
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
                if position['net_return'] < 0 and cooldown_sessions:
                    cooldown_until = next_days[cooldown_sessions - 1]
                trades.append(position); position = None; exit_days.add(day)
            requested = actions.get(decision_at, 'CASH')
            allocation = allocations.get(decision_at, 0.)
            status = 'no_action'
            if position is not None:
                status = 'position_held'
            elif day in entry_days or day in exit_days:
                status = 'daily_entry_limit'
            elif day <= cooldown_until:
                status = 'loss_cooldown'
            elif cash <= 1e-15:
                status = 'cash_unsettled'
            elif require_full_cash and (unsettled or receivables):
                status = 'waiting_for_full_cash'
            elif requested in ETFS:
                label = label_map[(decision_at, requested)]
                if 'entry_fill_economic' not in label:
                    status = label['reason']
                else:
                    capital = cash * allocation
                    units = capital / (label['entry_fill_economic'] * (1 + FEE * cost_multiple))
                    position = {**label, 'capital': capital, 'units_economic': units, 'allocation': allocation,
                                'accrued_dividends': 0.0}
                    cash -= capital; entry_days.add(day); status = 'buy_' + requested
                    orders.append({'decision_at': decision_at, 'entry_at': label['entry_at'],
                                   'symbol': requested, 'raw_quantity_at_entry': units * label['entry_share_factor'],
                                   **label['attached_order']})
            decisions.append({'decision_at': decision_at, 'date': day, 'clock': row.clock,
                              'requested_action': requested, 'requested_allocation': allocation, 'decision': status})
        if position is not None and position['exit_at'] is not None:
            exit_day = position['exit_at'].tz_convert(ET).tz_localize(None).normalize()
            if exit_day == day:
                proceeds = position['units_economic'] * position['exit_fill_economic'] * (1 - FEE * cost_multiple)
                next_days = market.sessions[market.sessions > day]
                unsettled.append({'settle': next_days[0], 'amount': proceeds})
                if position['net_return'] < 0 and cooldown_sessions:
                    cooldown_until = next_days[cooldown_sessions - 1]
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

