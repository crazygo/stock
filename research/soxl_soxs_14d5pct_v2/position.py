"""Paper position management; exit obligations are independent of new buys."""
import math

import pandas as pd

import core


def manage(market, position, decision_at):
    at = core.timestamp(decision_at)
    result = {'action': 'NO_ACTION', 'reason': 'existing_position',
              'broker_order_submitted': False, 'position': position}
    if position.get('closed') or position.get('take_profit_filled'):
        result.update({'reason': 'position_closed_confirmed', 'exit_action': 'NONE',
                       'cash_rule': 'Sale proceeds reusable from the next official session'})
        return result
    missing = [x for x in ['symbol', 'entry_at', 'entry_fill_raw'] if x not in position]
    if missing:
        result.update({'reason': 'position_entry_details_missing', 'missing_fields': missing})
        return result
    symbol = position['symbol']
    if symbol not in core.ETFS:
        raise ValueError('Position must be SOXL or SOXS')
    entry = core.timestamp(position['entry_at'])
    if entry > at or float(position['entry_fill_raw']) <= 0:
        raise ValueError('Position entry/fill is invalid for this analysis timestamp')
    day = entry.tz_convert(core.ET).tz_localize(None).normalize()
    expiry = core.expiry_day(market, day)
    if expiry is None:
        result['reason'] = 'position_future_calendar_missing'; return result
    deadline = pd.Timestamp(market.calendar[expiry]['close_at'])
    current_day = at.tz_convert(core.ET).tz_localize(None).normalize()
    ratio = math.prod(float(a['value']) for a in market.actions[symbol]
                      if a['type'] == 'Splits' and day < pd.Timestamp(a['date']) <= current_day)
    original_limit = math.ceil(float(position['entry_fill_raw']) * 1.05 * 100 - 1e-10) / 100
    limit = math.ceil(original_limit / ratio * 100 - 1e-10) / 100
    quantity = position.get('quantity')
    if quantity is not None and float(quantity) <= 0:
        raise ValueError('Position quantity must be positive')
    result['take_profit_order'] = {'side': 'SELL', 'type': 'LIMIT', 'tif': 'GTC',
                                    'limit_price': limit, 'active_session': 'regular',
                                    'quantity': float(quantity) * ratio if quantity is not None else None,
                                    'price_basis': 'raw_actual_fill_adjusted_only_for_real_splits',
                                    'broker_submission': False}
    due = at >= deadline
    result.update({'expiry_at': deadline, 'exit_action': 'FORCE_SELL_AT_EXPIRY' if due else 'WAIT_FOR_LIMIT_OR_EXPIRY',
                   'force_sell_intent': {'side': 'SELL', 'type': 'MARKET_AT_REGISTERED_CLOSE',
                                         'scheduled_at': deadline, 'due': due,
                                         'cancel_attached_take_profit_first': True,
                                         'only_if_position_still_open': True,
                                         'quantity': float(quantity) * ratio if quantity is not None else None,
                                         'broker_submission': False},
                   'reason': 'expiry_force_exit_due' if due else 'existing_position_before_expiry'})
    if at > deadline:
        result['late_execution_warning'] = 'This is an exit obligation; no retrospective close fill is invented when queried after the deadline'
    return result
