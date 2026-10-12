"""Causal protective exits with adverse gap and ambiguous-bar treatment."""
from __future__ import annotations
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v2'
if str(V2) not in sys.path:
    sys.path.insert(0, str(V2))
import core


def first_exit(path, entry, target, stop_fraction, breakeven_trigger, fee, slip):
    """Economic prices; raise protection only after the trigger bar is available plus order latency."""
    stop = entry * (1 - stop_fraction)
    raised = False
    pending = None
    for i, row in enumerate(path.itertuples()):
        if pending is not None and row.start_at > pending[1]:
            stop = max(stop, pending[0]); raised = True; pending = None
        factor = float(row.share_factor)
        rounded_stop = math.floor(stop / factor * 100 + 1e-10) / 100 * factor
        rounded_target = math.ceil(target / factor * 100 - 1e-10) / 100 * factor
        stop_hit = row.econ_low <= rounded_stop
        target_hit = row.econ_high >= rounded_target
        if stop_hit:
            return {'offset': i, 'reason': 'profit_protection' if raised else 'risk_stop',
                    'fill': min(row.econ_open, rounded_stop) * (1 - slip),
                    'stop_economic': rounded_stop, 'same_bar_both_barriers': bool(target_hit),
                    'gap_below_stop': bool(row.econ_open < rounded_stop), 'protection_raised': raised}
        if target_hit:
            return {'offset': i, 'reason': 'target', 'fill': rounded_target,
                    'stop_economic': rounded_stop, 'same_bar_both_barriers': False,
                    'gap_below_stop': False, 'protection_raised': raised}
        if breakeven_trigger is not None and not raised and pending is None and row.econ_high >= entry * (1 + breakeven_trigger):
            # Round upward before the next bar so transaction costs are covered
            # when execution occurs exactly at the trigger; gaps still lose money.
            net_flat = entry * (1 + fee) / ((1 - fee) * (1 - slip))
            level = math.ceil(net_flat / factor * 100 - 1e-10) / 100 * factor
            pending = (level, row.available_at + pd.Timedelta(seconds=30))
    return None


def risk_label(market, symbol, at, rule, cost_multiple=1.0):
    base = core.label_entry(market, symbol, at, cost_multiple)
    if 'entry_fill_economic' not in base:
        return base
    out = dict(base)
    out.update({'risk_rule': rule['id'], 'stop_fraction': rule['stop_fraction'],
                'breakeven_trigger': rule['breakeven_trigger'], 'same_bar_both_barriers': False,
                'gap_below_stop': False, 'protection_raised': False})
    frame = market.bars[symbol]
    path = frame.loc[base['entry_at']:base['expiry_at']]
    path = path[(path.start_at >= base['entry_at']) & (path.session_type == 'regular')]
    event = first_exit(path, base['entry_fill_economic'], base['limit_economic'],
                       rule['stop_fraction'], rule['breakeven_trigger'],
                       core.FEE * cost_multiple, core.SLIP * cost_multiple)
    if event is None:
        return out
    bar = path.iloc[event['offset']]
    needed = market.sessions[(market.sessions >= base['entry_date']) & (market.sessions <= bar.date)]
    if not all(market.day_complete[symbol].get(d, False) for d in needed):
        out.update({'exit_at': None, 'exit_fill_economic': np.nan,
                    'reason': 'missing_prefix', 'target_hit': None, 'net_return': np.nan})
        return out
    divs = [d for d in base['dividends'] if d['ex'] <= bar.date]
    dividend = sum(d['economic_amount'] for d in divs)
    fee = core.FEE * cost_multiple
    net = (event['fill'] * (1 - fee) + dividend) / (base['entry_fill_economic'] * (1 + fee)) - 1
    out.update({k: v for k, v in event.items() if k not in ['offset', 'fill']})
    out.update({'exit_at': bar.end_at, 'exit_bar_start': bar.start_at,
                'exit_fill_economic': event['fill'], 'reason': event['reason'],
                'target_hit': int(event['reason'] == 'target'), 'net_return': net,
                'prefix_complete': True, 'dividends': divs, 'dividend_per_economic_unit': dividend})
    out['attached_order'] = {**base['attached_order'],
                             'protective_stop_fraction': rule['stop_fraction'],
                             'breakeven_trigger': rule['breakeven_trigger'],
                             'oco_cancel_other_exit_after_fill': True}
    return out


def direction(row, name):
    if name.startswith('trend'):
        values = [row['SOXX_ret' + name[5:]]]
    elif name == 'agree5_20':
        values = [row['SOXX_ret5'], row['SOXX_ret20']]
    elif name == 'agree1_5':
        values = [row['SOXX_ret1'], row['SOXX_ret5']]
    elif name == 'agree5_premarket':
        values = [row['SOXX_ret5'], row['SOXX_today_return']]
    else:
        raise ValueError(name)
    if all(v > 0 for v in values):
        return 'SOXL'
    if all(v < 0 for v in values):
        return 'SOXS'
    return 'CASH'
