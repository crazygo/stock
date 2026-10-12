"""Available intraday states and exits that respect still-active protective orders."""
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v9'))
from continuation import (core, ENTRY_COLUMNS, POSITION_COLUMNS, entry_state, label_key, alive,
                          baseline_labels, joint_releases, eligible_dates, fit, calibration, predict,
                          choose_entry, day_cutoff)

MICRO_FIELDS = ['session_return', 'vwap_distance', 'last5', 'recent15', 'range_position',
                'efficiency', 'observed_range', 'bars', 'relative_volume']
DIRECTIONAL = {'session_return', 'vwap_distance', 'last5', 'recent15', 'efficiency'}
MICRO_COLUMNS = ['regular_ready'] + ['micro_' + p + '_' + k for p in ['own', 'sector', 'qqq'] for k in MICRO_FIELDS]
MANAGER_COLUMNS = ENTRY_COLUMNS + POSITION_COLUMNS + MICRO_COLUMNS


def context(market):
    """One cumulative prefix per symbol/date, before any trade outcomes are inspected."""
    contexts = {}; by_day = {}
    for symbol in core.SYMBOLS:
        bars = market.bars[symbol]; by_day[symbol] = {d: f for d, f in bars.groupby('date')}
        daily = market.daily[symbol]; rows = []
        for date, f in bars[bars.session_type.eq('regular')].groupby('date'):
            past = daily.reindex(market.sessions[market.sessions < date][-20:])
            if len(past) != 20 or not past.complete.fillna(False).all():
                continue
            prior_volume = float(past.volume.mean())
            if prior_volume <= 0:
                continue
            opening = f.econ_open.iloc[0]; close = f.econ_close; high = f.econ_high.cummax(); low = f.econ_low.cummin()
            volume = f.econ_volume.cumsum(); vwap = ((f.econ_high + f.econ_low + close) / 3 * f.econ_volume).cumsum() / volume.replace(0, np.nan)
            delta = close.diff(); delta.iloc[0] = close.iloc[0] - opening
            oldest = f.econ_open.shift(2).fillna(opening)
            expected = pd.date_range(market.calendar[date]['open_at'], market.calendar[date]['close_at'], freq='5min', inclusive='right')
            complete = np.logical_and.accumulate(expected.get_indexer(f.index) == np.arange(len(f)))
            rows.append(pd.DataFrame({'session_return': close / opening - 1, 'vwap_distance': close / vwap - 1,
                'last5': close / f.econ_open - 1, 'recent15': close / oldest - 1,
                'range_position': ((close-low)/(high-low).replace(0, np.nan)).fillna(.5),
                'efficiency': ((close-opening)/delta.abs().cumsum().replace(0, np.nan)).fillna(0.),
                'observed_range': (high-low)/opening, 'bars': np.arange(1, len(f)+1)/78.,
                'relative_volume': volume/prior_volume, 'observed_high': high,
                'available_at': f.available_at.cummax(), 'prefix_complete': complete,
                'date': date}, index=f.index))
        contexts[symbol] = pd.concat(rows).sort_index()
    return contexts, by_day


def state_at(market, contexts, by_day, premarket_row, label, at):
    day = at.tz_convert(core.ET).tz_localize(None).normalize(); symbol = label['symbol']
    if premarket_row.date != day:
        raise ValueError('Premarket context must belong to review date')
    opening = pd.Timestamp(market.calendar[day]['open_at']); regular = at >= opening
    observed = by_day[symbol].get(day, pd.DataFrame())
    observed = observed[(observed.available_at <= at) & (observed.end_at <= at)] if len(observed) else observed
    if observed.empty or at - observed.end_at.iloc[-1] > pd.Timedelta(minutes=6):
        return None, {'reason': 'holding_mark_missing_or_stale'}
    quote = observed.iloc[-1]; price = float(quote.econ_close)
    required = market.sessions[(market.sessions >= label['entry_date']) & (market.sessions < day)]
    past = market.daily[symbol].reindex(required)
    if len(past) and (not past.complete.fillna(False).all() or not (past.available_at <= at).all()):
        return None, {'reason': 'prior_holding_prefix_missing'}
    peak = max(label['entry_fill_economic'], float(past.high.max()) if len(past) else 0.)
    state = {**entry_state(premarket_row, symbol), 'regular_ready': float(regular)}
    audit = {'mark_bar_end': quote.end_at, 'mark_available_at': quote.available_at, 'context_ends': {}}
    for source, prefix in [(symbol, 'own'), ('SOXX', 'sector'), ('QQQ', 'qqq')]:
        if regular:
            f = contexts[source]
            end = pd.Timestamp(at.floor('5min'))
            if end not in f.index or f.loc[end, 'date'] != day or not f.loc[end, 'prefix_complete'] or f.loc[end, 'available_at'] > at:
                return None, {'reason': source + ':regular_prefix_missing'}
            row = f.loc[end]
            if prefix == 'own': peak = max(peak, float(row.observed_high))
            audit['context_ends'][source] = {'bar_end': end, 'available_at': row.available_at}
            for k in MICRO_FIELDS:
                sign = -1. if symbol == 'SOXS' and prefix != 'own' and k in DIRECTIONAL else 1.
                # Range location is reflected for the inverse direction, not made negative.
                value = 1-float(row[k]) if symbol == 'SOXS' and prefix != 'own' and k == 'range_position' else sign*float(row[k])
                state['micro_'+prefix+'_'+k] = value
        else:
            state.update({'micro_'+prefix+'_'+k: 0. for k in MICRO_FIELDS})
    age = (day-label['entry_date']).days; entry = label['entry_fill_economic']
    state.update(age_calendar=float(age), remaining_calendar=float(13-age), mark_return=price/entry-1,
        target_distance=label['limit_economic']/price-1, stop_distance=entry*.94/price-1,
        observed_peak_drawdown=price/peak-1)
    audit['observed_peak'] = peak
    if not np.isfinite([state[c] for c in MANAGER_COLUMNS]).all():
        return None, {'reason': 'nonfinite_holding_context'}
    return state, audit


def delayed_sale(by_day, label, at, multiple):
    """Outcome label for an intent, including a barrier filling during order latency."""
    day = at.tz_convert(core.ET).tz_localize(None).normalize()
    f = by_day[label['symbol']].get(day, pd.DataFrame())
    choices = f[f.session_type.eq('regular') & (f.start_at > at+pd.Timedelta(seconds=30))] if len(f) else f
    if choices.empty or choices.start_at.iloc[0]-at > pd.Timedelta(minutes=6):
        return None
    first = choices.iloc[0]
    barrier_first = (label['exit_at'] is not None and label['exit_bar_start'] < first.start_at)
    if barrier_first:
        # The existing OCO exits before the replacement market order can execute.
        return {'scheduled_open': first.start_at, 'barrier_during_latency': True,
                'exit_at': label['exit_at'], 'exit_bar_start': label['exit_bar_start'],
                'exit_fill_economic': label['exit_fill_economic'],
                'sale_net': label['exit_fill_economic']*(1-label['fee_each_side'])}
    factor = float(first.share_factor)
    target = math.ceil(label['limit_economic']/factor*100-1e-10)/100
    fill = min(float(first.open), target)*factor*(1-core.SLIP*multiple)
    return {'scheduled_open': first.start_at, 'barrier_during_latency': False,
            'exit_at': first.start_at, 'exit_bar_start': first.start_at,
            'exit_fill_economic': fill, 'sale_net': fill*(1-core.FEE*multiple)}


def target(label, sale, day):
    if sale is None or label['exit_at'] is None or not np.isfinite(label['net_return']):
        return np.nan
    if sale['barrier_during_latency']:
        return 0.
    remaining_div = sum(d['economic_amount'] for d in label['dividends'] if d['ex'] > day)
    return float(np.log((label['exit_fill_economic']*(1-label['fee_each_side'])+remaining_div)/sale['sale_net']))


def apply_exit(label, sale, at, reason='intraday_value_exit'):
    if sale is None or sale['barrier_during_latency']:
        return label
    day = at.tz_convert(core.ET).tz_localize(None).normalize()
    divs = [d for d in label['dividends'] if d['ex'] <= day]
    dividend = sum(d['economic_amount'] for d in divs)
    net = (sale['sale_net']+dividend)/(label['entry_fill_economic']*(1+label['fee_each_side']))-1
    return {**label, **{k: sale[k] for k in ['exit_at','exit_bar_start','exit_fill_economic']},
            'manager_decision_at': at, 'reason': reason, 'net_return': net, 'target_hit': 0,
            'prefix_complete': True, 'dividends': divs, 'dividend_per_economic_unit': dividend,
            'same_bar_both_barriers': False, 'gap_below_stop': False, 'protection_raised': False}


def prepare_reviews(market, features, baseline, prepared, protocol, multiple=1):
    contexts, by_day = prepared; records = []
    for key, label in baseline.items():
        if 'entry_fill_economic' not in label: continue
        subset = features[(features.date >= label['entry_date']) & (features.date <= label['expiry_date'])]
        for _, row in subset.iterrows():
            for clock in protocol['review_clocks_et']:
                at = pd.Timestamp(str(row.date.date())+' '+clock, tz=core.ET).tz_convert('UTC')
                if not alive(label, at) or at >= pd.Timestamp(market.calendar[row.date]['close_at']): continue
                state, audit = state_at(market, contexts, by_day, row, label, at)
                sale = delayed_sale(by_day, label, at, multiple)
                records.append({'id': key+'@'+at.isoformat(), 'key': key, 'at': at, 'entry_date': label['entry_date'],
                    'symbol': label['symbol'], 'state': state, 'audit': audit, 'sale': sale,
                    'target': target(label, sale, row.date)})
    return records


def simple_exit(state):
    return state is not None and bool(state['regular_ready']) and all(state[k] < 0 for k in [
        'micro_sector_session_return','micro_sector_vwap_distance','micro_sector_recent15',
        'micro_qqq_session_return','micro_qqq_vwap_distance'])
