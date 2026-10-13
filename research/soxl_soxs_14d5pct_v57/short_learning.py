"""Short-feedback base targets; original fourteen-day calibration and execution."""
from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v56'))
import activity_learning as prior
from activity_learning import (core, np, pd, read, dump, sha, labels, day_cutoff,
    restrict_market, development_gate, tag, fit_clock, predict, training_dates,
    registry, fit_day, issue, decide, CLOCKS, OLD_CLOCKS, load_sources, load_panel, PANEL_ROOT)

SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE))
VARIANTS = ['fast', 'lagged']
ROUTES = ['fast_band', 'fast_point', 'lagged_band', 'source_v56', 'source_v55',
          'premarket_only', 'cash', 'double_cost']


def columns(clock):
    return prior.columns(clock, 'full')


def verified_source():
    source = core.ROOT / 'research/soxl_soxs_14d5pct_v56/evidence/delivery_manifest.json'
    m = read(source)
    run = core.ROOT / '.cache/soxl_soxs_14d5pct_v56/runs/development_20261013'
    for field, folder in [('top_level_sha256', ''), ('training_records_sha256', 'training'),
                          ('daily_records_sha256', 'daily'), ('source_daily_records_sha256', 'source_daily'),
                          ('base_models_sha256', 'models'), ('baseline_models_sha256', 'baseline_models'),
                          ('executed_sources_sha256', 'executed_sources')]:
        for f, h in m['runs'][run.name][field].items():
            assert sha(run / folder / f) == h, f
    for f, h in m['delivery_files_sha256'].items():
        assert sha(source.parents[1] / f) == h, f
    v = read(run / 'verification.json')
    assert v['passed'] and v['results_sha256'] == sha(run / 'results.json')
    return run, sha(source)


def short_label(market, symbol, at):
    """Full next-session window must be observed, including after an early hit."""
    at = core.timestamp(at)
    date = at.tz_convert(core.ET).tz_localize(None).normalize()
    after = market.sessions[market.sessions > date]
    out = {'symbol': symbol, 'decision_at': at, 'date': date, 'known': False,
           'reason': 'pending_calendar', 'release': None, 'nominal_release': None,
           'net_return': None, 'next_date': None}
    if date not in market.calendar or not len(after):
        return out
    next_date = after[0]
    deadline = pd.Timestamp(market.calendar[next_date]['close_at'])
    nominal = deadline + pd.Timedelta(seconds=1)
    out.update(next_date=next_date, short_close_at=deadline, nominal_release=nominal)
    frame = market.bars[symbol]
    all_days = frame[(frame.date >= date) & (frame.date <= next_date) & (frame.session_type == 'regular')]
    required = pd.DatetimeIndex([]).tz_localize('UTC')
    for day in [date, next_date]:
        cal = market.calendar[day]
        grid = pd.date_range(pd.Timestamp(cal['open_at']) + pd.Timedelta(minutes=5),
                             pd.Timestamp(cal['close_at']), freq='5min')
        required = required.append(grid)
    good = (all_days.index.equals(required) and all_days.start_at.eq(all_days.end_at - pd.Timedelta(minutes=5)).all()
            and all_days.available_at.notna().all() and all_days.available_at.ge(all_days.end_at).all())
    if not good:
        out['reason'] = 'missing_full_short_window'
        return out
    eligible = all_days[(all_days.date == date) & (all_days.start_at > at + pd.Timedelta(seconds=30))]
    if eligible.empty or (eligible.iloc[0].start_at - at).total_seconds() > 360:
        out['reason'] = 'no_eligible_entry'
        return out
    first = eligible.iloc[0]
    fill_raw = float(first.open) * (1 + core.SLIP)
    entry = fill_raw * float(first.share_factor)
    target = math.ceil(fill_raw * 1.05 * 100 - 1e-10) / 100 * float(first.share_factor)
    stop = entry * .94
    path = all_days[all_days.start_at >= first.start_at]
    reason = 'short_close'
    end = path.iloc[-1]
    exit_fill = float(end.econ_close) * (1 - core.SLIP)
    both = gap = False
    for row in path.itertuples():
        factor = float(row.share_factor)
        stop_now = math.floor(stop / factor * 100 + 1e-10) / 100 * factor
        target_now = math.ceil(target / factor * 100 - 1e-10) / 100 * factor
        if row.econ_low <= stop_now:
            reason = 'risk_stop'; end = path.loc[row.end_at]
            exit_fill = min(float(row.econ_open), stop_now) * (1 - core.SLIP)
            both = bool(row.econ_high >= target_now); gap = bool(row.econ_open < stop_now)
            break
        if row.econ_high >= target_now:
            reason = 'target'; end = path.loc[row.end_at]; exit_fill = target_now
            break
    divs = []
    for action in market.actions[symbol]:
        ex = pd.Timestamp(action['date'])
        if action['type'] == 'CashDividends' and date < ex <= end.date:
            factor = float(all_days[all_days.date == ex].share_factor.iloc[0])
            divs.append({'ex': ex, 'economic_amount': float(action['value']) * factor})
    dividend = sum(d['economic_amount'] for d in divs)
    value = (exit_fill * (1 - core.FEE) + dividend) / (entry * (1 + core.FEE)) - 1
    out.update(known=True, reason=reason, release=max(nominal, all_days.available_at.max()),
               entry_at=first.start_at, entry_fill_economic=entry, target_economic=target,
               exit_at=end.end_at, exit_fill_economic=exit_fill, net_return=value,
               same_bar_both_barriers=both, gap_below_stop=gap, dividends=divs,
               dividend_per_economic_unit=dividend, required_rows=len(required))
    return out


def build_short_panel(original, records):
    panel = original.copy()
    for c in ['C', 'D', 'M']:
        panel[c] = np.nan
    for c in ['release', 'terminal_release', 'short_nominal_release']:
        panel[c] = pd.Series(pd.NaT, index=panel.index, dtype='datetime64[ns, UTC]')
    keyed = {(pd.Timestamp(r['decision_at']), r['symbol']): r for r in records}
    for at in panel.index:
        pair = [keyed[(at, s)] for s in core.ETFS]
        nominal = [pd.Timestamp(x['nominal_release']) for x in pair if x['nominal_release'] is not None]
        if len(nominal) == 2:
            panel.loc[at, 'short_nominal_release'] = max(nominal)
        if not all(x['known'] for x in pair):
            continue
        values = np.log1p([x['net_return'] for x in pair])
        panel.loc[at, ['C', 'D', 'M']] = [values.mean(), (values[0] - values[1]) / 2, abs(values[0] - values[1]) / 2]
        panel.loc[at, ['release', 'terminal_release']] = max(pd.Timestamp(x['release']) for x in pair)
    return panel


def training_panels(original, short):
    lagged = short.copy()
    ready = short.terminal_release.notna() & original.terminal_release.notna()
    lagged['terminal_release'] = pd.Series(pd.NaT, index=short.index, dtype='datetime64[ns, UTC]')
    for at in short.index[ready]:
        lagged.loc[at, 'terminal_release'] = max(short.loc[at, 'terminal_release'], original.loc[at, 'terminal_release'])
    lagged['release'] = lagged.terminal_release
    return {'fast': short, 'lagged': lagged}


def short_gap(short, at):
    cutoff = day_cutoff(at)
    past = short[short.feature_ready & (short.clock == short.loc[at, 'clock']) & (short.date < short.loc[at, 'date'])]
    return bool(((past.short_nominal_release < cutoff) &
                 (past.terminal_release.isna() | (past.terminal_release >= cutoff))).any())


def fit_bases(panels, at):
    models, fits, raws = {}, {}, {}
    for kind in VARIANTS:
        panel = panels[kind]; same = panel[panel.clock == panel.loc[at, 'clock']]
        dates, fb = training_dates(same, day_cutoff(at))
        model, info = fit_clock(same, dates['early'], columns(panel.loc[at, 'clock']))
        raw = predict(model, same.loc[[at]])
        models[kind] = model
        fits[kind] = {'fit': info, 'feedback': fb, 'raw': raw, 'short_unresolved_expired': short_gap(panels['fast'], at)}
        raws[kind] = np.array([raw[s] for s in core.ETFS]) if raw else None
    return models, fits, raws


def feedback_for_action(daily, fit):
    return {**daily['feedback'], 'expired_unresolved_prefix': bool(daily['feedback']['expired_unresolved_prefix'] or fit['short_unresolved_expired'])}


def original_fit(panel, at, cols):
    same = panel[panel.clock == panel.loc[at, 'clock']]
    dates, fb = training_dates(same, day_cutoff(at))
    model, info = fit_clock(same, dates['early'], cols)
    raw = predict(model, same.loc[[at]])
    return model, {'fit': info, 'feedback': fb, 'raw': raw}, np.array([raw[s] for s in core.ETFS]) if raw else None


def events(at, date, clock, estimates, daily, fits, old56, old55):
    feedback = {k: feedback_for_action(daily[k], fits[k]) for k in VARIANTS}
    primary = decide(estimates['fast'], feedback['fast'], True)
    cash = {'ready': True, 'action': 'CASH', 'allocation': 0.}
    actions = {'fast_band': primary, 'fast_point': decide(estimates['fast'], feedback['fast'], False),
               'lagged_band': decide(estimates['lagged'], feedback['lagged'], True),
               'source_v56': old56, 'source_v55': old55,
               'premarket_only': primary if clock == CLOCKS[0] else cash,
               'cash': cash, 'double_cost': primary}
    return {k: {**e, 'at': at, 'date': date} for k, e in actions.items()}
