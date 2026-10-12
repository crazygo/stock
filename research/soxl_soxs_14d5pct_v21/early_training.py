"""Daily base training and calibration on a chronological terminal prefix."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v20'))
from resolved_feedback import core, prefix_receipt, risk_decision, bank, restrict_market, development_gate, day_cutoff
from paired_entry import fit, predict, paired_calibration, decision, BASE_COLUMNS, REGIME_COLUMNS
from verify_continuation import key

ROUTES = {
    'early_base_cal': ('early', 'early', 'paired'),
    'early_base_full_cal': ('early', 'full', 'paired'),
    'full_base_early_cal': ('full', 'early', 'paired'),
    'early_raw': ('early', 'early', 'raw'),
    'full_calibrated': ('full', 'full', 'paired'),
}


def terminal_panel(market, source_panel, labels):
    panel = source_panel.copy()
    panel['terminal_release'] = pd.Series(pd.NaT, index=panel.index, dtype='datetime64[ns, UTC]')
    receipts = []
    for at, row in panel[panel.feature_ready].iterrows():
        sides = {s: prefix_receipt(market, labels[key(at, s)]) for s in core.ETFS}
        release = max(v['release'] for v in sides.values()) if all(v['known'] for v in sides.values()) else None
        if release is not None:
            panel.loc[at, 'terminal_release'] = release
        receipts.append({'at': at, 'date': row.date, 'terminal_release': release, 'sides': sides})
    return panel, receipts


def training_dates(panel, cutoff):
    day = cutoff.tz_convert(core.ET).tz_localize(None).normalize()
    past = panel[panel.feature_ready & (panel.date < day) & (panel.index < cutoff)]
    full = past[past.release.notna() & past.release.lt(cutoff)]
    known = past.terminal_release.notna() & past.terminal_release.lt(cutoff)
    missing = np.flatnonzero(~known.to_numpy())
    stop = int(missing[0]) if len(missing) else len(past)
    prefix = past.iloc[:stop]
    expired = stop < len(past) and past.iloc[stop].date + pd.Timedelta(days=14) <= day
    # Unknown outcomes cannot disappear from the source-date universe.
    for selected in [prefix, full]:
        assert np.isfinite(selected[[*BASE_COLUMNS, *REGIME_COLUMNS, 'C', 'D']]).all().all()
    return {'early': list(prefix.date), 'full': list(full.date)}, {
        'prior_dates': len(past), 'selected_early_dates': len(prefix), 'selected_full_dates': len(full),
        'last_early_date': prefix.date.max() if len(prefix) else None,
        'last_full_date': full.date.max() if len(full) else None,
        'first_pending_date': past.iloc[stop].date if stop < len(past) else None,
        'known_but_waiting': int(known.iloc[stop+1:].sum()) if stop < len(past) else 0,
        'expired_unresolved_prefix': bool(expired)}


def forecast_routes(panel, at, full_event, early_history, full_history):
    dates, receipt = training_dates(panel, day_cutoff(at))
    model, fit_info = fit(panel, dates['early'], 'regret_regime')
    raw = predict(model, panel.loc[[at]])
    record = {'fit': fit_info, 'feedback': receipt, 'forecasts': {}}
    for name, (base_mode, cal_mode, method) in ROUTES.items():
        history = early_history if base_mode == 'early' else full_history
        current = raw if base_mode == 'early' else full_event['raw']
        calibration = paired_calibration(history, panel, dates[cal_mode])
        event = decision(current, calibration, method)
        if name != 'full_calibrated' and receipt['expired_unresolved_prefix']:
            event.update(ready=False, action='CASH', adjusted=None, top_symbol=None)
        record['forecasts'][name] = {'calibration': calibration, 'event': event}
    return model, record, raw


def capital_decision(risk_bank, cutoff, at, forecast):
    # Original as-issued risk coordinates are required, even when new forecasts
    # became ready earlier. Never backfill old coordinates with a new model.
    if forecast['ready'] and (at, forecast['top_symbol']) not in risk_bank.index:
        return {'ready': False, 'action': 'CASH', 'allocation': 0.,
                'reason': 'original_risk_coordinates_unavailable', 'top_symbol': forecast['top_symbol'], 'fit': None}
    return risk_decision(risk_bank, cutoff, at, forecast)
