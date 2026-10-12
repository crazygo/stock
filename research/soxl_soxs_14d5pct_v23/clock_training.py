"""Separate causal payoff models per clock, with sequential capital decisions."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v21'))
from early_training import (core, terminal_panel, training_dates, capital_decision, bank,
                            restrict_market, development_gate, day_cutoff)
from paired_entry import (BASE_COLUMNS, REGIME_COLUMNS, factor_frame, prepare, predict,
                          paired_calibration, decision, date_weights, baseline_labels, joint_releases)
from breadth import PANEL, PANEL_ROOT, load_panel

CLOCKS = ['09:25:30', '10:05:30', '10:35:30', '13:05:30']
EXTRA = ['intraday_'+c for c in ['sector_session','sector_vwap','sector_recent15','sector_previous15',
    'sector_high_distance','sector_low_distance','market_session','market_recent15',
    'breadth_session','breadth_vwap','breadth_recent_positive','sector_efficiency']]
ROUTES = ['context_sequential','context_after_open','base_sequential','premarket_only',
          'context_1005','context_1035','context_1305','context_raw','context_no_risk',
          'context_full_window','cash','double_cost']


def clock_volumes(bars, sessions, calendar, cutoff, clock):
    """Identical elapsed-time volume on 20 prior eligible official sessions."""
    valid_days = [d for d in sessions if d.tz_localize(core.ET) < cutoff and
                  pd.Timestamp(str(d.date())+' '+clock,tz=core.ET).tz_convert('UTC') < pd.Timestamp(calendar[d]['close_at'])]
    frame = pd.DataFrame(index=pd.DatetimeIndex(valid_days), columns=PANEL, dtype=float)
    audits = []
    for s in PANEL:
        groups = {d:g for d,g in bars[s][(bars[s].end_at<cutoff)&(bars[s].available_at<cutoff)].groupby('date')}
        for d in valid_days:
            at = pd.Timestamp(str(d.date())+' '+clock,tz=core.ET).tz_convert('UTC')
            start = (d.tz_localize(core.ET)+pd.Timedelta(hours=4)).tz_convert('UTC'); end = at-pd.Timedelta(seconds=30)
            expected = pd.date_range(start,end,freq='5min',inclusive='right').tz_convert('UTC')
            g = groups.get(d,bars[s].iloc[:0]); g = g[(g.end_at>start)&(g.end_at<=end)]
            good = (g.index.equals(expected) and g.available_at.notna().all() and
                    g.available_at.le(at).all() and g.available_at.ge(g.end_at).all() and
                    g.start_at.eq(g.end_at-pd.Timedelta(minutes=5)).all() and
                    g.econ_volume.ge(0).all() and g.econ_volume.sum()>0)
            frame.loc[d,s] = float(g.econ_volume.sum()) if good else np.nan
            audits.append({'symbol':s,'date':d,'clock':clock,'complete':bool(good),'rows':len(g),
                           'latest_available':g.available_at.max() if len(g) else None})
    return frame,audits


def factors(features, volumes):
    p = factor_frame(features,volumes)
    for at,r in features.iterrows():
        sv=max(float(r.SOXX_vol20),1e-4); qv=max(float(r.QQQ_vol20),1e-4)
        vals=[r['x_'+c]/sv for c in ['session_return','vwap_distance','recent15','previous15',
                                   'range_high_distance','range_low_distance']]
        vals += [r['q_'+c]/qv for c in ['session_return','recent15']]
        vals += [r.breadth_session_return_mean/sv,r.breadth_vwap_distance_mean/sv,
                 2*r.breadth_recent15_positive_fraction-1,r.x_efficiency]
        p.loc[at,EXTRA]=np.clip(vals,-5,5)
        if r.clock!=CLOCKS[0]:
            p.loc[at,'feature_ready'] = bool(p.loc[at,'feature_ready'] and r.regular_context_ready==1 and
                                           r.breadth_regular_ready==1 and np.isfinite(vals).all())
    return p


def columns(clock,variant):
    return BASE_COLUMNS+REGIME_COLUMNS+(EXTRA if variant=='context' and clock!=CLOCKS[0] else [])


def forecasts(panel,at,clock,variant,histories):
    dates,feedback=training_dates(panel,day_cutoff(at)); models={}; record={'feedback':feedback,'fits':{},'forecasts':{}}
    for mode in ['early','full']:
        model,info=fit_clock(panel,dates[mode],columns(clock,variant))
        raw=predict(model,panel.loc[[at]])
        cal=paired_calibration(histories[mode],panel,dates[mode]); event=decision(raw,cal)
        if mode=='early' and feedback['expired_unresolved_prefix']:
            event.update(ready=False,action='CASH',adjusted=None,top_symbol=None)
        models[mode]=model; record['fits'][mode]=info
        record['forecasts'][mode]={'calibration':cal,'event':event}
    raw_event=decision(record['forecasts']['early']['event']['raw'],record['forecasts']['early']['calibration'],'raw')
    if feedback['expired_unresolved_prefix']:
        raw_event.update(ready=False,action='CASH',adjusted=None,top_symbol=None)
    record['forecasts']['raw']={'event':raw_event}
    return models,record


def route_events(at,date,clock,capital,forecasts_by_variant):
    cash={'at':at,'date':date,'ready':True,'action':'CASH','allocation':0.}
    selected={}
    for name in ROUTES:
        variant='base' if name=='base_sequential' else 'context'
        mode='raw' if name=='context_raw' else 'full' if name=='context_full_window' else 'early'
        enabled = not (name=='cash' or name=='context_after_open' and clock==CLOCKS[0] or
                       name=='premarket_only' and clock!=CLOCKS[0] or
                       name.startswith('context_1') and clock!={'context_1005':CLOCKS[1],
                       'context_1035':CLOCKS[2],'context_1305':CLOCKS[3]}[name])
        e=capital[variant][mode]['event'] if enabled else cash
        if name=='context_no_risk':
            f=forecasts_by_variant['context']['early']['event']; take=f['ready'] and f['action']!='CASH'
            e={**cash,'ready':f['ready'],'action':f['top_symbol'] if take else 'CASH','allocation':float(take)}
        selected[name]=e
    selected['double_cost']=selected['context_sequential']
    return selected


def fit_clock(panel, dates, cols, minimum=120):
    kind = 'regret'; rows = panel[panel.date.isin(dates)]
    w = date_weights(len(dates), True)
    info = {'dates': dates, 'members': list(rows.index), 'columns': cols, 'ready': len(dates)>=minimum,
        'parent_weights': w.tolist(), 'weight_sum': float(w.sum()),
        'parent_effective_n': float(w.sum()**2/(w@w)) if len(w) else 0.}
    assert len(rows) == len(dates) and rows.feature_ready.all()
    if not info['ready']: return None, info
    X = rows[cols].to_numpy(); C = rows.C.to_numpy(); D = rows.D.to_numpy(); M = np.abs(D)
    assert np.isfinite(X).all() and np.isfinite(C).all() and np.isfinite(D).all()
    scaler = StandardScaler().fit(X, sample_weight=w); Z = scaler.transform(X)
    model = {'kind': kind, 'columns': cols, 'scaler': scaler, 'heads': {},
             'means': {'C': float(np.average(C, weights=w)), 'D': float(np.average(D, weights=w))}}
    cw = w*M if kind=='regret' else w.copy()
    if cw.sum() > 0: cw = cw*len(w)/cw.sum()
    fallback_q = float(np.average(D>0, weights=cw)) if cw.sum()>0 else .5
    model['q'] = fallback_q
    if kind in ['regret', 'sign']:
        for target, y in [('C', C), ('M', M)]:
            model['heads'][target] = Ridge(alpha=100., solver='svd').fit(Z, y, sample_weight=w)
        if len(np.unique((D[cw>0]>0).astype(int))) == 2:
            classifier = LogisticRegression(C=.1, solver='lbfgs', max_iter=2000, tol=1e-10)
            classifier.fit(Z, (D>0).astype(int), sample_weight=cw)
            assert classifier.n_iter_.max() < 2000
            model['heads']['q'] = classifier
    elif kind=='ridge':
        for target, y in [('C', C), ('D', D)]:
            model['heads'][target] = Ridge(alpha=100., solver='svd').fit(Z, y, sample_weight=w)
    info.update(classification_weights=cw.tolist(), fallback_q=fallback_q,
                center=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), means=model['means'],
                heads={k: {'coefficients': h.coef_.tolist(), 'intercept': np.asarray(h.intercept_).tolist()}
                       for k,h in model['heads'].items()})
    return model, info

