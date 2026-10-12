"""As-of selected-trade calibration and learned saved-policy continuation."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

V7 = Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v7'
sys.path.insert(0,str(V7))
from breadth import core
from path_engine import day_cutoff

META = ['top_score','top_margin','component_disagreement','direction','sector_vol20',
        'signed_sector_session_return','signed_sector_vwap_distance','signed_qqq_recent15']


def meta_state(row, ensemble, ridge, boost):
    if not row.regular_context_ready or not bool(ensemble.ready):
        return None
    values = [ensemble[s+'_adjusted'] for s in core.ETFS]
    if not np.isfinite(values).all():
        return None
    j = int(np.argmax(values)); s = core.ETFS[j]; direction = 1. if j==0 else -1.
    state = {'symbol':s, 'top_score':values[j], 'top_margin':values[j]-values[1-j],
        'component_disagreement':abs(ridge[s+'_adjusted']-boost[s+'_adjusted']),
        'direction':direction, 'sector_vol20':row.SOXX_vol20,
        'signed_sector_session_return':direction*row.x_session_return,
        'signed_sector_vwap_distance':direction*row.x_vwap_distance,
        'signed_qqq_recent15':direction*row.q_recent15}
    return state if np.isfinite([state[k] for k in META]).all() else None


def date_releases(features, tables):
    """All observed intraday states and both ETF contracts mature together."""
    releases = {}
    for date, group in features[features.regular_context_ready.eq(1.)].groupby('date'):
        parts = [tables[s].reindex(group.index) for s in core.ETFS]
        if all(t.complete_window.eq(True).all() and t.label_available_at.notna().all()
               and np.isfinite(t.net_return).all() for t in parts):
            releases[date] = max(t.label_available_at.max() for t in parts)
    return releases


def neighbors(pool, query, cutoff, protocol):
    if pool.empty:
        return None, {'eligible':[], 'neighbors':[], 'ready':False}
    # Explicitly exclude current/future decisions even if a malformed caller
    # supplies an early release timestamp. A late prior label stays excluded.
    past = pool[(pool.index<cutoff) & (pool.release<cutoff)].sort_index().tail(protocol['window_dates'])
    info = {'eligible':list(past.index), 'neighbors':[], 'ready':len(past)>=protocol['minimum_dates']}
    if not info['ready']:
        return None, info
    if past.index.tz_convert(core.ET).normalize().duplicated().any():
        raise ValueError('One same-clock observation per historical date required')
    X = past[META].to_numpy(dtype=float)
    q = np.array([query[k] for k in META],dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(q).all() or not np.isfinite(past.buy_return).all():
        raise ValueError('Unknown observations cannot be replaced by zero')
    center = X.mean(axis=0); scale = X.std(axis=0,ddof=0); scale[scale==0]=1.
    distance = np.sum(((X-q)/scale)**2,axis=1)
    order = np.argsort(distance,kind='stable')[:protocol['neighbors']]
    chosen = past.iloc[order]
    info.update({'neighbors':list(chosen.index),'distances':distance[order].tolist(),
                 'center':center.tolist(),'scale':scale.tolist()})
    return chosen,info


def decision(query, pool, at, has_later, settings, protocol):
    event = {'at':at,'action':'WAIT' if has_later else 'CASH','selected_symbol':None,
             'ready':False,'cutoff':day_cutoff(at),'buy_mean':None,'buy_se':None,
             'buy_value':None,'wait_mean':None,'wait_value':None,'reason':'meta_state_unavailable'}
    if query is None:
        return event, {'eligible':[],'neighbors':[],'ready':False}
    event['selected_symbol']=query['symbol']
    chosen,audit=neighbors(pool,query,day_cutoff(at),protocol)
    if chosen is None:
        event['reason']='insufficient_prior_meta_dates'
        return event,audit
    mu=float(chosen.buy_return.mean()); se=float(chosen.buy_return.std(ddof=1)/len(chosen)**.5)
    if settings['wait'] and has_later:
        if not np.isfinite(chosen.wait_return).all():
            event['reason']='unknown_continuation'; return event,audit
        wait=float(chosen.wait_return.mean())
    else:
        wait=0.
    value=mu-settings['penalty']*se; continuation=max(0.,wait)
    event.update({'ready':True,'buy_mean':mu,'buy_se':se,'buy_value':value,
                  'wait_mean':wait,'wait_value':continuation,'reason':'wait_or_cash_preferred'})
    if value>continuation:
        event['action']=query['symbol'];event['reason']='buy_value_exceeds_wait_and_cash'
    return event,audit


def realized_suffix(events, tables, at, expected):
    """Return of first saved later buy, NOT best realized future trade."""
    future = sorted([e for e in events if e['at']>at],key=lambda e:e['at'])
    if {e['at'] for e in future} != {t for t in expected if t>at}:
        return np.nan
    for e in future:
        if e['action'] in core.ETFS:
            r=tables[e['action']].loc[e['at'],'net_return']
            return float(np.log1p(r)) if np.isfinite(r) else np.nan
    return 0.


def append_day(pools, states, events, tables, release, expected):
    """Archive end-of-episode labels; their release still gates later training."""
    for e in events:
        at=e['at'];q=states.get(at)
        if q is None:
            continue
        value=tables[q['symbol']].loc[at,'net_return']
        row={**q,'release':release,'buy_return':float(np.log1p(value)) if np.isfinite(value) else np.nan,
             'wait_return':realized_suffix(events,tables,at,expected)}
        clock=at.tz_convert(core.ET).strftime('%H:%M:%S')
        new=pd.DataFrame([row],index=pd.DatetimeIndex([at]))
        pools[clock]=new if pools[clock].empty else pd.concat([pools[clock],new])
