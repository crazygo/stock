"""Parent-date empirical uncertainty for an existing as-of holding-value model."""
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v10'))
from micro_manager import (core,MANAGER_COLUMNS,context,prepare_reviews,apply_exit,alive,label_key,
                           baseline_labels,joint_releases)


def restrict_market(market,cutoff):
    """Trim the engine's observations before labeling/scoring; retain the known calendar."""
    cutoff = core.timestamp(cutoff);day = cutoff.tz_convert(core.ET).tz_localize(None).normalize()
    bars = {s:f[(f.end_at<cutoff)&(f.available_at<cutoff)].copy() for s,f in market.bars.items()}
    daily = {s:f[(f.index<day)&(f.available_at<cutoff)].copy() for s,f in market.daily.items()}
    complete = {s:{d:bool(row.complete) for d,row in f.iterrows()} for s,f in daily.items()}
    actions = {s:[a for a in rows if pd.Timestamp(a['date'])<day] for s,rows in market.actions.items()}
    return core.Market(bars,daily,market.calendar,market.sessions,actions,complete,market.manifest)


def build_bank(records,releases,columns):
    rows = []
    for r in records:
        state = r['state']
        rows.append({'id':r['id'],'entry_date':r['entry_date'],'at':r['at'],
            'release':releases.get(r['entry_date'],pd.NaT),'target':r['target'],
            **{c:(np.nan if state is None else r['raw'] if c=='raw' else state[c]) for c in columns}})
    frame = pd.DataFrame(rows).set_index('id')
    frame['raw'] = pd.to_numeric(frame.raw,errors='coerce');frame['target'] = pd.to_numeric(frame.target,errors='coerce')
    frame['release'] = pd.to_datetime(frame.release,utc=True)
    return frame


def pool_for_day(bank,cutoff,protocol):
    eligible = []
    for day,group in bank.groupby('entry_date',sort=True):
        if day.tz_localize(core.ET)>=cutoff:continue
        if ((group['at']<cutoff)&(group.release<cutoff)).all() and np.isfinite(group[[*protocol['local_columns'],'target']]).all().all():
            eligible.append(day)
    dates = eligible[-protocol['local_window']:]
    pool = bank[bank.entry_date.isin(dates)].copy()
    recent = dates[-protocol['parent_window']:]
    maxima = []
    for day in recent:
        g = pool[pool.entry_date==day];maxima.append(float((g.target-g.raw).max()))
    ready = len(recent)>=protocol['minimum_parents'];q = None;rank = None
    if ready:
        rank = min(len(recent),math.ceil((len(recent)+1)*protocol['quantile']))
        q = sorted(maxima)[rank-1]
    info = {'cutoff':cutoff,'eligible_parents':dates,'quantile_parents':recent,
            'parent_maxima':maxima,'quantile_rank_one_based':rank,'residual_upper':q,'ready':ready,
            'source_rows':len(pool)}
    local = None
    if len(dates)>=max(protocol['minimum_parents'],protocol['local_neighbors']):
        cols = protocol['local_columns'];means = [];seconds = []
        groups = []
        for day in dates:
            idx = np.flatnonzero(pool.entry_date.to_numpy()==np.datetime64(day))
            x = pool.iloc[idx][cols].to_numpy(dtype=float)
            means.append(x.mean(axis=0));seconds.append((x*x).mean(axis=0));groups.append(idx)
        center = np.mean(means,axis=0);variance = np.maximum(np.mean(seconds,axis=0)-center*center,0.)
        scale = np.sqrt(variance);scale[scale<1e-12]=1.
        x = pool[cols].to_numpy(dtype=float)
        assert np.isfinite(x).all()
        local = {'X':x,'target':pool.target.to_numpy(dtype=float),'ids':pool.index.to_numpy(dtype=str),
                 'groups':groups,'center':center,'scale':scale}
        info.update(local_center=center.tolist(),local_scale=scale.tolist())
    return info,local


def decision(raw,state,pool,local,protocol,method):
    event = {'action':'HOLD','ready':False,'bound':None,'raw':raw}
    if raw is None or state is None or not np.isfinite(raw):return event
    if method==protocol['primary']:
        if pool['ready']:
            event.update(ready=True,bound=float(raw+pool['residual_upper']))
    elif method==protocol['diagnostic']:
        if local is not None:
            q = np.array([raw if c=='raw' else state[c] for c in protocol['local_columns']])
            distance = np.sum(((local['X']-q)/local['scale'])**2,axis=1)
            representatives = np.array([g[np.lexsort((local['ids'][g],distance[g]))[0]] for g in local['groups']])
            order = np.lexsort((local['ids'][representatives],distance[representatives]))[:protocol['local_neighbors']]
            chosen = representatives[order];values = local['target'][chosen]
            mean = float(values.mean());se = float(values.std(ddof=1)/np.sqrt(len(values)))
            event.update(ready=True,bound=mean+protocol['local_z']*se,mean=mean,se=se,
                         neighbors=local['ids'][chosen].tolist(),distances=distance[chosen].tolist())
    else:raise ValueError(method)
    if event['ready'] and event['bound']<0:event['action']='EXIT'
    return event


def development_gate(stats,coverage,protocol):
    wanted = protocol['development']['months'];months = stats['monthly']
    rates = [m['return'] for m in months if m['month'] in wanted]
    complete = (len(months)==len(wanted) and {m['month'] for m in months}==set(wanted)
                and all(m['mark_complete'] for m in months) and coverage and np.isfinite(rates).all())
    average = float(np.mean(rates)) if len(rates)==len(wanted) else None
    nonnegative = bool(complete and all(r>=-1e-12 for r in rates))
    return {'passed':bool(complete and average>=.10 and nonnegative and stats['completed_trades']>=10),
        'four_month_arithmetic_mean':average,'all_months_nonnegative':nonnegative,'complete_coverage':bool(complete),
        'completed_actual_trades':stats['completed_trades'],'minimum_completed_trades':10,
        'not_original_case_acceptance':True}
