"""Prequential state-conditioned payoff distributions and bounded capital sizing."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v14'))
from paired_entry import core, day_cutoff, restrict_market, development_gate

COLUMNS=['score','advantage','relative_5','breadth_5','volume_direction_diffusion','log_sector_volatility']
FRACTIONS=np.array([0.,.25,.5,.75,1.])
ROUTES={'local_power':('local',5.,FRACTIONS), 'global_power':('global',5.,FRACTIONS),
        'score_power':('score',5.,FRACTIONS), 'local_log':('local',1.,FRACTIONS),
        'local_full':('local',5.,np.array([0.,1.]))}


def bank(panel, events):
    rows=[]
    for e in events:
        if not e['ready']:continue
        at=pd.Timestamp(e['at']);r=panel.loc[at]
        for symbol,sign in [('SOXL',1.),('SOXS',-1.)]:
            rows.append({'at':at,'date':r.date,'symbol':symbol,'release':r.release,
                'net_return':float(np.expm1(r.C+sign*r.D)),
                'score':e['adjusted'][symbol],'advantage':sign*e['adjusted']['D'],
                'relative_5':sign*r.relative_5,'breadth_5':sign*r.breadth_5,
                'volume_direction_diffusion':sign*r.volume_direction_diffusion,
                'log_sector_volatility':r.log_sector_volatility})
    return pd.DataFrame(rows).set_index(['at','symbol']).sort_index()


def prior_pool(frame, cutoff, symbol):
    rows=frame.xs(symbol,level='symbol')
    rows=rows[(rows.release<cutoff)&(rows.index<cutoff)&(rows.date<cutoff.tz_convert(core.ET).tz_localize(None).normalize())]
    rows=rows[np.isfinite(rows.net_return)]
    return rows.tail(120)


def weights(pool, current, mode):
    n=len(pool);base=2.**(-np.arange(n)[::-1]/63.)
    base/=base.sum()
    columns=COLUMNS if mode=='local' else COLUMNS[:2] if mode=='score' else []
    info={'mode':mode,'columns':columns,'base_weights':base.tolist(),'center':None,'scale':None,
          'squared_distances':None,'bandwidth_squared':None}
    w=base.copy()
    if columns:
        X=pool[columns].to_numpy(dtype=float);q=current[columns].to_numpy(dtype=float)
        center=np.average(X,axis=0,weights=base);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=base))
        scale[scale<1e-12]=1.
        distance=np.mean(((X-q)/scale)**2,axis=1)
        bandwidth=max(float(np.sort(distance)[min(40,n)-1]),1e-12)
        w*=np.exp(-.5*distance/bandwidth)
        info.update(center=center.tolist(),scale=scale.tolist(),squared_distances=distance.tolist(),bandwidth_squared=bandwidth)
    assert np.isfinite(w).all() and w.sum()>0
    w/=w.sum();info.update(weights=w.tolist(),effective_dates=float(1/(w@w)))
    return w,info


def utility(returns, fractions, gamma):
    wealth=1+np.outer(np.asarray(fractions),np.asarray(returns))
    if not np.isfinite(wealth).all() or (wealth<=0).any():raise ValueError('Nonpositive or invalid scenario wealth')
    return np.log(wealth) if gamma==1 else np.expm1((1-gamma)*np.log(wealth))/(1-gamma)


def weighted_tail(returns, weights, mass=.05):
    order=np.argsort(returns,kind='stable');r=returns[order];w=weights[order]
    used=np.minimum(w,np.maximum(0.,mass-np.r_[0.,np.cumsum(w)[:-1]]))
    return float((r*used).sum()/mass)


def choose(pool,current,mode,gamma,fractions,minimum=60):
    result={'ready':False,'members':[str(at) for at in pool.index],'dates':list(pool.date),
            'allocation':0.,'reason':'insufficient_mature_dates','distribution':None,'utility':None}
    if len(pool)<minimum:return result
    w,info=weights(pool,current,mode);r=pool.net_return.to_numpy(dtype=float)
    assert np.isfinite(r).all() and (r>-1).all()
    info.update(mean_return=float(w@r),second_moment=float(w@(r*r)),loss_frequency=float(w@(r<0)),
        lower_5pct_mean_return=weighted_tail(r,w),returns=r.tolist())
    result['distribution']=info
    if info['effective_dates']<20:
        result['reason']='insufficient_effective_dates';return result
    scores=utility(r,fractions,gamma)@w
    # Fractions are ascending, so exact or numerically negligible ties prefer cash/smaller exposure.
    best=int(np.flatnonzero(scores>=scores.max()-1e-12)[0])
    fraction=float(fractions[best])
    result.update(ready=True,allocation=fraction,reason='positive_utility' if fraction else 'cash_best',
        utility={'gamma':gamma,'fractions':fractions.tolist(),'scores':scores.tolist(),'selected_index':best})
    return result
