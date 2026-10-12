"""Preserve an as-issued log-payoff mean while reweighting empirical tail scenarios."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v15'))
from allocation import (core,bank,prior_pool,weights,utility,weighted_tail,FRACTIONS,COLUMNS,
                        day_cutoff,restrict_market,development_gate)

ROUTES={
    'anchor_local_power':('local',5.,FRACTIONS,True,False),
    'anchor_global_power':('global',5.,FRACTIONS,True,False),
    'anchor_score_power':('score',5.,FRACTIONS,True,False),
    'anchor_local_log':('local',1.,FRACTIONS,True,False),
    'anchor_local_full':('local',5.,np.array([0.,1.]),True,False),
    'unanchored_local_power':('local',5.,FRACTIONS,False,False),
    'anchor_source_gate':('local',5.,FRACTIONS,True,True),
}


def mean_anchor(returns,prior,target):
    """Minimum relative entropy q_i ∝ p_i exp(lambda*log(1+r_i))."""
    r=np.asarray(returns,dtype=float);p=np.asarray(prior,dtype=float)
    if not np.isfinite(r).all() or not np.isfinite(p).all() or not np.isfinite(target) or (r<=-1).any() or (p<0).any() or p.sum()<=0:
        raise ValueError('Invalid empirical scenarios or mean')
    p=p/p.sum();x=np.log1p(r);positive=p>0;lo=float(x[positive].min());hi=float(x[positive].max())
    old_mean=float(p@x)
    info={'valid':False,'reason':'outside_empirical_support','target_log_mean':float(target),'prior_log_mean':old_mean,
          'support_min':lo,'support_max':hi,'lambda':None,'mean_error':None,'kl_divergence':None}
    if hi-lo<=1e-12:
        if abs(target-old_mean)>1e-12:return None,info
        info.update(valid=True,reason='constant_matching_mean')
        info['lambda']=0.;info['mean_error']=old_mean-target;info['kl_divergence']=0.
        return p,info
    if not lo+1e-12<target<hi-1e-12:return None,info
    logp=np.log(p[positive]);active=x[positive]
    def at_lambda(value):
        z=logp+value*(active-target);z-=z.max();q=np.exp(z);q/=q.sum()
        return q,float(q@active)
    left,right=-1.,1.
    for _ in range(64):
        if at_lambda(left)[1]<=target:break
        left*=2
    else:raise ArithmeticError('Could not bracket lower mean')
    for _ in range(64):
        if at_lambda(right)[1]>=target:break
        right*=2
    else:raise ArithmeticError('Could not bracket upper mean')
    for _ in range(100):
        middle=(left+right)/2;q,mean=at_lambda(middle)
        if mean<target:left=middle
        else:right=middle
    parameter=(left+right)/2;active_q,mean=at_lambda(parameter)
    q=np.zeros(len(p));q[positive]=active_q
    nonzero=q>0;kl=float(np.sum(q[nonzero]*np.log(q[nonzero]/p[nonzero])))
    assert abs(mean-target)<1e-10 and kl>=-1e-12
    info.update(valid=True,reason='mean_matched',mean_error=mean-target,kl_divergence=max(0.,kl))
    info['lambda']=parameter
    return q,info


def choose(pool,current,mode,gamma,fractions,anchor=True,source_gate=False,minimum=60):
    out={'ready':False,'members':[str(at) for at in pool.index],'dates':list(pool.date),
         'allocation':0.,'reason':'insufficient_mature_dates','distribution':None,'utility':None,'anchor':None}
    if len(pool)<minimum:return out
    w,info=weights(pool,current,mode);r=pool.net_return.to_numpy(dtype=float)
    assert np.isfinite(r).all() and (r>-1).all()
    info['prior_weights']=w.tolist();info['prior_effective_dates']=info['effective_dates']
    out['distribution']=info
    if info['effective_dates']<20:
        out['reason']='insufficient_prior_effective_dates';return out
    if anchor:
        w,anchor_info=mean_anchor(r,w,float(current['score']));out['anchor']=anchor_info
        if w is None:
            out['reason']='mean_outside_empirical_support';return out
    info.update(weights=w.tolist(),effective_dates=float(1/(w@w)),mean_return=float(w@r),mean_log_return=float(w@np.log1p(r)),
        second_moment=float(w@(r*r)),loss_frequency=float(w@(r<0)),lower_5pct_mean_return=weighted_tail(r,w),returns=r.tolist())
    if info['effective_dates']<20:
        out['reason']='insufficient_posterior_effective_dates';return out
    scores=utility(r,fractions,gamma)@w;best=int(np.flatnonzero(scores>=scores.max()-1e-12)[0]);fraction=float(fractions[best])
    reason='positive_utility' if fraction else 'cash_best'
    if source_gate and current['score']<=0:fraction=0.;reason='source_value_gate'
    out.update(ready=True,allocation=fraction,reason=reason,
        utility={'gamma':gamma,'fractions':fractions.tolist(),'scores':scores.tolist(),'selected_index_before_source_gate':best})
    return out
