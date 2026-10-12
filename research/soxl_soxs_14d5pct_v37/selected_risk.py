"""Loss-magnitude-weighted logistic risk filtering for as-issued top choices."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v36'))
from adaptive_correction import (core,np,pd,read,dump,sha,CLOCKS,tag,day_cutoff,
    registry,pool,block_counts,weights,restrict_market,labels,development_gate,load_source)
from scipy.special import expit

SOURCE=Path(__file__).resolve().parent
SCORES=['top_lower','top_gap','raw_direction_agreement','is_soxs']+['clock_'+c for c in CLOCKS[1:]]
CONTEXT=['aligned_relative5','aligned_breadth5','aligned_volume_diffusion','volatility','aligned_growth_acceleration','recent_earnings']
KINDS=['context','scores','frequency']
ROUTES=['context_gate','scores_gate','frequency_gate','context_band','standalone','source_v35','cash','double_cost']


def selected_panel(panel,records):
    p=panel.copy();extra=[]
    for at,row in panel.iterrows():
        r=records[at];v=r['estimates']['full'];out={'top_ready':v['ready'],'top_symbol':None,'selected_y':np.nan}
        out.update({c:np.nan for c in SCORES+CONTEXT})
        if v['ready']:
            j=0 if v['point'][0]>=v['point'][1] else 1;s=1 if j==0 else -1
            out.update(top_symbol=core.ETFS[j],selected_y=row.C+s*row.D,
                top_lower=v['lower'][j],top_gap=v['point'][j]-v['point'][1-j],
                raw_direction_agreement=s*(r['raws']['full'][0]-r['raws']['full'][1]),is_soxs=float(j==1),
                aligned_relative5=s*row.relative_5,aligned_breadth5=s*row.breadth_5,
                aligned_volume_diffusion=s*row.volume_direction_diffusion,volatility=row.log_sector_volatility,
                aligned_growth_acceleration=s*row.growth_acceleration,recent_earnings=row.recent_earnings_share)
            out.update({'clock_'+c:float(row.clock==c) for c in CLOCKS[1:]})
        extra.append(out)
    return p.join(pd.DataFrame(extra,index=p.index))


def columns(kind):
    return SCORES+CONTEXT if kind=='context' else SCORES if kind=='scores' else []


def fit_logit(X,r,w,n):
    """Strictly convex penalized logistic loss; intercept is unpenalized."""
    keep=w>0;X=np.asarray(X,float)[keep];r=np.asarray(r,float)[keep];w=np.asarray(w,float)[keep]
    w=w*n/w.sum();assert np.isfinite(X).all() and np.isfinite(r).all()
    center=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
    m=w*np.abs(r);total=float(m.sum());positive=float(m[r>0].sum());q=positive/total if total else .5
    out={'center':center,'scale':scale,'positive_share':q,'magnitude_total':total,'weight_sum':n,
        'mean_return':float(np.average(r,weights=w)),'raw_win_fraction':float(np.average(r>0,weights=w))}
    if q in [0.,1.] or total<=1e-16:
        out.update(mode='constant',quality=q,coef=np.zeros(X.shape[1]),intercept=0.,iterations=0,gradient_max=None)
        return out
    m*=n/total;A=np.column_stack((np.ones(len(X)),(X-center)/scale));y=(r>0).astype(float)
    penalty=np.r_[0.,np.repeat(10.,X.shape[1])];b=np.r_[np.log(q/(1-q)),np.zeros(X.shape[1])]
    objective=lambda z:float(m@(np.logaddexp(0.,A@z)-y*(A@z))+.5*np.sum(penalty*z*z))
    for iteration in range(100):
        prob=expit(A@b);g=A.T@(m*(prob-y))+penalty*b
        if np.max(np.abs(g))<1e-8:break
        h=A.T@((m*prob*(1-prob))[:,None]*A)+np.diag(penalty);step=np.linalg.solve(h,g)
        old=objective(b);size=1.
        while objective(b-size*step)>old-1e-4*size*float(g@step)+1e-13:
            size*=.5
            if size<2.**-40:raise ArithmeticError('Logistic line search failed')
        b-=size*step
    else:raise ArithmeticError('Logistic fit did not converge')
    out.update(mode='logit',coef=b[1:],intercept=b[0],iterations=iteration,gradient_max=float(np.max(np.abs(g))))
    return out


def quality(model,x):
    if model['mode']=='constant':return float(model['quality'])
    return float(expit(((np.asarray(x)-model['center'])/model['scale'])@model['coef']+model['intercept']))


def fit_day(panel,records,reg,at,source_day=None):
    history={t:r['raws']['full'] if r['estimates']['full']['ready'] else None for t,r in records.items() if t<day_cutoff(at)}
    parents,fb=pool(reg,history,at);ids=[t for p in parents for t in p['members']];n=len(parents)
    out={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'parents':parents,'members':ids,'feedback':fb,
        'ready':n>=60,'models':{},'bootstrap':[]}
    if n<60:return out
    _,w,groups=weights(parents);out.update(row_weights=w,row_groups=groups)
    r=panel.loc[ids,'selected_y'].to_numpy(float);counts,starts,seed=block_counts(n,at,10);out.update(block_starts=starts,seed=seed)
    X={kind:panel.loc[ids,columns(kind)].to_numpy(float) for kind in KINDS}
    out['models']={kind:fit_logit(X[kind],r,w,n) for kind in KINDS}
    out['bootstrap']=[{kind:fit_logit(X[kind],r,w*c[groups],n) for kind in KINDS} for c in counts]
    return out


def issue(panel,at,record,daily):
    source=record['estimates']['full'];ready=bool(source['ready'] and panel.loc[at,'feature_ready'])
    out={k:{'ready':ready,'quality':None,'lower_quality':None,'source_fallback':bool(ready and not daily['ready'])} for k in KINDS}
    if not ready or not daily['ready']:return out
    for kind in KINDS:
        x=panel.loc[at,columns(kind)].to_numpy(float);point=quality(daily['models'][kind],x)
        draws=np.array([quality(z[kind],x) for z in daily['bootstrap']])
        radius=max(0.,float(np.quantile(draws-point,.9,method='higher')))
        out[kind].update(quality=point,lower_quality=point-radius,upward_radius=radius,bootstrap_qualities=draws)
    return out


def events(at,date,estimates,daily,record):
    old=record['events']['full_band'];expired=daily['feedback']['expired_unresolved_prefix']
    def gate(kind,band=False,standalone=False):
        v=estimates[kind];ready=v['ready'] and not expired
        take=False;reason='unready';q=None
        if ready:
            if v['source_fallback']:
                take=old['action'] in core.ETFS;reason='source_warmup'
            else:
                q=v['lower_quality'] if band else v['quality']
                take=q>.5+1e-12 and (standalone or old['action'] in core.ETFS)
                reason='positive_gain_loss_balance' if take else 'rejected'
        values=record['estimates']['full']['point'];j=0 if values is not None and values[0]>=values[1] else 1
        return {'ready':ready,'action':core.ETFS[j] if take else 'CASH','allocation':float(take),'reason':reason,'quality_used':q}
    primary=gate('context')
    actions={'context_gate':primary,'scores_gate':gate('scores'),'frequency_gate':gate('frequency'),
        'context_band':gate('context',band=True),'standalone':gate('context',standalone=True),
        'source_v35':old,'cash':{'ready':True,'action':'CASH','allocation':0.},'double_cost':primary}
    return {k:{**v,'at':at,'date':date} for k,v in actions.items()}
