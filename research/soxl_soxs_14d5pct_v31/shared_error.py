"""Day-clustered cross-clock error learning with causal varying coefficients."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v27'))
from clock_reliability import *
from reliability import STATE_COLUMNS,WeightedScale,Ridge,block_counts,decide

VARIANTS=['interaction','additive','shared']
ROUTES=['interaction_band','interaction_point','additive_band','shared_band','premarket_only','source_v27','cash','double_cost']


def registry(panel,market):
    out=[]
    for day,g in panel.groupby('date',sort=True):
        expected=[c for c in CLOCKS if pd.Timestamp(str(day.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[day]['close_at'])]
        ready=list(g.clock)==expected and g.feature_ready.all()
        release=g.terminal_release.max() if ready and g.terminal_release.notna().all() else None
        out.append({'date':day,'members':list(g.index),'feature_ready':bool(ready),'release':release})
    return out


def pool(reg,history,at):
    cutoff=day_cutoff(at);day=cutoff.tz_convert(core.ET).tz_localize(None).normalize()
    prior=[r for r in reg if r['date']<day and r['feature_ready']];prefix=[]
    for r in prior:
        if r['release'] is None or r['release']>=cutoff:break
        prefix.append(r)
    pending=prior[len(prefix)] if len(prefix)<len(prior) else None
    mature=[r for r in prefix if all(t in history and history[t] is not None for t in r['members'])][-120:]
    return mature,{'prior_dates':len(prior),'selected_early_dates':len(prefix),
        'first_pending_date':pending['date'] if pending else None,
        'expired_unresolved_prefix':bool(pending and pending['date']+pd.Timedelta(days=14)<=day),
        'eligible_forecast_dates':len(mature),'candidate_rows':sum(len(r['members']) for r in mature)}


def encode(panel,history,ids):
    raw=np.asarray([history[t] for t in ids],float);state=panel.loc[ids,STATE_COLUMNS].to_numpy(float)
    clock=np.asarray([[float(panel.loc[t,'clock']==c) for c in CLOCKS[1:]] for t in ids])
    return np.column_stack((raw,state,clock))


def expand(Z,kind):
    raw=Z[:,:2];state=Z[:,2:8];clocks=Z[:,8:]
    pieces=[raw,clocks]
    if kind!='shared':pieces.append(state)
    if kind=='interaction':pieces.append((raw[:,:,None]*state[:,None,:]).reshape(len(Z),12))
    return np.column_stack(pieces)


def fit_map(X,Y,w,kind,n_dates):
    w=np.asarray(w,float);w=w*n_dates/w.sum();keep=w>0
    # Delete zero-weight rows before arithmetic so unseen/resampled-out data cannot affect the fit.
    X=X[keep];Y=Y[keep];w=w[keep]
    s=WeightedScale(X,w);Z=s.transform(X);A=expand(Z,kind);v=WeightedScale(A,w)
    h=Ridge(alpha=20.,solver='svd').fit(v.transform(A),Y,sample_weight=w)
    return {'kind':kind,'input_center':s.mean_.tolist(),'input_scale':s.scale_.tolist(),
        'basis_center':v.mean_.tolist(),'basis_scale':v.scale_.tolist(),'coef':h.coef_.tolist(),'intercept':h.intercept_.tolist()}


def forecast(model,x):
    z=(np.atleast_2d(x)-model['input_center'])/model['input_scale'];a=expand(z,model['kind'])
    return ((a-model['basis_center'])/model['basis_scale'])@np.asarray(model['coef']).T+model['intercept']


def fit_day(panel,history,reg,at):
    parents,fb=pool(reg,history,at);ids=[t for r in parents for t in r['members']];n=len(parents)
    receipt={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'feedback':fb,'parents':parents,'members':ids,
        'ready':n>=60,'variants':{}}
    if n<60:return receipt
    dayweights=2.**(-np.arange(n)[::-1]/63.);dayweights*=n/dayweights.sum()
    groups=np.repeat(np.arange(n),[len(r['members']) for r in parents]);w=np.concatenate([np.repeat(dayweights[j]/len(r['members']),len(r['members'])) for j,r in enumerate(parents)])
    X=encode(panel,history,ids);raw=X[:,:2];truth=panel.loc[ids,['C','D']].to_numpy(float)
    Y=np.column_stack((truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]))-raw
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    counts,starts,seed=block_counts(n,at,10)
    receipt.update(day_weights=dayweights,row_weights=w,row_groups=groups,block_starts=starts,seed=seed)
    for kind in VARIANTS:
        point=fit_map(X,Y,w,kind,n)
        samples=[fit_map(X,Y,w*c[groups],kind,n) for c in counts]
        receipt['variants'][kind]={'point_model':point,'bootstrap_models':samples}
    return receipt


def issue_shared(panel,at,raw,daily):
    ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];out={}
    for kind in VARIANTS:
        info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            x=encode(panel,{at:raw},[at])[0];models=daily['variants'][kind]
            point=raw+forecast(models['point_model'],x)[0]
            samples=np.asarray([raw+forecast(m,x)[0] for m in models['bootstrap_models']])
            radius=max(0.,float(np.quantile(np.max(samples-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=samples.tolist())
        out[kind]=info
    return out


def events(at,date,clock,estimates,feedback,old):
    primary=decide(estimates['interaction'],feedback,True);point=decide(estimates['interaction'],feedback,False)
    cash={'ready':True,'action':'CASH','allocation':0.}
    out={'interaction_band':primary,'interaction_point':point,'additive_band':decide(estimates['additive'],feedback,True),
        'shared_band':decide(estimates['shared'],feedback,True),'premarket_only':primary if clock==CLOCKS[0] else cash,
        'source_v27':old,'cash':cash,'double_cost':primary}
    return {n:{**e,'at':at,'date':date} for n,e in out.items()}
