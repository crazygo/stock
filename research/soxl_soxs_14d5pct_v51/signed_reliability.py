"""Signed scalar logit reliability, with exact earlier nonnegative control."""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v50'))
import confidence_shrinkage_v2 as prior
from confidence_shrinkage_v2 import (core,np,pd,read,dump,sha,CLOCKS,tag,day_cutoff,registry,pool,
    block_counts,encode,fit_map,forecast,restrict_market,labels,development_gate,decide,
    verified_source,option_columns,load_raw,build_features,calendar,payoff)
from scipy.special import expit,logit

SOURCE=Path(__file__).resolve().parent
VARIANTS=['signed','nonnegative','inverse','neutral','baseline']
CORRECTIONS=VARIANTS[:2]
ROUTES=['signed_band','signed_point','nonnegative_band','inverse_band','neutral_band',
        'direction_only_band','baseline_band','premarket_only','cash','double_cost']
EPS=1e-6


def correction_columns(kind):return []


def fit_shrinkage(X,q,D,time_weights,n_dates,kind):
    if kind=='nonnegative':return prior.fit_shrinkage(X,q,D,time_weights,n_dates,'temperature')
    assert kind=='signed'
    X=np.asarray(X,float);q=np.asarray(q,float);D=np.asarray(D,float);w=np.asarray(time_weights,float)
    assert X.shape==(len(q),1) and len(q)==len(D)==len(w) and n_dates>0
    assert np.isfinite(X).all() and np.isfinite(q).all() and np.isfinite(D).all() and np.isfinite(w).all()
    assert ((q>=0)&(q<=1)).all() and (w>=0).all() and w.sum()>0
    assert np.array_equal(X[:,0],X[:,0].astype(int)) and ((X[:,0]>=0)&(X[:,0]<4)).all()
    w=w*n_dates/w.sum();cw=w*abs(D);mass=float(cw.sum())
    if mass>0:cw*=n_dates/mass
    x=logit(np.clip(q,EPS,1-EPS));y=(D>0).astype(float)
    gradient=lambda s:float(cw@((expit(s*x)-y)*x)+s-1.)
    lo,hi=-1.,1.;steps=0
    if gradient(lo)>=0:s=lo
    elif gradient(hi)<=0:s=hi
    else:
        for _ in range(80):
            mid=(lo+hi)/2;steps+=1
            if gradient(mid)>0:hi=mid
            else:lo=mid
            if hi-lo<=1e-14:break
        s=(lo+hi)/2
    g=gradient(s);pg=min(g,0.) if s==-1 else max(g,0.) if s==1 else g
    if abs(pg)>1e-8*max(1,n_dates):raise ArithmeticError('Signed scalar optimality check failed')
    loss=float(cw@(np.logaddexp(0,s*x)-y*s*x)+.5*(s-1)**2)
    return {'kind':'signed','center':[],'scale':[],'coefficients':[s],'penalty':[1.],'prior':[1.],
        'bounds':[-1.,1.],'objective':loss,'bisection_steps':steps,'gradient':g,
        'projected_gradient_max':abs(pg),'no_direction_mass':mass==0.,'n_dates':int(n_dates)}


def correct_q(model,x,q):
    if model['kind']=='temperature':return prior.correct_q(model,x,q)
    assert model['kind']=='signed'
    if model['no_direction_mass']:return float(q)
    return float(expit(model['coefficients'][0]*logit(np.clip(q,EPS,1-EPS))))


def q_inputs(panel,records,ids,kind):
    cols=correction_columns(kind)
    X=np.array([[CLOCKS.index(panel.loc[t,'clock']),*panel.loc[t,cols].to_numpy(float)] for t in ids],float)
    q=np.array([records[t]['fits']['full']['raw']['q'] for t in ids],float)
    return X,q


def fit_day(panel,records,histories,reg,at):
    common={t:histories['baseline'][t] if all(histories[k][t] is not None for k in VARIANTS)
            else None for t in sorted(histories['baseline'])}
    parents,fb=pool(reg,common,at);ids=[t for p in parents for t in p['members']];n=len(parents)
    out={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'parents':parents,'members':ids,
         'feedback':fb,'ready':n>=60,'variants':{},'offsets':{}}
    if n<60:return out
    dw=2.**(-np.arange(n)[::-1]/63.);dw*=n/dw.sum()
    groups=np.repeat(np.arange(n),[len(p['members']) for p in parents])
    w=np.concatenate([np.repeat(dw[j]/len(p['members']),len(p['members'])) for j,p in enumerate(parents)])
    truth=panel.loc[ids,['C','D']].to_numpy(float);Ytrue=np.column_stack([truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]])
    counts,starts,seed=block_counts(n,at,10)
    out.update(day_weights=dw,row_weights=w,row_groups=groups,block_starts=starts,seed=seed)
    for kind in CORRECTIONS:
        X,q=q_inputs(panel,records,ids,kind)
        out['offsets'][kind]={'columns':correction_columns(kind),'point_model':fit_shrinkage(X,q,truth[:,1],w,n,kind),
            'bootstrap_models':[fit_shrinkage(X,q,truth[:,1],w*c[groups],n,kind) for c in counts]}
    for kind in VARIANTS:
        X=encode(panel,histories[kind],ids);Y=Ytrue-X[:,:2]
        out['variants'][kind]={'point_model':fit_map(X,Y,w,'partial',n),
            'bootstrap_models':[fit_map(X,Y,w*c[groups],'partial',n) for c in counts]}
    return out


def payoff(source,q):
    c,m=float(source['C']),float(source['M']);d=m*(2*q-1)
    return {'C':c,'M':m,'D':d,'q':float(q),'SOXL':c+d,'SOXS':c-d}


def issue(panel,records,at,daily):
    source=records[at]['fits']['full']['raw'];raws={};details={};estimates={}
    direct={'ready':False,'point':None,'lower':None}
    for kind in VARIANTS:
        raw=None;values={};x=None
        if source and panel.loc[at,'feature_ready']:
            values=source.copy()
            if kind=='inverse':values=payoff(source,1-float(source['q']))
            elif kind=='neutral':values=payoff(source,.5)
            elif kind in CORRECTIONS and daily['ready']:
                X,q=q_inputs(panel,records,[at],kind);x=X[0]
                values=payoff(source,correct_q(daily['offsets'][kind]['point_model'],x,q[0]))
            raw=np.array([values[s] for s in core.ETFS])
        ready=raw is not None and daily['ready']
        info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            models=daily['variants'][kind];row=np.array([*raw,CLOCKS.index(panel.loc[at,'clock'])])
            point=raw+forecast(models['point_model'],row)[0]
            draws=[];raw_draws=[]
            for b,cal in enumerate(models['bootstrap_models']):
                raw_b=raw
                if kind in CORRECTIONS:
                    q_b=correct_q(daily['offsets'][kind]['bootstrap_models'][b],x,source['q'])
                    v=payoff(source,q_b);raw_b=np.array([v[s] for s in core.ETFS])
                row_b=np.array([*raw_b,row[2]])
                raw_draws.append(raw_b);draws.append(raw_b+forecast(cal,row_b)[0])
            draws=np.array(draws)
            radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
            if kind=='signed':
                rd=np.array(raw_draws)
                r=max(0.,float(np.quantile(np.max(rd-raw,axis=1),.9,method='higher')))
                direct={'ready':True,'point':raw.tolist(),'lower':(raw-r).tolist(),'radius':r,'bootstrap_predictions':rd.tolist()}
        raws[kind]=raw;details[kind]={'raw':values};estimates[kind]=info
    return raws,details,estimates,direct


def events(at,date,clock,estimates,direct,feedback):
    primary=decide(estimates['signed'],feedback,True)
    cash={'ready':True,'action':'CASH','allocation':0.}
    out={'signed_band':primary,'signed_point':decide(estimates['signed'],feedback,False),
         'nonnegative_band':decide(estimates['nonnegative'],feedback,True),
         'inverse_band':decide(estimates['inverse'],feedback,True),
         'neutral_band':decide(estimates['neutral'],feedback,True),
         'direction_only_band':decide(direct,feedback,True),
         'baseline_band':decide(estimates['baseline'],feedback,True),
         'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in out.items()}


def verified_v50(p):
    root=core.ROOT/p['matched_control_source'];manifest=read(core.ROOT/'research/soxl_soxs_14d5pct_v50/evidence/delivery_manifest.json')['runs'][root.name]
    for field,folder in [('top_level_sha256',''),('training_records_sha256','training'),('daily_records_sha256','daily')]:
        for f,h in manifest[field].items():assert sha(root/folder/f)==h
    v=read(root/'verification.json');assert v['passed'] and v['results_sha256']==sha(root/'results.json')
    return root
