"""Three bounded reliability anchors with a nested linear-state penalty."""
from pathlib import Path
import sys,itertools
sys.path.append(str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v52'))
import state_reliability as prior
from state_reliability import (core,np,pd,read,dump,sha,CLOCKS,tag,day_cutoff,registry,pool,
    block_counts,encode,fit_map,forecast,restrict_market,labels,development_gate,decide,
    verified_source,option_columns,load_raw,build_features,calendar,add_state,STATE_INPUTS)
from scipy.special import expit,logit

SOURCE=Path(__file__).resolve().parent
VARIANTS=['curve','linear','global','neutral','baseline']
CORRECTIONS=VARIANTS[:3]
ROUTES=['curve_band','curve_point','linear_band','global_band','neutral_band',
        'direction_only_band','baseline_band','premarket_only','cash','double_cost']
EPS=1e-6
TRANSFORM=np.array([[.5,0.,.5],[-.5,0.,.5],[-.5,1.,-.5]])
PENALTY=TRANSFORM.T@TRANSFORM


def curve_basis(u):
    u=np.asarray(u,float);assert np.isfinite(u).all() and ((u>=0)&(u<=1)).all()
    z=2*u-1
    return np.column_stack([np.maximum(-z,0),1-abs(z),np.maximum(z,0)])


def correction_columns(kind):return ['state_heat'] if kind in ['curve','linear'] else []


def box_step(H,g,b):
    """Exact box-constrained quadratic step by all faces (27 in primary)."""
    best=None;value=float('inf');lo=-1.-b;hi=1.-b;n=len(b)
    for status in itertools.product([-1,0,1],repeat=n):
        fixed=np.array([i for i,v in enumerate(status) if v],int)
        free=np.array([i for i,v in enumerate(status) if not v],int);step=np.zeros(n)
        for i in fixed:step[i]=lo[i] if status[i]<0 else hi[i]
        if len(free):step[free]=np.linalg.solve(H[np.ix_(free,free)],-g[free]-H[np.ix_(free,fixed)]@step[fixed])
        if np.any(step<lo-1e-12) or np.any(step>hi+1e-12):continue
        step=np.clip(step,lo,hi);cost=float(g@step+.5*step@H@step)
        if cost<value:best=step;value=cost
    assert best is not None
    return best


def fit_shrinkage(X,q,D,time_weights,n_dates,kind):
    if kind in ['linear','global']:return prior.fit_shrinkage(X,q,D,time_weights,n_dates,'state' if kind=='linear' else 'global')
    assert kind=='curve'
    X=np.asarray(X,float);q=np.asarray(q,float);D=np.asarray(D,float);w=np.asarray(time_weights,float)
    assert X.shape==(len(q),2) and len(q)==len(D)==len(w) and n_dates>0
    assert np.isfinite(X).all() and np.isfinite(q).all() and np.isfinite(D).all() and np.isfinite(w).all()
    assert ((q>=0)&(q<=1)).all() and (w>=0).all() and w.sum()>0
    assert np.array_equal(X[:,0],X[:,0].astype(int)) and ((X[:,0]>=0)&(X[:,0]<4)).all()
    w=w*n_dates/w.sum();cw=w*abs(D);mass=float(cw.sum())
    if mass>0:cw*=n_dates/mass
    x=logit(np.clip(q,EPS,1-EPS));y=(D>0).astype(float);A=x[:,None]*curve_basis(X[:,1]);center=np.ones(3)
    def objective(b):
        eta=A@b;delta=b-center
        return float(cw@(np.logaddexp(0,eta)-y*eta)+.5*delta@PENALTY@delta),A.T@(cw*(expit(eta)-y))+PENALTY@delta
    def projected(b,g):return np.where(b==-1,np.minimum(g,0),np.where(b==1,np.maximum(g,0),g))
    b=center.copy();steps=0
    for _ in range(100):
        loss,g=objective(b)
        if max(abs(projected(b,g)))<=1e-10*max(1,n_dates):break
        prob=expit(A@b);H=A.T@((cw*prob*(1-prob))[:,None]*A)+PENALTY
        step=box_step(H,g,b);rate=1.
        for _ in range(50):
            candidate=np.clip(b+rate*step,-1,1);value,_=objective(candidate)
            if value<=loss+1e-4*float(g@(candidate-b))+1e-12*max(1,n_dates):break
            rate*=.5
        else:raise ArithmeticError('Curve calibration line search failed')
        b=candidate;steps+=1
    else:raise ArithmeticError('Curve calibration did not converge')
    value,g=objective(b);pg=projected(b,g)
    if max(abs(pg))>1e-8*max(1,n_dates):raise ArithmeticError('Curve calibration KKT failed')
    return {'kind':kind,'coefficients':b.tolist(),'penalty_matrix':PENALTY.tolist(),'prior':center.tolist(),
        'knots':[0.,.5,1.],'bounds':[[-1.,1.]]*3,'objective':value,'iterations':steps,'gradient':g.tolist(),
        'projected_gradient_max':float(max(abs(pg))),'no_direction_mass':mass==0.,'n_dates':int(n_dates)}


def correct_q(model,x,q):
    if model['kind']!='curve':return prior.correct_q(model,x,q)
    if model['no_direction_mass']:return float(q)
    slope=float(curve_basis([x[1]])[0]@model['coefficients'])
    return float(expit(slope*logit(np.clip(q,EPS,1-EPS))))


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
            if kind=='neutral':values=payoff(source,.5)
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
            if kind=='curve':
                rd=np.array(raw_draws)
                r=max(0.,float(np.quantile(np.max(rd-raw,axis=1),.9,method='higher')))
                direct={'ready':True,'point':raw.tolist(),'lower':(raw-r).tolist(),'radius':r,'bootstrap_predictions':rd.tolist()}
        raws[kind]=raw;details[kind]={'raw':values};estimates[kind]=info
    return raws,details,estimates,direct


def events(at,date,clock,estimates,direct,feedback):
    primary=decide(estimates['curve'],feedback,True)
    cash={'ready':True,'action':'CASH','allocation':0.}
    out={'curve_band':primary,'curve_point':decide(estimates['curve'],feedback,False),
         'linear_band':decide(estimates['linear'],feedback,True),
         'global_band':decide(estimates['global'],feedback,True),
         'neutral_band':decide(estimates['neutral'],feedback,True),
         'direction_only_band':decide(direct,feedback,True),
         'baseline_band':decide(estimates['baseline'],feedback,True),
         'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in out.items()}


def verified_v52(p):
    root=core.ROOT/p['matched_control_source'];manifest=read(core.ROOT/'research/soxl_soxs_14d5pct_v52/evidence/delivery_manifest.json')['runs'][root.name]
    for field,folder in [('top_level_sha256',''),('training_records_sha256','training'),('daily_records_sha256','daily')]:
        for f,h in manifest[field].items():assert sha(root/folder/f)==h
    v=read(root/'verification.json');assert v['passed'] and v['results_sha256']==sha(root/'results.json')
    return root
