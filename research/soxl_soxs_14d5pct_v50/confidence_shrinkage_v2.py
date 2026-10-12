"""Bounded as-issued confidence calibration with shared date uncertainty."""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v47'))
from option_learning import (core,np,pd,read,dump,sha,CLOCKS,tag,day_cutoff,registry,pool,
    block_counts,encode,fit_map,forecast,restrict_market,labels,development_gate,decide,
    verified_source,option_columns,load_raw,build_features,calendar)
from scipy.special import expit, logit
from scipy.optimize import minimize

SOURCE=Path(__file__).resolve().parent
VARIANTS=['temperature','affine','options','neutral','baseline']
CORRECTIONS=VARIANTS[:3]
ROUTES=['temperature_band','temperature_point','affine_band','options_band','neutral_band',
        'direction_only_band','baseline_band','premarket_only','cash','double_cost']
EPS=1e-6


def correction_columns(kind):
    return option_columns(2) if kind=='options' else []


def fit_shrinkage(X,q,D,time_weights,n_dates,kind):
    """Strictly convex penalized logistic fit; first coefficient bounded [0,1]."""
    assert kind in CORRECTIONS
    X=np.asarray(X,float);q=np.asarray(q,float);D=np.asarray(D,float);w=np.asarray(time_weights,float)
    assert n_dates>0 and np.isfinite(w).all() and (w>=0).all() and w.sum()>0 and np.isfinite(D).all()
    assert np.isfinite(X).all() and np.isfinite(q).all() and ((q>=0)&(q<=1)).all()
    w=w*n_dates/w.sum();clocks=X[:,0].astype(int)
    assert np.array_equal(clocks,X[:,0]) and ((clocks>=0)&(clocks<4)).all()
    assert X.shape[1]==(9 if kind=='options' else 1)
    center=np.average(X[:,1:],axis=0,weights=w)
    scale=np.sqrt(np.average((X[:,1:]-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
    log_odds=logit(np.clip(q,EPS,1-EPS))
    A=log_odds[:,None] if kind=='temperature' else np.column_stack([log_odds,np.eye(4)[clocks],(X[:,1:]-center)/scale])
    penalty=np.r_[1.,np.ones(4),np.repeat(10.,X.shape[1]-1)] if kind!='temperature' else np.array([1.])
    prior=np.zeros(A.shape[1]);prior[0]=1.
    cw=w*abs(D);mass=float(cw.sum())
    if mass>0:cw*=n_dates/mass
    y=(D>0).astype(float)
    def loss_gradient(b):
        eta=A@b;delta=b-prior
        return float(cw@(np.logaddexp(0.,eta)-y*eta)+.5*(penalty*delta)@delta),A.T@(cw*(expit(eta)-y))+penalty*delta
    result=minimize(loss_gradient,prior,method='L-BFGS-B',jac=True,bounds=[(0.,1.)]+[(None,None)]*(len(prior)-1),
                    options={'ftol':0.,'gtol':1e-10,'maxiter':1000,'maxls':50})
    b=result.x.copy();steps=0;initial_residual=None
    def projected_residual(b,g):
        pg=g.copy()
        if b[0]==0:pg[0]=min(g[0],0.)
        if b[0]==1:pg[0]=max(g[0],0.)
        return float(max(abs(pg)))
    value,g=loss_gradient(b);initial_residual=projected_residual(b,g)
    if initial_residual>1e-8*max(1,n_dates):
        # Polish the SAME strictly convex optimum; do not relax the KKT gate.
        # Solve each local Newton quadratic exactly with its one box constraint.
        for _ in range(25):
            value,g=loss_gradient(b)
            if projected_residual(b,g)<=1e-10*max(1,n_dates):break
            prob=expit(A@b);H=A.T@((cw*prob*(1-prob))[:,None]*A)+np.diag(penalty)
            step=-np.linalg.solve(H,g);lo,hi=-b[0],1-b[0]
            if not lo<=step[0]<=hi:
                step[0]=min(max(step[0],lo),hi)
                if len(b)>1:step[1:]=-np.linalg.solve(H[1:,1:],g[1:]+H[1:,0]*step[0])
            rate=1.
            for _ in range(50):
                candidate=b+rate*step;candidate[0]=min(max(candidate[0],0.),1.)
                candidate_value,_=loss_gradient(candidate)
                if candidate_value<=value+1e-4*float(g@(candidate-b))+1e-12*max(1,n_dates):break
                rate*=.5
            else:raise ArithmeticError('Bounded Newton refinement line search failed')
            b=candidate;steps+=1
        else:raise ArithmeticError('Bounded Newton refinement did not converge')
    value,g=loss_gradient(b);residual=projected_residual(b,g)
    if residual>1e-8*max(1,n_dates):raise ArithmeticError(f'Bounded logistic KKT failed: {residual}')
    model={'kind':kind,'center':center.tolist(),'scale':scale.tolist(),'coefficients':b.tolist(),
            'penalty':penalty.tolist(),'prior':prior.tolist(),'objective':value,'iterations':int(result.nit),
            'solver_success':bool(result.success),'solver_message':str(result.message),
            'projected_gradient_max':residual,'no_direction_mass':mass==0.,'n_dates':int(n_dates)}
    if steps:model['numerical_refinement']={'steps':steps,'initial_projected_gradient_max':initial_residual,'same_objective_and_bounds':True}
    return model


def correct_q(model,x,q):
    if model['no_direction_mass']:return float(q)
    c=int(x[0]);assert c==x[0] and c in range(4)
    b=np.asarray(model['coefficients']);value=b[0]*logit(np.clip(q,EPS,1-EPS))
    if model['kind']!='temperature':
        z=(np.asarray(x[1:],float)-model['center'])/model['scale'];value+=b[1+c]+z@b[5:]
    return float(expit(value))


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
            if kind=='temperature':
                rd=np.array(raw_draws)
                r=max(0.,float(np.quantile(np.max(rd-raw,axis=1),.9,method='higher')))
                direct={'ready':True,'point':raw.tolist(),'lower':(raw-r).tolist(),'radius':r,'bootstrap_predictions':rd.tolist()}
        raws[kind]=raw;details[kind]={'raw':values};estimates[kind]=info
    return raws,details,estimates,direct


def events(at,date,clock,estimates,direct,feedback):
    primary=decide(estimates['temperature'],feedback,True)
    cash={'ready':True,'action':'CASH','allocation':0.}
    out={'temperature_band':primary,'temperature_point':decide(estimates['temperature'],feedback,False),
         'affine_band':decide(estimates['affine'],feedback,True),
         'options_band':decide(estimates['options'],feedback,True),
         'neutral_band':decide(estimates['neutral'],feedback,True),
         'direction_only_band':decide(direct,feedback,True),
         'baseline_band':decide(estimates['baseline'],feedback,True),
         'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in out.items()}
