"""Separate calendar/context construction and balanced logistic gradient oracle."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize,root
from scipy.special import expit


def windows(panel,sessions,calendar,history,at,clocks):
    date=at.tz_convert('America/New_York').date()
    days=[d for d in sessions if d.date()<date];out={}
    for mode,selected in [('recent',days[-20:]),('lagged',days[-40:-20])]:
        parents=[];ready=len(selected)==20
        for d in selected:
            members=[pd.Timestamp(str(d.date())+' '+c,tz='America/New_York').tz_convert('UTC') for c in clocks]
            members=[t for t in members if t<pd.Timestamp(calendar[d]['close_at'])]
            okay=bool(members) and all(t in panel.index and bool(panel.loc[t,'feature_ready']) and t in history and history[t]['events']['gap_blend']['ready'] for t in members)
            ready=ready and okay;parents.append({'date':d,'members':members})
        out[mode]={'ready':ready,'dates':selected,'parents':parents if ready else []}
    return out


def contexts(panel,parents,history,fields):
    rows=[];members=[];weights=[]
    # Independent original coordinate mapping, not production signed_price.
    keep=[0,1,2,3,4,5,6,7,8,9,10,16,17];direction={0,1,2,5,6,7,8,9,10}
    for p in parents:
        for t in p['members']:
            values=history[t]['estimates']['gap_blend']['values'];side=0 if values[0]>=values[1] else 1
            raw=panel.loc[t,fields].to_numpy(float);assert np.isfinite(raw).all()
            rows.append([raw[k]*(-1 if side==1 and k in direction else 1) for k in keep]+[float(side)])
            members.append([t,side]);weights.append(1/(len(parents)*len(p['members'])))
    return np.array(rows).reshape(-1,14),members,np.array(weights)


def design(X,Z,u,v):
    a=np.asarray(u)/sum(u);b=np.asarray(v)/sum(v)
    center=np.array([sum(a*X[:,j]) for j in range(14)])
    scale=np.array([np.sqrt(sum(a*(X[:,j]-center[j])**2)) for j in range(14)]);scale[scale<1e-12]=1
    A=np.c_[np.ones(len(X)+len(Z)),(np.vstack([X,Z])-center)/scale]
    return A,np.r_[np.zeros(len(X)),np.ones(len(Z))],np.r_[20*a,20*b],center,scale


def objective(beta,A,y,w):
    z=np.sum(A*beta,axis=1);prob=expit(z)
    value=float(sum(w*(np.logaddexp(0,-z)+(1-y)*z))+5*sum(beta*beta))
    gradient=np.array([sum(w*(prob-y)*A[:,j])+10*beta[j] for j in range(A.shape[1])])
    return value,gradient


def solve(A,y,w):
    r=minimize(objective,np.zeros(A.shape[1]),args=(A,y,w),jac=True,method='L-BFGS-B',
        options={'maxiter':2000,'gtol':1e-10,'ftol':1e-15,'maxls':50})
    assert max(abs(objective(r.x,A,y,w)[1]))<1e-6
    return r.x


def solve_calibrator(P,y,w):
    from reference_tilt import objective as class_objective
    r=minimize(class_objective,np.zeros(4),args=(P,y,w),jac=True,method='L-BFGS-B',
        options={'maxiter':2000,'gtol':1e-10,'ftol':1e-15,'maxls':50})
    initial=float(max(abs(class_objective(r.x,P,y,w)[1])))
    # Refine floating-point objective-stagnation using a separate numerical-Jacobian root solver.
    refined=root(lambda b:class_objective(b,P,y,w)[1],r.x,method='hybr',options={'xtol':1e-10})
    norm=float(max(abs(class_objective(refined.x,P,y,w)[1])))
    assert norm<1e-7
    return refined.x,{'lbfgs_gradient_inf':initial,'refined_gradient_inf':norm}


def weights(original,ratio,members):
    raw=np.array([u*v for u,v in zip(original,ratio)]);out=raw*sum(original)/sum(raw);totals={}
    for i,(t,_) in enumerate(members):
        d=pd.Timestamp(t).tz_convert('America/New_York').strftime('%Y-%m-%d');totals[d]=totals.get(d,0.)+out[i]
    values=list(totals.values());ess=sum(values)**2/sum(v*v for v in values)
    return out,ess
