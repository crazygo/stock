"""Separate population construction, class-tilt derivatives, and anchor optimizer."""
import numpy as np
from scipy.optimize import minimize


def data(panel,parents,history,mode):
    rows=[];day_weights=[];dates=[];N=len(parents)
    for i,parent in enumerate(parents):
        selected=[]
        for at in parent['members']:
            record=history[at];e=record['events']['gap_blend'];info=record['estimates']['gap_blend']
            winner=0 if info['values'][0]>=info['values'][1] else 1
            if mode=='eligible' and e['action']=='CASH':continue
            for side in ([0,1] if mode=='both' else [winner]):
                selected.append((at,side,info['probabilities'][side],int(panel.loc[at,'L_gap' if side==0 else 'S_gap'])))
        if selected:
            b=2**(-(N-1-i)/63);day_weights.append(b);dates.append(parent['date'])
            rows.extend([(*r,b/len(selected)) for r in selected])
    total=sum(day_weights);ess=total*total/sum(v*v for v in day_weights) if total else 0.
    weights=np.array([r[-1]*ess/total for r in rows])
    return np.array([r[2] for r in rows]).reshape(-1,4),np.array([r[3] for r in rows],int),weights,{
        'dates':dates,'real_dates':len(dates),'date_ess':ess,'members':[[r[0],r[1]] for r in rows],'weights':weights}


def probabilities(P,beta):
    raw=np.asarray(P)*np.exp(np.asarray(beta)-max(beta));return raw/raw.sum(axis=-1,keepdims=True)


def objective(beta,P,y,w):
    q=probabilities(P,beta)
    value=float(sum(-w[i]*np.log(q[i,y[i]]) for i in range(len(y)))+5*sum(v*v for v in beta))
    gradient=np.array([sum(w*(q[:,j]-(y==j)))+10*beta[j] for j in range(4)])
    return value,gradient


def solve(P,y,w):
    result=minimize(objective,np.zeros(4),args=(P,y,w),jac=True,method='L-BFGS-B',
        options={'maxiter':2000,'gtol':1e-10,'ftol':1e-15,'maxls':50})
    assert max(abs(objective(result.x,P,y,w)[1]))<1e-6
    return result.x
