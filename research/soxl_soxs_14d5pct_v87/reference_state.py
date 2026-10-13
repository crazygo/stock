"""Independent scalar sample construction and multiclass optimizer."""
import numpy as np
from scipy.optimize import minimize


def coordinates(row,j):
    sign=1-2*j
    return [float(row['trend5'])*sign,float(row['current'])*sign,float(row['relative5'])*sign,
            float(row['log_vol20']),float(row['trend5'])*float(row['current']),float(row['event_future_count']),float(row['event_future_proximity']),float(row['event_recent_count']),float(row['event_recent_proximity'])]


def data(panel,parents,history,population,clocks):
    rows=[];dates=[];masses=[];N=len(parents)
    for i,p in enumerate(parents):
        day=[]
        for t in p['members']:
            e=history[t]['events']['recent_ratio']
            if population=='eligible' and e['action']=='CASH':continue
            j=['SOXL','SOXS'].index(e['top_symbol']);r=panel.loc[t]
            day.append((t,j,e['probabilities'][j],int(r['L_gap' if j==0 else 'S_gap']),coordinates(r,j),clocks.index(r['clock'])))
        if day:
            mass=2**(-(N-i-1)/63);dates.append(p['date']);masses.append(mass)
            rows.extend([(*r,mass/len(day)) for r in day])
    total=sum(masses);ess=total**2/sum(x*x for x in masses) if masses else 0.
    w=np.array([r[-1]*ess/total for r in rows]);X=np.array([r[4] for r in rows]).reshape(-1,9)
    return np.array([r[2] for r in rows]).reshape(-1,4),np.array([r[3] for r in rows],int),w,X,np.array([r[1] for r in rows],int),np.array([r[5] for r in rows],int),{
        'dates':dates,'real_dates':len(dates),'date_ess':ess,'members':[[r[0],r[1]] for r in rows],'weights':w}


def matrix(X,sides,clocks,w,mode):
    mu=np.array([sum(w*X[:,j])/sum(w) for j in range(9)])
    sd=np.array([np.sqrt(sum(w*(X[:,j]-mu[j])**2)/sum(w)) for j in range(9)]);sd[sd<1e-12]=1.
    rows=[]
    for x,s,c in zip(X,sides,clocks):
        z=[1.]
        z.extend([float(s)]+[float(c==k) for k in range(1,8)])
        selected=range(9) if mode=='event' else range(5) if mode=='state' else [5,6,7,8]
        z.extend([(x[j]-mu[j])/sd[j] for j in selected])
        rows.append(z)
    return np.array(rows),mu,sd


def posterior(P,Z,B):
    adjustments=Z@B
    unnormalized=np.asarray(P)*np.exp(adjustments-adjustments.max(axis=1)[:,None])
    return unnormalized/unnormalized.sum(axis=1)[:,None]


def loss_gradient(flat,P,y,w,Z):
    B=np.asarray(flat).reshape(Z.shape[1],4);Q=posterior(P,Z,B)
    value=sum(-w[i]*np.log(Q[i,y[i]]) for i in range(len(y)))+5*sum(flat*flat)
    g=np.empty_like(B)
    for j in range(Z.shape[1]):
        for k in range(4):g[j,k]=sum(w*Z[:,j]*(Q[:,k]-(y==k)))+10*B[j,k]
    return float(value),g.ravel()


def solve(P,y,w,Z):
    r=minimize(loss_gradient,np.zeros(Z.shape[1]*4),args=(P,y,w,Z),jac=True,method='L-BFGS-B',
               options={'maxiter':2000,'gtol':1e-10,'ftol':0.,'maxls':50,'maxcor':50})
    g=loss_gradient(r.x,P,y,w,Z)[1]
    assert max(abs(g))<1e-6,max(abs(g))
    return r.x.reshape(-1,4)


def predict(model,P,x,side,clock):
    z=[1.]
    z.extend([float(side)]+[float(clock==k) for k in range(1,8)])
    selected=range(9) if model['mode']=='event' else range(5) if model['mode']=='state' else [5,6,7,8]
    z.extend([(x[j]-model['center'][j])/model['scale'][j] for j in selected])
    return posterior(np.asarray(P).reshape(1,4),np.array(z).reshape(1,-1),np.asarray(model['beta']))[0]
