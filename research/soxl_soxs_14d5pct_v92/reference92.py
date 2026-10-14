"""Independent simplex-constrained optimizer and scalar projection certificate."""
import math
import numpy as np
from scipy.optimize import minimize


def certificate(raw,observed,record):
    c=np.array([float(v)-sum(raw)/4 for v in raw]);points=np.array([[0.]*4]+[[float(v)-sum(row)/4 for v in row] for row in observed])
    ids=list(record['vertex_indices']);weights=list(record['vertex_weights']);assert len(ids)==len(weights) and len(set(ids))==len(ids)
    assert all(0<=i<len(points) for i in ids) and all(w>=0 for w in weights)
    assert abs(sum(weights)-1)<1e-10
    u=np.array([sum(w*points[i,k] for i,w in zip(ids,weights)) for k in range(4)])
    np.testing.assert_allclose(u,record['correction'],atol=1e-10,rtol=1e-10)
    gap=max([0.]+[sum((t[k]-u[k])*(c[k]-u[k]) for k in range(4)) for t in points])
    tol=1e-9*max(1.,math.sqrt(sum(c*c))*max(math.sqrt(sum(t*t)) for t in points));assert gap<=tol,(gap,tol)
    assert np.linalg.norm(u)<=np.linalg.norm(c)+1e-9*max(1.,np.linalg.norm(c))
    for k in range(4):
        for j in range(k):
            lo=min(t[k]-t[j] for t in points);hi=max(t[k]-t[j] for t in points)
            assert lo-1e-10<=u[k]-u[j]<=hi+1e-10
    np.testing.assert_allclose([record['distance'],record['optimality_gap'],record['numerical_tolerance']],[np.linalg.norm(c-u),gap,tol],atol=1e-10,rtol=1e-7)
    assert record['context_rows']==len(observed)
    np.testing.assert_allclose(record['zero_anchor_weight'],sum(w for i,w in zip(ids,weights) if i==0),atol=1e-13)
    return dict(optimality_gap=gap,vertices=len(ids))


def solve(raw,observed):
    c=np.array(raw,float,copy=True);c-=c.mean()
    T=np.vstack([np.zeros(4),np.asarray(observed,float).reshape(-1,4)]);T-=T.mean(axis=1)[:,None]
    if len(T)==1:return np.zeros(4)
    # Scale the objective only for numerical conditioning; minimizer unchanged.
    unit=max(1.,np.linalg.norm(c),np.max(np.linalg.norm(T,axis=1)));t=T/unit;target=c/unit
    def objective(a):
        d=a@t-target;return float(.5*d@d),t@d
    initial=np.zeros(len(T));initial[np.argmin(np.sum((t-target)**2,axis=1))]=1
    fit=minimize(objective,initial,jac=True,method='SLSQP',bounds=[(0,1)]*len(T),
        constraints=[{'type':'eq','fun':lambda a:sum(a)-1,'jac':lambda a:np.ones(len(a))}],
        options={'ftol':1e-14,'maxiter':2000})
    assert fit.success,fit.message
    return fit.x@T
