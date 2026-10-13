"""Independent scalar convex optimum certificates and inference."""
from signed_reliability import np,pd,CLOCKS,core
from scipy.special import expit
from verify_reliability import close
from verify_continuation import near


def check_shrinkage(X,q,D,w,n,kind,model):
    assert kind in ['signed','nonnegative']
    assert model['kind']==('signed' if kind=='signed' else 'temperature')
    X=np.asarray(X,float);q=np.asarray(q,float);D=np.asarray(D,float)
    assert X.shape==(len(q),1) and len(q)==len(D)==len(w)
    close(model['center'],[]);close(model['scale'],[])
    close(model['penalty'],[1.]);close(model['prior'],[1.])
    w=np.asarray(w,float);w=w*n/sum(w)
    weights=w*abs(D);mass=sum(weights)
    assert model['no_direction_mass']==bool(mass==0) and model['n_dates']==n
    if mass>0:weights*=n/mass
    qc=np.clip(q,1e-6,1-1e-6);x=np.log(qc)-np.log1p(-qc)
    b=np.array(model['coefficients']);assert b.shape==(1,) and np.isfinite(b).all()
    lower=-1. if kind=='signed' else 0.;s=float(b[0]);assert lower<=s<=1
    eta=s*x;y=D>0
    gradient=float(sum(weights*(expit(eta)-y)*x)+s-1)
    pg=min(gradient,0.) if s==lower else max(gradient,0.) if s==1 else gradient
    assert abs(pg)<=2e-8*max(1,n)
    objective=float(sum(weights*(np.maximum(eta,0)+np.log1p(np.exp(-abs(eta)))-y*eta))+.5*(s-1)**2)
    near(model['objective'],objective)
    assert np.isfinite(model['projected_gradient_max']) and model['projected_gradient_max']<=1e-8*max(1,n)
    near(model['projected_gradient_max'],abs(pg))
    if kind=='signed':
        assert model['bounds']==[-1.,1.] and 0<=model['bisection_steps']<=80
        near(model['gradient'],gradient)
    else:assert 0<=model['iterations']<=1000
    if mass==0:close(b,[1.])
    # Hessian sum(w*p*(1-p)*x*x)+1 > 0: feasibility and KKT certify
    # the unique global constrained optimum without reusing the fitter.
    return {'kind':kind,'coefficients':b,'no_mass':mass==0},1


def reference_q(model,x,q):
    if model['no_mass']:return float(q)
    qc=min(max(float(q),1e-6),1-1e-6)
    return float(expit(model['coefficients'][0]*(np.log(qc)-np.log1p(-qc))))


def input_rows(panel,records,ids,cols):
    assert cols==[]
    X=np.array([[CLOCKS.index(panel.loc[t,'clock'])] for t in ids])
    q=np.array([records[t]['fits']['full']['raw']['q'] for t in ids],float)
    return X,q
