"""Independent weighted box-logistic optimum certificates and prediction."""
from state_reliability import np,pd,CLOCKS,core
from scipy.special import expit
from verify_reliability import close
from verify_continuation import near


def design(X,q,kind):
    rows=[]
    for row,value in zip(X,q):
        qc=min(max(float(value),1e-6),1-1e-6);f=np.log(qc)-np.log1p(-qc)
        if kind=='global':rows.append([f])
        elif kind=='state':rows.append([f*(1-row[1]),f*row[1]])
        else:assert kind=='additive';rows.append([f,2*row[1]-1])
    return np.array(rows)


def check_shrinkage(X,q,D,w,n,kind,model):
    assert kind in ['state','additive','global']
    assert model['kind']==('signed' if kind=='global' else kind)
    X=np.asarray(X,float);q=np.asarray(q,float);D=np.asarray(D,float)
    assert X.shape==(len(q),1 if kind=='global' else 2) and len(q)==len(D)==len(w)
    prior=np.array([1.] if kind=='global' else [1.,1.] if kind=='state' else [1.,0.])
    penalty=np.array([1.] if kind=='global' else [.5,.5] if kind=='state' else [1.,1.])
    close(model['prior'],prior);close(model['penalty'],penalty)
    w=np.asarray(w,float);w=w*n/sum(w);weights=w*abs(D);mass=sum(weights)
    assert model['no_direction_mass']==bool(mass==0) and model['n_dates']==n
    if mass>0:weights*=n/mass
    A=design(X,q,kind);b=np.array(model['coefficients'])
    assert b.shape==prior.shape and np.isfinite(b).all() and ((b>=-1)&(b<=1)).all()
    eta=np.sum(A*b,axis=1);y=D>0
    gradient=np.sum((weights*(expit(eta)-y))[:,None]*A,axis=0)+penalty*(b-prior)
    pg=[]
    for x,g in zip(b,gradient):pg.append(min(g,0.) if x==-1 else max(g,0.) if x==1 else g)
    assert max(abs(np.array(pg)))<=2e-8*max(1,n)
    objective=float(sum(weights*(np.maximum(eta,0)+np.log1p(np.exp(-abs(eta)))-y*eta))+.5*sum(penalty*(b-prior)**2))
    near(model['objective'],objective);near(model['projected_gradient_max'],max(abs(np.array(pg))))
    assert model['projected_gradient_max']<=1e-8*max(1,n)
    if kind=='global':
        assert model['bounds']==[-1.,1.] and 0<=model['bisection_steps']<=80
        near(model['gradient'],gradient[0]);close(model['center'],[]);close(model['scale'],[])
    else:
        assert model['bounds']==[[-1.,1.],[-1.,1.]] and 0<=model['iterations']<=100
        close(model['gradient'],gradient)
    if mass==0:close(b,prior)
    # Positive penalty diagonal makes the Hessian positive definite. Together
    # with feasibility, the coordinate KKT signs certify the unique minimum.
    return {'kind':kind,'coefficients':b,'no_mass':mass==0},1


def reference_q(model,x,q):
    if model['no_mass']:return float(q)
    return float(expit(design([x],[q],model['kind'])[0]@model['coefficients']))


def input_rows(panel,records,ids,cols):
    assert cols in [[],['state_heat']]
    X=np.array([[CLOCKS.index(panel.loc[t,'clock'])]+[float(panel.loc[t,c]) for c in cols] for t in ids])
    q=np.array([records[t]['fits']['full']['raw']['q'] for t in ids],float)
    return X,q
