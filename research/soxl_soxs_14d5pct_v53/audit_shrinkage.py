"""Independent piecewise logit design and transformed-coordinate KKT audit."""
from curve_reliability import np,pd,CLOCKS,core
from scipy.special import expit
from verify_reliability import close
from verify_continuation import near


def design(X,q,kind):
    rows=[]
    for row,value in zip(X,q):
        qc=min(max(float(value),1e-6),1-1e-6);f=np.log(qc)-np.log1p(-qc)
        if kind=='global':rows.append([f]);continue
        u=float(row[1]);assert 0<=u<=1
        if kind=='linear':rows.append([f*(1-u),f*u])
        else:
            assert kind=='curve'
            rows.append([f*(1-2*u),f*2*u,0.] if u<=.5 else [0.,f*2*(1-u),f*(2*u-1)])
    return np.array(rows)


def regularizer(b,kind):
    if kind=='curve':
        a=(b[0]+b[2])/2-1;linear=(b[2]-b[0])/2;curve=b[1]-(b[0]+b[2])/2
        return .5*(a*a+linear*linear+curve*curve),np.array([.5*(a-linear-curve),curve,.5*(a+linear-curve)])
    delta=b-1;factor=.5 if kind=='linear' else 1.
    return .5*factor*sum(delta**2),factor*delta


def check_shrinkage(X,q,D,w,n,kind,model):
    assert kind in ['curve','linear','global']
    assert model['kind']=={'curve':'curve','linear':'state','global':'signed'}[kind]
    X=np.asarray(X,float);q=np.asarray(q,float);D=np.asarray(D,float)
    assert X.shape==(len(q),1 if kind=='global' else 2) and len(q)==len(D)==len(w)
    count={'curve':3,'linear':2,'global':1}[kind];prior=np.ones(count);close(model['prior'],prior)
    if kind=='curve':
        Q=np.array([[.75,-.5,.25],[-.5,1.,-.5],[.25,-.5,.75]])
        close(model['penalty_matrix'],Q);assert min(np.linalg.eigvalsh(Q))>0;assert model['knots']==[0.,.5,1.]
    else:close(model['penalty'],[.5,.5] if kind=='linear' else [1.])
    w=np.asarray(w,float);w=w*n/sum(w);weights=w*abs(D);mass=sum(weights)
    assert model['no_direction_mass']==bool(mass==0) and model['n_dates']==n
    if mass>0:weights*=n/mass
    A=design(X,q,kind);b=np.array(model['coefficients'])
    assert b.shape==prior.shape and np.isfinite(b).all() and ((b>=-1)&(b<=1)).all()
    eta=np.sum(A*b,axis=1);y=D>0;penalty,penalty_gradient=regularizer(b,kind)
    gradient=np.sum((weights*(expit(eta)-y))[:,None]*A,axis=0)+penalty_gradient
    pg=[min(g,0.) if x==-1 else max(g,0.) if x==1 else g for x,g in zip(b,gradient)]
    assert max(abs(np.array(pg)))<=2e-8*max(1,n)
    objective=float(sum(weights*(np.maximum(eta,0)+np.log1p(np.exp(-abs(eta)))-y*eta))+penalty)
    near(model['objective'],objective);near(model['projected_gradient_max'],max(abs(np.array(pg))))
    assert model['projected_gradient_max']<=1e-8*max(1,n)
    if kind=='global':
        assert model['bounds']==[-1.,1.] and 0<=model['bisection_steps']<=80
        near(model['gradient'],gradient[0]);close(model['center'],[]);close(model['scale'],[])
    else:
        assert model['bounds']==[[-1.,1.]]*count and 0<=model['iterations']<=100
        close(model['gradient'],gradient)
    if mass==0:close(b,prior)
    return {'kind':kind,'coefficients':b,'no_mass':mass==0},1


def reference_q(model,x,q):
    if model['no_mass']:return float(q)
    return float(expit(design([x],[q],model['kind'])[0]@model['coefficients']))


def input_rows(panel,records,ids,cols):
    assert cols in [[],['state_heat']]
    X=np.array([[CLOCKS.index(panel.loc[t,'clock'])]+[float(panel.loc[t,c]) for c in cols] for t in ids])
    q=np.array([records[t]['fits']['full']['raw']['q'] for t in ids],float)
    return X,q
