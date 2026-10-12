"""Independent convex gradient, preprocessing, objective and inference equations."""
from direction_correction import np,pd,CLOCKS,core
from scipy.special import expit
from verify_reliability import close
from verify_continuation import near


def check_offset(X,q,D,w,n,model):
    w=np.asarray(w,float);w=w*n/w.sum();X=np.asarray(X,float)
    p=X.shape[1]-1;center=[];scale=[]
    for j in range(p):
        m=float(sum(w*X[:,j+1])/sum(w));v=float(sum(w*(X[:,j+1]-m)**2)/sum(w))
        center.append(m);scale.append(np.sqrt(v) if np.sqrt(v)>=1e-12 else 1.)
    close(model['center'],center);close(model['scale'],scale)
    A=np.array([[float(int(row[0])==c) for c in range(4)]+
                [(row[j+1]-center[j])/scale[j] for j in range(p)] for row in X])
    qc=np.clip(q,1e-6,1-1e-6);offset=np.log(qc)-np.log1p(-qc)
    penalty=np.array([1.]*4+[10.]*p);close(model['penalty'],penalty)
    weights=w*abs(np.asarray(D));mass=weights.sum()
    assert model['no_direction_mass']==bool(mass==0) and model['n_dates']==n
    if mass>0:weights*=n/mass
    b=np.array(model['coefficients']);eta=offset+A@b;y=np.asarray(D)>0
    gradient=np.sum((weights*(expit(eta)-y))[:,None]*A,axis=0)+penalty*b
    assert max(abs(gradient))<2e-8*max(1,n)
    objective=float(sum(weights*(np.maximum(eta,0)+np.log1p(np.exp(-abs(eta)))-y*eta))+.5*sum(penalty*b*b))
    near(model['objective'],objective);assert 0<=model['iterations']<60
    if mass==0:close(b,np.zeros_like(b))
    # Strictly positive penalties make the Hessian positive definite: stationary
    # point is the unique optimum, including single-class bootstrap samples.
    return {'center':np.array(center),'scale':np.array(scale),'coefficients':b,'no_mass':mass==0},1


def reference_q(model,x,q):
    if model['no_mass']:return float(q)
    q0=min(max(float(q),1e-6),1-1e-6);b=model['coefficients']
    log_odds=np.log(q0)-np.log1p(-q0)+b[int(x[0])]
    for j,value in enumerate(x[1:]):log_odds+=(value-model['center'][j])/model['scale'][j]*b[4+j]
    return float(expit(log_odds))


def input_rows(panel,records,ids,cols):
    X=np.array([[CLOCKS.index(panel.loc[t,'clock'])]+[float(panel.loc[t,c]) for c in cols] for t in ids])
    q=np.array([records[t]['fits']['full']['raw']['q'] for t in ids],float)
    return X,q
