"""Independent constrained optimum certificates, preprocessing and inference."""
from confidence_shrinkage import np,pd,CLOCKS,core
from scipy.special import expit
from verify_reliability import close
from verify_continuation import near


def check_shrinkage(X,q,D,w,n,kind,model):
    assert model['kind']==kind
    w=np.asarray(w,float);w=w*n/w.sum();X=np.asarray(X,float);p=X.shape[1]-1
    center=[];scale=[]
    for j in range(p):
        m=float(sum(w*X[:,j+1])/sum(w));v=float(sum(w*(X[:,j+1]-m)**2)/sum(w))
        center.append(m);scale.append(np.sqrt(v) if np.sqrt(v)>=1e-12 else 1.)
    close(model['center'],center);close(model['scale'],scale)
    qc=np.clip(q,1e-6,1-1e-6);log_odds=np.log(qc)-np.log1p(-qc)
    A=np.array([[log_odds[i]]+([] if kind=='temperature' else [float(int(row[0])==c) for c in range(4)]+[(row[j+1]-center[j])/scale[j] for j in range(p)]) for i,row in enumerate(X)])
    penalty=np.array([1.] if kind=='temperature' else [1.]*5+[10.]*p);prior=np.zeros(len(penalty));prior[0]=1
    close(model['penalty'],penalty);close(model['prior'],prior)
    weights=w*abs(np.asarray(D));mass=weights.sum()
    assert model['no_direction_mass']==bool(mass==0) and model['n_dates']==n
    if mass>0:weights*=n/mass
    b=np.array(model['coefficients']);assert np.isfinite(b).all() and 0<=b[0]<=1
    eta=A@b;y=np.asarray(D)>0
    gradient=np.sum((weights*(expit(eta)-y))[:,None]*A,axis=0)+penalty*(b-prior)
    tolerance=2e-8*max(1,n)
    if b[0]==0:assert gradient[0]>=-tolerance
    elif b[0]==1:assert gradient[0]<=tolerance
    else:assert abs(gradient[0])<tolerance
    if len(b)>1:assert max(abs(gradient[1:]))<tolerance
    objective=float(sum(weights*(np.maximum(eta,0)+np.log1p(np.exp(-abs(eta)))-y*eta))+.5*sum(penalty*(b-prior)**2))
    near(model['objective'],objective);assert 0<=model['iterations']<=1000
    assert np.isfinite(model['projected_gradient_max']) and model['projected_gradient_max']<=1e-8*max(1,n)
    if mass==0:close(b,prior)
    # Positive diagonal penalties imply strict convexity. Feasibility and KKT
    # conditions certify the unique bounded optimum, including boundary fits.
    return {'kind':kind,'center':np.array(center),'scale':np.array(scale),'coefficients':b,'no_mass':mass==0},1


def reference_q(model,x,q):
    if model['no_mass']:return float(q)
    q0=min(max(float(q),1e-6),1-1e-6);b=model['coefficients']
    value=b[0]*(np.log(q0)-np.log1p(-q0))
    if model['kind']!='temperature':
        value+=b[1+int(x[0])]
        for j,f in enumerate(x[1:]):value+=(f-model['center'][j])/model['scale'][j]*b[5+j]
    return float(expit(value))


def input_rows(panel,records,ids,cols):
    X=np.array([[CLOCKS.index(panel.loc[t,'clock'])]+[float(panel.loc[t,c]) for c in cols] for t in ids])
    q=np.array([records[t]['fits']['full']['raw']['q'] for t in ids],float)
    return X,q
