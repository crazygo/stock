"""Independent design/statistics, objective derivative, and damped Newton reference."""
import numpy as np
from scipy.special import softmax


def arrays(panels,ids,fields,mode):
    n=len(ids);base=np.array([2.**((j-n+1)/126) for j in range(n)])
    ess=sum(base)**2/sum(base*base);dw=base*ess/sum(base);x=[];y=[];outcomes=[];groups=[]
    keep=[0,1,2,3,4,5,6,7,8,9,10,16,17];signed={0,1,2,5,6,7,8,9,10}
    for pair,p in panels.items():
        for side,sign in [('L',1),('S',-1)]:
            group=len(x)
            values=p.loc[ids,fields].to_numpy(float)
            x.append(values[:,keep]*[sign if j in signed else 1 for j in keep])
            y.extend(p.loc[ids,side+'_'+mode].astype(int));outcomes.extend(p.loc[ids,side+'_log']);groups.extend([group]*n)
    return np.concatenate(x),np.array(y),np.array(outcomes),np.array(groups),dw,ess


def setup(X,y,values,groups,dw,classes):
    n=len(dw);w=np.concatenate([dw/6 for _ in range(6)]);ess=sum(dw)
    center=np.sum(w[:,None]*X,axis=0)/ess
    sd=np.sqrt(np.sum(w[:,None]*(X-center)**2,axis=0)/ess);sd=np.where(sd<1e-12,1,sd)
    A=np.zeros((len(X),19));A[np.arange(len(X)),groups]=1;A[:,6:]=(X-center)/sd
    pooled=np.array([sum(w[y==j])/ess for j in range(classes)])
    pooled_sum=np.array([sum(w[y==j]*values[y==j])/ess for j in range(classes)])
    active=np.flatnonzero(pooled>0);Y=np.array([[int(v==j) for j in active] for v in y]);offset=np.log(pooled[active])
    prior=np.zeros((6,classes));means=np.zeros_like(prior);mass=np.zeros_like(prior);totals=np.zeros_like(prior)
    for g in range(6):
        group=groups==g;gy=y[group];gv=values[group]
        for j in range(classes):
            mass[g,j]=sum(dw[gy==j]);totals[g,j]=sum(dw[gy==j]*gv[gy==j]);den=mass[g,j]+20*pooled[j]
            prior[g,j]=den/(ess+20)
            means[g,j]=(totals[g,j]+20*pooled_sum[j])/den if den else 0
    return {'A':A,'Y':Y,'weights':w,'center':center,'scale':sd,'active':active,'offset':offset,
        'prior':prior,'payoff_means':means,'pooled_prior':pooled,'pooled_payoff_sum':pooled_sum,
        'own_class_mass':mass,'own_payoff_sum':totals}


def evaluate(coeff,ref,hessian=False):
    A,Y,w,off=ref['A'],ref['Y'],ref['weights'],ref['offset'];scores=A@coeff+off
    p=softmax(scores,axis=1);largest=scores.max(axis=1);logden=largest+np.log(np.exp(scores-largest[:,None]).sum(axis=1))
    loss=float(sum(w*(logden-(scores*Y).sum(axis=1)))+5*np.sum(coeff**2))
    grad=np.column_stack([A.T@(w*(p[:,j]-Y[:,j])) for j in range(Y.shape[1])])+10*coeff
    if not hessian:return loss,grad
    width,k=coeff.shape;H=np.eye(width*k)*10
    for a in range(k):
        for b in range(k):
            weighted=w*p[:,a]*((1 if a==b else 0)-p[:,b])
            H[a::k,b::k]+=A.T@(weighted[:,None]*A)
    return loss,grad,H


def newton(ref):
    B=np.zeros((ref['A'].shape[1],ref['Y'].shape[1]))
    for iteration in range(80):
        value,grad,H=evaluate(B,ref,True)
        if np.max(abs(grad))<1e-9:return B,iteration
        step=np.linalg.solve(H,grad.ravel()).reshape(B.shape);alpha=1.
        for _ in range(50):
            trial=B-alpha*step;new,newgrad=evaluate(trial,ref)
            roundoff=64*np.finfo(float).eps*max(1.,abs(value))
            derivative_improves=(np.max(abs(grad))<1e-5 and np.max(abs(newgrad))<np.max(abs(grad))/2)
            if new<=value-1e-4*alpha*float(np.sum(grad*step)) or (new<=value+roundoff and derivative_improves):break
            alpha*=.5
        else:
            if np.max(abs(grad))<1e-7:return B,iteration
            raise ValueError('Independent Newton line search failed')
        B=trial
    if np.max(abs(evaluate(B,ref)[1]))<1e-7:return B,80
    raise ValueError('Independent Newton did not converge')
