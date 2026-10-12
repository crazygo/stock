"""Independent weighted Ridge equations and direction logit gradients."""
from macro_learning import np
from verify_reliability import close
from verify_continuation import near

def weighted_equations(panel,ids,at,model,info,columns,w):
    from scipy.special import expit
    assert info['ready']==(len(ids)>=120)
    if len(ids)<120:assert model is None;return {},0
    X=panel.loc[ids,columns].to_numpy(float);C=panel.loc[ids,'C'].to_numpy(float);D=panel.loc[ids,'D'].to_numpy(float);M=abs(D)
    center=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1
    close(model['scaler'].mean_,center);close(model['scaler'].scale_,scale);close(info['center'],center);close(info['scale'],scale)
    near(info['means']['C'],np.average(C,weights=w));near(info['means']['D'],np.average(D,weights=w))
    Z=(X-center)/scale;heads=model['heads'];checks=0
    for target,y in [('C',C),('M',M)]:
        zm=np.average(Z,axis=0,weights=w);ym=np.average(y,weights=w);A=Z-zm
        coef=np.linalg.solve(A.T@(w[:,None]*A)+100*np.eye(len(columns)),A.T@(w*(y-ym)));intercept=ym-zm@coef
        close(heads[target].coef_,coef);near(heads[target].intercept_,intercept);close(info['heads'][target]['coefficients'],coef);near(info['heads'][target]['intercept'],intercept);checks+=1
    cw=w*M
    if cw.sum()>0:cw*=len(w)/cw.sum()
    close(info['classification_weights'],cw);q=float(np.average(D>0,weights=cw)) if cw.sum()>0 else .5
    near(model['q'],q);near(info['fallback_q'],q)
    if len(np.unique(D[cw>0]>0))==2:
        h=heads['q'];residual=cw*(expit(Z@h.coef_[0]+h.intercept_[0])-(D>0));gradient=Z.T@residual+h.coef_[0]/.1
        assert max(abs(gradient).max(),abs(residual.sum()))<1e-6*max(1,len(w));close(info['heads']['q']['coefficients'],h.coef_);close(info['heads']['q']['intercept'],h.intercept_);checks+=1
    else:assert 'q' not in heads
    if not panel.loc[at,'feature_ready']:return {},checks
    x=(panel.loc[at,columns].to_numpy(float)-center)/scale;c=float(x@heads['C'].coef_+heads['C'].intercept_);m=max(0.,float(x@heads['M'].coef_+heads['M'].intercept_))
    if 'q' in heads:q=float(expit(x@heads['q'].coef_[0]+heads['q'].intercept_[0]))
    d=m*(2*q-1);return {'C':c,'D':d,'M':m,'q':q,'SOXL':c+d,'SOXS':c-d},checks
