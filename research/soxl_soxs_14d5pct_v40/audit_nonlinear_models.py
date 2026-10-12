"""Independent ridge equations and logit gradients for arbitrary frozen columns."""
import numpy as np
from scipy.special import expit
from verify_continuation import near
from audit_trees import fit_check,probability

def model_equations(panel, ids, at, model, info, columns, kind):
    w = 2. ** (-np.arange(len(ids))[::-1] / 126.)
    if len(w):
        w *= len(w) / w.sum()
    np.testing.assert_allclose(info['parent_weights'], w, atol=1e-12, rtol=1e-10)
    assert info['ready'] == (len(ids) >= 120)
    if len(ids) < 120:
        assert model is None
        return {}, 0
    X = panel.loc[ids, columns].to_numpy(dtype=float)
    C = panel.loc[ids, 'C'].to_numpy(dtype=float); D = panel.loc[ids, 'D'].to_numpy(dtype=float); M = abs(D)
    center = np.average(X, axis=0, weights=w)
    scale = np.sqrt(np.average((X-center)**2, axis=0, weights=w)); scale[scale < 1e-12] = 1.
    np.testing.assert_allclose(model['scaler'].mean_, center, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(model['scaler'].scale_, scale, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(info['center'], center, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(info['scale'], scale, atol=1e-10, rtol=1e-10)
    Z = (X-center)/scale; heads = model['heads']; checks = 0
    for target, y in [('C', C), ('M', M)]:
        # Solve weighted ridge normal equations independently of sklearn SVD.
        zmean = np.average(Z, axis=0, weights=w); ymean = np.average(y, weights=w)
        A = Z-zmean
        coefficient = np.linalg.solve(A.T @ (w[:, None]*A) + 100*np.eye(len(columns)), A.T @ (w*(y-ymean)))
        intercept = ymean-zmean@coefficient
        np.testing.assert_allclose(heads[target].coef_, coefficient, atol=1e-10, rtol=1e-8)
        near(heads[target].intercept_, intercept)
        np.testing.assert_allclose(info['heads'][target]['coefficients'], coefficient, atol=1e-10, rtol=1e-8)
        near(info['heads'][target]['intercept'], intercept); checks += 1
    cw = w*M
    if cw.sum() > 0:
        cw *= len(w)/cw.sum()
    np.testing.assert_allclose(info['classification_weights'], cw, atol=1e-10, rtol=1e-10)
    fallback = np.average(D > 0, weights=cw) if cw.sum() > 0 else .5
    near(model['q'], fallback); near(info['fallback_q'], fallback)
    if len(np.unique((D[cw>0]>0).astype(int)))==2:
        if kind=='linear':
            h=heads['q'];residual=cw*(expit(Z@h.coef_[0]+h.intercept_[0])-(D>0))
            gradient=Z.T@residual+h.coef_[0]/.1
            assert max(np.max(abs(gradient)),abs(residual.sum()))<1e-6*max(1,len(w));checks+=1
        else:checks+=fit_check(heads['q'],Z,(D>0).astype(int),cw,2 if kind=='tree' else 1)
    else:assert 'q' not in heads
    if not panel.loc[at, 'feature_ready']:
        return {}, checks
    query = (panel.loc[at, columns].to_numpy(dtype=float)-center)/scale
    c = float(query@heads['C'].coef_+heads['C'].intercept_)
    m = max(0., float(query@heads['M'].coef_+heads['M'].intercept_))
    q=fallback
    if 'q' in heads:
        q=float(expit(query@heads['q'].coef_[0]+heads['q'].intercept_[0])) if kind=='linear' else float(probability(heads['q'],query[None,:])[0])
    d = m*(2*q-1)
    return {'C': c, 'D': d, 'M': m, 'q': q, 'SOXL': c+d, 'SOXS': c-d}, checks

