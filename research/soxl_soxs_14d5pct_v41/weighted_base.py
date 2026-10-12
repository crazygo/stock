"""Original paired Ridge/logit learner with explicit registered sample weights."""
from clock_training import np,StandardScaler,Ridge,LogisticRegression

def fit_weighted(panel, dates, cols, w, minimum=120):
    kind = 'regret'; rows = panel[panel.date.isin(dates)]
    w = np.asarray(w, dtype=float)
    info = {'dates': dates, 'members': list(rows.index), 'columns': cols, 'ready': len(dates)>=minimum,
        'parent_weights': w.tolist(), 'weight_sum': float(w.sum()),
        'parent_effective_n': float(w.sum()**2/(w@w)) if len(w) else 0.}
    assert len(rows) == len(dates) and rows.feature_ready.all()
    if not info['ready']: return None, info
    X = rows[cols].to_numpy(); C = rows.C.to_numpy(); D = rows.D.to_numpy(); M = np.abs(D)
    assert np.isfinite(X).all() and np.isfinite(C).all() and np.isfinite(D).all()
    scaler = StandardScaler().fit(X, sample_weight=w); Z = scaler.transform(X)
    model = {'kind': kind, 'columns': cols, 'scaler': scaler, 'heads': {},
             'means': {'C': float(np.average(C, weights=w)), 'D': float(np.average(D, weights=w))}}
    cw = w*M if kind=='regret' else w.copy()
    if cw.sum() > 0: cw = cw*len(w)/cw.sum()
    fallback_q = float(np.average(D>0, weights=cw)) if cw.sum()>0 else .5
    model['q'] = fallback_q
    if kind in ['regret', 'sign']:
        for target, y in [('C', C), ('M', M)]:
            model['heads'][target] = Ridge(alpha=100., solver='svd').fit(Z, y, sample_weight=w)
        if len(np.unique((D[cw>0]>0).astype(int))) == 2:
            classifier = LogisticRegression(C=.1, solver='lbfgs', max_iter=2000, tol=1e-10)
            classifier.fit(Z, (D>0).astype(int), sample_weight=cw)
            assert classifier.n_iter_.max() < 2000
            model['heads']['q'] = classifier
    elif kind=='ridge':
        for target, y in [('C', C), ('D', D)]:
            model['heads'][target] = Ridge(alpha=100., solver='svd').fit(Z, y, sample_weight=w)
    info.update(classification_weights=cw.tolist(), fallback_q=fallback_q,
                center=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), means=model['means'],
                heads={k: {'coefficients': h.coef_.tolist(), 'intercept': np.asarray(h.intercept_).tolist()}
                       for k,h in model['heads'].items()})
    return model, info
