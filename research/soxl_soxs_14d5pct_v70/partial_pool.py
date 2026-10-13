"""Numerical prototype: date-weighted, pair-specific slopes with shared shrinkage.

This module does not read market data, construct labels, or issue trade signals.
The caller must establish point-in-time eligibility and distinct-date weights.
"""
import numpy as np


def penalty(mass, common=100., deviation=300., mode='partial'):
    mass = np.asarray(mass, float)
    if mass.ndim != 1 or not np.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError('Positive pair loss-mass fractions required')
    mass = mass / mass.sum()
    if common <= 0 or deviation <= 0:
        raise ValueError('Positive regularization required')
    if mode == 'partial':
        return deviation * np.diag(mass) + (common - deviation) * np.outer(mass, mass)
    if mode == 'independent':
        return common * np.diag(mass)
    raise ValueError('Unknown sharing mode')


def fit(X, Y, pair, weights, distinct_dates, mode='partial', common=100., deviation=300.):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    pair = np.asarray(pair); weights = np.asarray(weights, float)
    if X.ndim != 2 or Y.ndim != 2 or len(Y) != len(X) or pair.shape != (len(X),) or weights.shape != (len(X),):
        raise ValueError('Inconsistent training arrays')
    if not len(X) or not np.isfinite(X).all() or not np.isfinite(Y).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Finite complete training arrays and positive weights required')
    if not isinstance(distinct_dates, int) or distinct_dates < 1:
        raise ValueError('Real distinct-date count required, not row count')
    names = np.unique(pair).tolist(); k = len(names); width = X.shape[1]
    w = weights * distinct_dates / weights.sum()
    indices = np.array([names.index(value) for value in pair])
    centers = []; scales = []; intercepts = []; mass = []
    Z = np.empty_like(X); R = np.empty_like(Y)
    for i in range(k):
        mask = indices == i; v = w[mask]
        center = np.average(X[mask], axis=0, weights=v)
        scale = np.sqrt(np.average((X[mask] - center) ** 2, axis=0, weights=v))
        scale[scale < 1e-12] = 1.
        intercept = np.average(Y[mask], axis=0, weights=v)
        Z[mask] = (X[mask] - center) / scale; R[mask] = Y[mask] - intercept
        centers.append(center); scales.append(scale); intercepts.append(intercept)
        mass.append(v.sum() / distinct_dates)
    if mode == 'pooled':
        design = Z; matrix = common * np.eye(width)
    else:
        design = (np.eye(k)[indices, :, None] * Z[:, None, :]).reshape(len(X), k * width)
        matrix = np.kron(penalty(mass, common, deviation, mode), np.eye(width))
    L = np.linalg.cholesky(matrix)
    A = np.vstack((design * np.sqrt(w)[:, None], L.T))
    target = np.vstack((R * np.sqrt(w)[:, None], np.zeros((len(matrix), Y.shape[1]))))
    beta = np.linalg.lstsq(A, target, rcond=None)[0]
    coefficients = (np.broadcast_to(beta, (k, *beta.shape)).copy() if mode == 'pooled'
                    else beta.reshape(k, width, Y.shape[1]))
    return {'mode': mode, 'names': names, 'centers': np.array(centers),
            'scales': np.array(scales), 'intercepts': np.array(intercepts),
            'coefficients': coefficients, 'pair_mass': np.array(mass),
            'weights': w, 'penalty': matrix, 'distinct_dates': distinct_dates,
            'common': common, 'deviation': deviation}


def predict(model, X, pair):
    X = np.asarray(X, float); pair = np.asarray(pair)
    if X.ndim != 2 or pair.shape != (len(X),) or not np.isfinite(X).all():
        raise ValueError('Finite query rows and explicit pair identity required')
    output = []
    for row, name in zip(X, pair):
        if name not in model['names']:
            raise ValueError('Unregistered pair')
        i = model['names'].index(name)
        z = (row - model['centers'][i]) / model['scales'][i]
        output.append(model['intercepts'][i] + z @ model['coefficients'][i])
    return np.asarray(output)
