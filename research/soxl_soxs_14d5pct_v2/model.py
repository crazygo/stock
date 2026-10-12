"""Registered model routes; no parameter choice from evaluation-month labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import ETFS


def feature_columns(table):
    return [c for c in table.columns if c not in ['date', 'clock']]


def fit(X, outcomes):
    fitted = {}
    for symbol in ETFS:
        labels = outcomes[symbol].loc[X.index]
        hit = labels.target_hit.astype(int)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=.2, max_iter=2000, random_state=814))
        classifier.fit(X, hit)
        def forest():
            return ExtraTreesRegressor(n_estimators=160, min_samples_leaf=16, max_depth=5,
                                       random_state=814, n_jobs=2, max_features=.8)
        direct = forest().fit(X, labels.net_return)
        missed = hit == 0
        miss_model = forest().fit(X.loc[missed], labels.loc[missed, 'net_return']) if missed.sum() >= 32 else None
        fitted[symbol] = {'p': classifier, 'direct': direct, 'miss': miss_model,
                          'miss_mean': float(labels.loc[missed, 'net_return'].mean()),
                          'win_mean': float(labels.loc[~missed, 'net_return'].mean()),
                          'base_probability': float(hit.mean())}
    return fitted


def predict(models, X):
    result = pd.DataFrame(index=X.index)
    for symbol, fitted in models.items():
        probability = fitted['p'].predict_proba(X)[:, 1]
        miss = fitted['miss'].predict(X) if fitted['miss'] is not None else np.full(len(X), fitted['miss_mean'])
        result[f'{symbol}_p'] = probability
        result[f'{symbol}_conditional_ev'] = probability * fitted['win_mean'] + (1 - probability) * miss
        result[f'{symbol}_direct_ev'] = fitted['direct'].predict(X)
        result[f'{symbol}_base_p'] = fitted['base_probability']
    return result


def actions(table, prediction, route, threshold=0.0, clocks='all_registered_clocks'):
    result = {}
    for at, row in table.iterrows():
        if clocks == '08:55:30' and row.clock != clocks:
            result[at] = 'CASH'; continue
        if route == 'cash':
            result[at] = 'CASH'
        elif route == 'trend20':
            result[at] = 'SOXL' if row.SOXX_ret20 > 0 else 'SOXS'
        else:
            a = prediction.loc[at, f'SOXL_{route}_ev']; b = prediction.loc[at, f'SOXS_{route}_ev']
            result[at] = ('SOXL' if a >= b else 'SOXS') if max(a, b) > threshold else 'CASH'
    return result
