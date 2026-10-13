"""Replicated randomized Sobol integration; no outcome-based seed selection."""
import numpy as np
from scipy.stats import qmc

REPLICATES = 8


def uniforms(seed, dimensions, count=2048, replication=0):
    points = count // REPLICATES
    if count % REPLICATES or points < 2 or points & (points - 1):
        raise ValueError('Each of eight scrambles requires a power-of-two point count >=2')
    children = np.random.SeedSequence([seed, replication]).spawn(REPLICATES)
    return np.concatenate([
        qmc.Sobol(dimensions, scramble=True, bits=30,
                  rng=np.random.default_rng(child)).random_base2(points.bit_length()-1)
        for child in children
    ])


def integration_statistics(log_returns):
    values = np.asarray(log_returns)
    if len(values) % REPLICATES:
        raise ValueError('Incomplete scramble')
    means = values.reshape(REPLICATES, -1, 2).mean(axis=1)
    covariance = np.cov(means, rowvar=False, ddof=1) / REPLICATES
    return {
        'method': 'eight_independent_scrambles',
        'replicates': REPLICATES,
        'points_per_replicate': len(values)//REPLICATES,
        'replicate_means': means.tolist(),
        'mean_covariance': covariance.tolist(),
        'standard_error': np.sqrt(np.diag(covariance)).tolist(),
    }
