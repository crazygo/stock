"""Post-result synthetic diagnosis only; never reads market data or emits actions.

Test whether cloning each clock and splitting its row weights leaves the
regression objective unchanged. The count-normalized alternative is a future
hypothesis, not a replacement for the preregistered v54 model.
"""
import argparse
from dense_learning import np, fit_map, forecast, dump, sha, Path


def joint_fit(X, Y, w, n, deviation):
    k = int(X[:, 2].max()) + 1
    w = w * n / w.sum()
    clocks = X[:, 2].astype(int)
    centers, scales = [], []
    A = np.zeros((len(X), 3 * k))
    for c in range(k):
        mask = clocks == c
        center = np.average(X[mask, :2], axis=0, weights=w[mask])
        scale = np.sqrt(np.average((X[mask, :2] - center) ** 2, axis=0, weights=w[mask]))
        scale[scale < 1e-12] = 1
        A[mask, c * 3] = 1
        A[mask, c * 3 + 1:c * 3 + 3] = (X[mask, :2] - center) / scale
        centers.append(center)
        scales.append(scale)
    P = np.zeros((3 * k, 3 * k))
    # Build mean and pairwise deviation penalties from coefficient operators,
    # independently of the production Kronecker matrix.
    for feature in range(2):
        mean = np.zeros(3 * k)
        mean[feature + 1::3] = 1 / k
        P += 20 * np.outer(mean, mean)
        for c in range(k):
            delta = -mean.copy()
            delta[c * 3 + feature + 1] += 1
            P += deviation * np.outer(delta, delta)
    normal = A.T @ (w[:, None] * A) + P
    rhs = A.T @ (w[:, None] * Y)
    beta = np.linalg.solve(normal, rhs)
    np.testing.assert_allclose(normal @ beta, rhs, atol=1e-10, rtol=1e-10)
    return {
        'kind': 'partial', 'input_center': np.array(centers).tolist(),
        'input_scale': np.array(scales).tolist(),
        'intercepts': beta.reshape(k, 3, 2)[:, 0].tolist(),
        'coefficients': beta.reshape(k, 3, 2)[:, 1:].tolist(),
    }


def penalty_value(beta, deviation):
    mean = beta.mean(axis=0)
    return 20 * np.sum(mean ** 2) + deviation * np.sum((beta - mean) ** 2)


def audit(output):
    rng = np.random.default_rng(20261013)
    n, k = 60, 4
    X = np.column_stack([rng.normal(size=(n * k, 2)), np.tile(np.arange(k), n)])
    coefficients = rng.normal(size=(k, 2, 2))
    intercepts = rng.normal(size=(k, 2))
    c = X[:, 2].astype(int)
    Y = np.einsum('ij,ijk->ik', X[:, :2], coefficients[c]) + intercepts[c] + rng.normal(0, .1, (n * k, 2))
    dw = 2. ** (-np.arange(n)[::-1] / 63)
    w = np.repeat(dw / k, k)
    base = fit_map(X, Y, w, n, k)
    independent = joint_fit(X, Y, w, n, 20)
    pred = forecast(base, X)
    np.testing.assert_allclose(pred, forecast(independent, X), atol=1e-10, rtol=1e-10)
    results = []
    for copies in (2, 3):
        clone_X = np.repeat(X, copies, axis=0)
        clone_X[:, 2] = np.repeat(c * copies, copies) + np.tile(np.arange(copies), len(X))
        clone_Y, clone_w = np.repeat(Y, copies, axis=0), np.repeat(w / copies, copies)
        clone_k = k * copies
        frozen = fit_map(clone_X, clone_Y, clone_w, n, clone_k)
        direct = joint_fit(clone_X, clone_Y, clone_w, n, 20)
        fixed = joint_fit(clone_X, clone_Y, clone_w, n, 80 / clone_k)
        frozen_pred, fixed_pred = forecast(frozen, clone_X), forecast(fixed, clone_X)
        expected = np.repeat(pred, copies, axis=0)
        np.testing.assert_allclose(frozen_pred, forecast(direct, clone_X), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(fixed_pred, expected, atol=1e-10, rtol=1e-10)
        old_penalty = penalty_value(coefficients, 20)
        cloned_coefficients = np.repeat(coefficients, copies, axis=0)
        normalized_penalty = penalty_value(cloned_coefficients, 80 / clone_k)
        np.testing.assert_allclose(normalized_penalty, old_penalty, atol=1e-10, rtol=1e-10)
        frozen_penalty = penalty_value(cloned_coefficients, 20)
        expected_extra = 20 * (copies - 1) * np.sum((coefficients - coefficients.mean(axis=0)) ** 2)
        np.testing.assert_allclose(frozen_penalty - old_penalty, expected_extra, atol=1e-10, rtol=1e-10)
        frozen_shift = float(np.max(np.abs(frozen_pred - expected)))
        assert frozen_shift > 1e-4
        results.append({'copies': copies, 'clocks': clone_k,
                        'frozen_prediction_max_abs_shift': frozen_shift,
                        'normalized_prediction_max_abs_shift': float(np.max(np.abs(fixed_pred - expected))),
                        'frozen_extra_penalty': float(frozen_penalty - old_penalty),
                        'extra_penalty_formula_verified': True, 'normalized_objective_invariant': True})
    receipt = {'passed': True, 'post_result_only': True, 'synthetic_only': True,
               'market_data_read': False, 'capital_replayed': False, 'primary_rules_changed': False,
               'seed': 20261013, 'dates': n, 'results': results,
               'scope': 'Clock cloning changes v54 deviation penalty relative to unchanged weighted data loss. No attribution of observed losses or evidence of better investment performance.',
               'candidate_formula': '20||mean(beta)||^2 + (80/k)sum||beta_clock-mean||^2; future experiment only',
               'source_sha256': sha(Path(__file__))}
    dump(output, receipt)
    print(receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output)
