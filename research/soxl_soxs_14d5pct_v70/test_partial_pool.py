"""Independent objective and invariance checks; synthetic data only."""
import unittest
import numpy as np
from partial_pool import fit, predict, penalty


def fixture():
    rng = np.random.default_rng(70)
    pair = np.repeat(['semiconductor', 'nasdaq', 'sp500'], 50)
    X = rng.normal(size=(150, 4)); X[:, -1] = 7.
    Y = np.column_stack((.02 * X[:, 0] + .01 * X[:, 1], -.03 * X[:, 0]))
    Y += rng.normal(scale=.005, size=Y.shape)
    return X, Y, pair, rng.uniform(.5, 1.5, size=150)


class PartialPoolTests(unittest.TestCase):
    def test_penalty_matches_weighted_mean_and_deviations(self):
        mass = np.array([.2, .3, .5]); beta = np.arange(12.).reshape(3, 4) / 10
        mean = mass @ beta
        expected = 100 * (mean @ mean) + 300 * np.sum(mass[:, None] * (beta - mean) ** 2)
        actual = sum(beta[:, j] @ penalty(mass) @ beta[:, j] for j in range(4))
        self.assertAlmostEqual(actual, expected, places=11)

    def test_independent_normal_equations(self):
        X, Y, pairs, weights = fixture(); m = fit(X, Y, pairs, weights, 50)
        names = m['names']; width = X.shape[1]; A = np.zeros((len(X), len(names) * width))
        R = np.empty_like(Y)
        for j, (x, y, pair) in enumerate(zip(X, Y, pairs)):
            i = names.index(pair)
            A[j, i * width:(i + 1) * width] = (x - m['centers'][i]) / m['scales'][i]
            R[j] = y - m['intercepts'][i]
        expected = np.linalg.solve(A.T @ (m['weights'][:, None] * A) + m['penalty'],
                                   A.T @ (m['weights'][:, None] * R))
        np.testing.assert_allclose(m['coefficients'].reshape(expected.shape), expected,
                                   rtol=1e-11, atol=1e-13)

    def test_single_pair_equals_independent(self):
        X, Y, p, w = fixture(); p[:] = 'semiconductor'
        a = fit(X, Y, p, w, 50); b = fit(X, Y, p, w, 50, mode='independent')
        np.testing.assert_allclose(predict(a, X, p), predict(b, X, p), rtol=1e-12, atol=1e-13)

    def test_independent_pair_equals_separate_training(self):
        X, Y, p, w = fixture(); full = fit(X, Y, p, w, 50, mode='independent')
        for name in np.unique(p):
            mask = p == name; solo = fit(X[mask], Y[mask], p[mask], w[mask], 50)
            np.testing.assert_allclose(predict(full, X[mask], p[mask]),
                                       predict(solo, X[mask], p[mask]), rtol=1e-11, atol=1e-13)

    def test_duplicate_rows_do_not_manufacture_dates(self):
        X, Y, p, w = fixture(); a = fit(X, Y, p, w, 50)
        b = fit(np.repeat(X, 3, axis=0), np.repeat(Y, 3, axis=0),
                np.repeat(p, 3), np.repeat(w / 3, 3), 50)
        np.testing.assert_allclose(predict(a, X, p), predict(b, X, p), rtol=1e-11, atol=1e-13)
        self.assertAlmostEqual(b['weights'].sum(), 50.)

    def test_permuting_rows_and_pair_names_preserves_predictions(self):
        X, Y, p, w = fixture(); a = fit(X, Y, p, w, 50)
        order = np.arange(len(X))[::-1]
        renamed = np.array([{'semiconductor': 'a', 'nasdaq': 'z', 'sp500': 'q'}[x] for x in p])
        b = fit(X[order], Y[order], renamed[order], w[order], 50)
        np.testing.assert_allclose(predict(a, X, p), predict(b, X, renamed), rtol=1e-11, atol=1e-13)

    def test_pooled_common_slopes_with_pair_intercepts(self):
        X, Y, p, w = fixture(); m = fit(X, Y, p, w, 50, mode='pooled')
        for coefficients in m['coefficients']:
            np.testing.assert_array_equal(coefficients, m['coefficients'][0])
        self.assertTrue(np.isfinite(predict(m, X, p)).all())

    def test_bad_weights_missing_values_and_unknown_pair_rejected(self):
        X, Y, p, w = fixture()
        with self.assertRaises(ValueError): fit(X, Y, p, -w, 50)
        bad = X.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError): fit(bad, Y, p, w, 50)
        m = fit(X, Y, p, w, 50)
        with self.assertRaises(ValueError): predict(m, X[:1], ['unknown'])


if __name__ == '__main__':
    unittest.main()
