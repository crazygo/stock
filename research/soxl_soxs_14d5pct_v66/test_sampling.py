import unittest
import numpy as np
from sampling import uniforms, integration_statistics


class SamplingTests(unittest.TestCase):
    def test_reproducibility_and_separate_scrambles(self):
        a = uniforms(1061, 5)
        np.testing.assert_array_equal(a, uniforms(1061, 5))
        self.assertFalse(np.array_equal(a, uniforms(1061, 5, replication=1)))
        self.assertTrue(((a >= 0) & (a < 1)).all())
        self.assertEqual(len(np.unique(a, axis=0)), 2048)

    def test_each_coordinate_stratified_in_each_scramble(self):
        for replicate in uniforms(17, 5).reshape(8, 256, 5):
            for dimension in replicate.T:
                np.testing.assert_array_equal(np.sort((dimension*256).astype(int)), np.arange(256))

    def test_invalid_sample_count_rejected(self):
        for count in [8, 100, 2047, 2056]:
            with self.assertRaises(ValueError):
                uniforms(1, 2, count)

    def test_error_uses_replicate_means_not_path_dispersion(self):
        values = np.tile(np.array([[1., -1.], [-1., 1.]]), (1024, 1))
        self.assertGreater(values.std(), 0)
        np.testing.assert_array_equal(integration_statistics(values)['standard_error'], [0., 0.])

    def test_paired_covariance_and_standard_error(self):
        means = np.column_stack([np.arange(8.), -2*np.arange(8.)])
        values = np.repeat(means, 256, axis=0)
        z = integration_statistics(values)
        centered = means - means.mean(axis=0)
        want = centered.T @ centered / 56
        np.testing.assert_allclose(z['mean_covariance'], want)
        np.testing.assert_allclose(np.square(z['standard_error']), np.diag(want))
        self.assertLess(want[0, 1], 0)


if __name__ == '__main__':
    unittest.main()
