import unittest

import numpy as np

from .experiment import curve_features, group_relative_features


class WindowBoundaryTest(unittest.TestCase):
    def test_regular_prefix_excludes_previous_gap(self):
        x = np.zeros((2, 8, 192, 14), np.float32)
        x[:, 7, 66:90, 10] = 1
        x[:, 7, 66:90, 0] = 0.001
        x[1, 7, 66, 3] = 0.2  # Gap from prior premarket bar into 09:30.
        x[1, 7, 65, 0] = -0.3  # Out-of-window price movement.
        group = np.zeros((2, 6, 6, 10), np.float32)
        group[:, -1, 0, 9] = 1
        group[:, -1, 0, 3] = 0.01
        data = {"x5": x, "group_seq": group}
        curve = curve_features(data)
        relative = group_relative_features(data)
        np.testing.assert_allclose(curve.iloc[0].to_numpy(), curve.iloc[1].to_numpy())
        np.testing.assert_allclose(relative.iloc[0, 0], relative.iloc[1, 0])
        self.assertAlmostEqual(curve.loc[0, "open40_ret"], 0.008, places=7)
        self.assertAlmostEqual(relative.loc[0, "own_minus_trend15_cutoff_return"], 0.014, places=7)


if __name__ == "__main__":
    unittest.main()
