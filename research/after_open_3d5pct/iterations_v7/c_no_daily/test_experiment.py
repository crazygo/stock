import unittest

import torch

from .experiment import relative_coordinates, shape_features


def bar(open_log, close_log, valid=True):
    x = torch.zeros(14)
    x[0] = close_log-open_log
    x[12] = open_log/5
    x[10] = int(valid)
    return x


class NoDailyBoundaryTests(unittest.TestCase):
    def test_relative_origin_and_invalid_tail(self):
        x = torch.stack([bar(4.0, 4.02), bar(4.03, 4.04), bar(10, 11, False)])[None]
        q = relative_coordinates(x)
        self.assertAlmostEqual(q[0, 0, 12].item(), 0, places=5)
        self.assertAlmostEqual(q[0, 1, 12].item(), .3, places=5)
        self.assertAlmostEqual(q[0, 1, 0].item(), .5, places=5)
        self.assertEqual(q[0, 2, 12].item(), 0)
        self.assertEqual(q[0, 2, 0].item(), 0)

    def test_gap_is_in_close_path(self):
        x = torch.stack([bar(4.0, 4.02), bar(4.05, 4.05)])[None]
        v = shape_features(x)
        # First intrabar move .02 plus next gap .03; channel 3 alone is not the path.
        self.assertAlmostEqual(v[0, :3].sum().item(), 5, places=3)

    def test_same_endpoints_different_path(self):
        a = torch.stack([bar(4.0, 4.03), bar(4.03, 4.00), bar(4.00, 4.01)])[None]
        b = torch.stack([bar(4.0, 3.98), bar(3.98, 4.03), bar(4.03, 4.01)])[None]
        self.assertAlmostEqual(shape_features(a)[0, :3].sum().item(),
                               shape_features(b)[0, :3].sum().item(), places=3)
        self.assertFalse(torch.allclose(shape_features(a), shape_features(b)))

    def test_masked_tail_cannot_change_shape(self):
        prefix = torch.stack([bar(4.0, 4.02), bar(4.02, 4.03)])[None]
        extended = torch.cat((prefix, torch.stack([bar(7, 8, False), bar(3, 1, False)])[None]), 1)
        self.assertTrue(torch.allclose(shape_features(prefix), shape_features(extended), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
