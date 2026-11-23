import unittest

from replay import Holding, process_price, signal


class ThresholdTest(unittest.TestCase):
    def test_exact_target_at_5832(self):
        pos = Holding({}, 54., 54. * 1.08, 1., 1.)
        event = process_price(pos, 58.32)
        self.assertEqual(event[0], "target")
        self.assertAlmostEqual(event[1], 58.32)

    def test_stop_rises_and_gap_can_fill_below_it(self):
        pos = Holding({}, 100., 108., 1., 1.)
        self.assertIsNone(process_price(pos, 105.))
        self.assertIsNone(process_price(pos, 104.))
        self.assertEqual(pos.peak, 105.)
        self.assertEqual(process_price(pos, 98., at_open=True), ("stop", 98.))

    def test_three_percent_drawdown_inclusive_inside_a_rising_bar(self):
        result = signal([(98., 100., 97., 99.)])
        self.assertTrue(result["rising"])
        self.assertFalse(result["passed"])
        self.assertTrue(signal([(98., 100., 97.01, 99.)])["passed"])


if __name__ == "__main__":
    unittest.main()
