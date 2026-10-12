import math
import unittest
from price_paths import centered_series


def chart(closes):
    dates = [f'{i:03d}' for i in range(1, 61)]
    return {'dates': dates, 'baseline_date': '000',
            'bars': [[d, v, v, v, v] for d, v in zip(dates, closes) if v is not None]}


class CenterTests(unittest.TestCase):
    def test_price_reference_uses_31_closes_and_is_scale_invariant(self):
        values = [100 * math.exp(i / 100) for i in range(1, 61)]
        a = centered_series(chart(values))
        b = centered_series(chart([v * 7 for v in values]))
        self.assertEqual(a['status'], 'complete')
        self.assertEqual(a['center_dates'], [f'{i:03d}' for i in range(30, 61)])
        self.assertAlmostEqual(a['price_reference'], 100 * math.exp(.45))
        self.assertAlmostEqual(sum(p[2] for p in a['points'][-31:]), 0)
        self.assertAlmostEqual(a['daily_return_reference'], math.exp(.01) - 1)
        self.assertTrue(all(abs(p[4]) < 1e-12 for p in a['points']))
        for p, q in zip(a['points'], b['points']):
            self.assertAlmostEqual(p[2], q[2])

    def test_daily_center_is_arithmetic_mean_not_net_window_return(self):
        values = [100] * 60
        values[-2], values[-1] = 110, 100
        result = centered_series(chart(values))
        mean = (.1 + 100 / 110 - 1) / 30
        self.assertAlmostEqual(result['daily_return_reference'], mean)
        self.assertNotEqual(mean, 0)
        self.assertAlmostEqual(result['points'][-1][4], 100 * math.log((100/110)/(1+mean)))

    def test_early_gap_breaks_price_and_next_daily_return_but_keeps_center(self):
        values = [100] * 60
        values[4] = None
        result = centered_series(chart(values))
        self.assertEqual(result['status'], 'complete')
        self.assertIsNone(result['points'][4][2])
        self.assertIsNone(result['points'][4][4])
        self.assertIsNone(result['points'][5][4])
        self.assertAlmostEqual(result['points'][5][2], 0)
        values[-15] = None
        result = centered_series(chart(values))
        self.assertEqual(result['status'], 'missing_center')
        self.assertTrue(all(p[2] is None and p[4] is None for p in result['points']))

    def test_short_or_invalid_center_is_never_filled(self):
        fixture = chart([100] * 60)
        fixture['dates'] = fixture['dates'][-20:]
        self.assertEqual(centered_series(fixture)['status'], 'missing_center')
        fixture = chart([100] * 59 + [0])
        self.assertEqual(centered_series(fixture)['status'], 'missing_center')


if __name__ == '__main__':
    unittest.main()
