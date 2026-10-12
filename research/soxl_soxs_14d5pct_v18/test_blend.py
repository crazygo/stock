import unittest
import numpy as np
import pandas as pd
from adaptive_blend import blend_head, matured_pool, forecast, STATE_COLUMNS


class BlendTests(unittest.TestCase):
    def test_closed_form_is_bounded_convex_minimum(self):
        raw = np.array([-.04, .01, .02, -.03]); cal = -raw
        y = .3 * raw + .7 * cal; w = np.array([.1, .2, .3, .4])
        fit = blend_head(raw, cal, y, w)
        grid = np.linspace(0, 1, 10001)
        objectives = ((y - raw - grid[:, None] * (cal - raw)) ** 2) @ w
        objectives += fit['penalty'] * (grid - .5) ** 2
        self.assertLessEqual(fit['objective'], objectives.min() + 1e-14)
        self.assertLess(abs(fit['alpha_calibrated'] - grid[objectives.argmin()]), .0001)

    def test_degenerate_heads_and_endpoint(self):
        x = np.array([-.01, .02]); w = np.array([.5, .5])
        self.assertEqual(blend_head(x, x, 3*x, w)['alpha_calibrated'], .5)
        self.assertEqual(blend_head(x, -x, 4*x, w)['alpha_calibrated'], 0.)
        self.assertEqual(blend_head(x, -x, -4*x, w)['alpha_calibrated'], 1.)

    def test_controls_and_unready_label_exclusion(self):
        x = np.array([-.01, .02]); w = np.array([.5, .5])
        self.assertEqual(blend_head(x, -x, -x, w, 'calibrated')['alpha_calibrated'], 1.)
        self.assertEqual(blend_head(x, -x, -x, w, 'raw')['alpha_calibrated'], 0.)
        self.assertEqual(blend_head(x, -x, -x, w, 'hard')['alpha_calibrated'], 1.)
        index = pd.date_range('2025-01-01 14:25:30', periods=3, tz='UTC')
        f = pd.DataFrame(1., index=index, columns=[*STATE_COLUMNS, 'raw_C', 'raw_D', 'cal_C', 'cal_D', 'actual_C', 'actual_D'])
        f['date'] = pd.date_range('2025-01-01', periods=3)
        f['release'] = pd.to_datetime(['2025-01-15', '2025-01-16', '2025-01-17'], utc=True)
        cutoff = pd.Timestamp('2025-01-16', tz='UTC')
        self.assertEqual(list(matured_pool(f, cutoff).index), [index[0]])

    def test_current_future_outcome_does_not_change_blend(self):
        index = pd.date_range('2025-01-01', periods=60, tz='UTC')
        f = pd.DataFrame(0., index=index, columns=STATE_COLUMNS)
        f['date'] = index.tz_localize(None)
        for k in ['C', 'D']:
            f['raw_' + k] = np.linspace(-.02, .02, 60)
            f['cal_' + k] = -f['raw_' + k]
            f['actual_' + k] = .4 * f['raw_' + k]
        current = f.iloc[-1].copy()
        e = {'ready': True, 'raw': {'C': -.001, 'D': .02}, 'adjusted': {'C': .001, 'D': -.01}}
        before = forecast(f, current, e, 'state', 'blend')
        current['actual_C'] = 1e9; current['actual_D'] = -1e9
        after = forecast(f, current, e, 'state', 'blend')
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
