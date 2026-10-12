"""Causal input, missingness and matched-experiment contracts."""
import unittest
import numpy as np
import pandas as pd
from breadth import PANEL, core, panel_at, freshness, timing_at, variant_columns
from run_breadth import MeanModel, paired_uncertainty


class BreadthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions = pd.bdate_range('2026-01-05', periods=25)
        cls.calendar, parts = {}, []
        for i, d in enumerate(cls.sessions):
            start = d.tz_localize(core.ET) + pd.Timedelta(hours=9, minutes=30)
            end = start + pd.Timedelta(hours=6, minutes=30)
            cls.calendar[d] = {'open_at': start.tz_convert('UTC'), 'close_at': end.tz_convert('UTC')}
            idx = pd.date_range(start, end, freq='5min', inclusive='right').tz_convert('UTC')
            price = 100. + i + np.arange(len(idx)) * .02
            f = pd.DataFrame({'date': d, 'regular': True, 'start_at': idx - pd.Timedelta(minutes=5),
                'end_at': idx, 'available_at': idx + pd.Timedelta(seconds=1), 'econ_open': price,
                'econ_high': price + .1, 'econ_low': price - .1, 'econ_close': price + .01,
                'econ_volume': 100.}, index=idx)
            parts.append(f)
        cls.frame = pd.concat(parts)
        cls.daily = cls.frame.groupby('date').agg(close=('econ_close', 'last'), volume=('econ_volume', 'sum'),
               available_at=('available_at', 'max'))
        cls.daily['complete'] = True
        cls.day = cls.sessions[-2]
        cls.at = (cls.day.tz_localize(core.ET) + pd.Timedelta(hours=10, minutes=35, seconds=30)).tz_convert('UTC')

    def get_panel(self):
        return {s: self.frame.copy() for s in PANEL}, {s: self.daily.copy() for s in PANEL}

    def calc(self, bars, daily):
        return panel_at(bars, daily, self.sessions, self.calendar, self.at)

    def test_future_prices_volume_daily_cannot_change_earlier_features(self):
        bars, daily = self.get_panel(); before, _ = self.calc(bars, daily)
        self.assertIsNotNone(before)
        for s in PANEL:
            bars[s].loc[bars[s].index > self.at, ['econ_open', 'econ_high', 'econ_low', 'econ_close', 'econ_volume']] *= 50
            daily[s].loc[daily[s].index >= self.day, ['close', 'volume']] *= 20
        after, _ = self.calc(bars, daily)
        self.assertEqual(before, after)

    def test_one_missing_member_regular_bar_causes_abstention(self):
        bars, daily = self.get_panel()
        idx = bars['NVDA'][(bars['NVDA'].date == self.day)].index[1]
        bars['NVDA'] = bars['NVDA'].drop(idx)
        result, audit = self.calc(bars, daily)
        self.assertIsNone(result)
        self.assertEqual(audit['reason'], 'NVDA:regular_prefix_incomplete')

    def test_late_old_bar_cannot_be_replaced_by_fresh_latest_bar(self):
        bars, daily = self.get_panel()
        idx = bars['AMD'][(bars['AMD'].date == self.day)].index[1]
        bars['AMD'].loc[idx, 'available_at'] = self.at + pd.Timedelta(days=1)
        result, audit = self.calc(bars, daily)
        self.assertIsNone(result)
        self.assertEqual(audit['reason'], 'AMD:regular_prefix_incomplete')

    def test_history_must_have_complete_official_lookback(self):
        bars, daily = self.get_panel()
        daily['MU'].loc[self.sessions[-4], 'complete'] = False
        self.assertIsNone(self.calc(bars, daily)[0])
        daily['MU'].loc[self.sessions[-4], 'complete'] = True
        daily['MU'].loc[self.sessions[-4], 'available_at'] = self.at + pd.Timedelta(days=1)
        self.assertIsNone(self.calc(bars, daily)[0])

    def test_timing_failure_expires_and_future_extrema_do_not_rewrite_it(self):
        f = self.frame[self.frame.date == self.day].copy()
        f[['econ_open', 'econ_high', 'econ_low', 'econ_close']] = [100., 101., 99., 100.]
        f.iloc[6, f.columns.get_loc('econ_close')] = 102.
        a = freshness(f, self.calendar)
        self.assertEqual(a.high_fresh_failure.iloc[2], 1.)  # index7, five minutes later
        self.assertEqual(a.high_fresh_failure.iloc[5], 0.)  # index10, twenty minutes later
        f.iloc[20:, f.columns.get_loc('econ_high')] = 900.
        b = freshness(f, self.calendar)
        pd.testing.assert_frame_equal(a.iloc[:15], b.iloc[:15])

    def test_timing_waits_for_first30m_and_delayed_prefix(self):
        f = self.frame[self.frame.date == self.day].copy()
        t = freshness(f, self.calendar)
        early = f.index[3] + pd.Timedelta(seconds=30)
        self.assertIsNone(timing_at({'SOXX': t}, early)[0])
        f.iloc[2, f.columns.get_loc('available_at')] = self.at + pd.Timedelta(days=1)
        delayed = freshness(f, self.calendar)
        self.assertIsNone(timing_at({'SOXX': delayed}, self.at)[0])

    def test_base_ablation_cannot_read_new_columns(self):
        f = pd.DataFrame(columns=['date','clock','SOXX_ret5','x_recent15','timing_SOXX_high_age',
                                 'leader_NVDA_today','breadth_today_mean'])
        self.assertEqual(variant_columns(f,'base'), ['SOXX_ret5','x_recent15'])
        self.assertEqual(variant_columns(f,'timing'), ['SOXX_ret5','x_recent15','timing_SOXX_high_age'])
        self.assertEqual(len(variant_columns(f,'breadth')), 5)

    def test_ensemble_fixed_equal_weight(self):
        class Fake:
            def __init__(self, x): self.x = x
            def predict(self, rows): return np.full(len(rows), self.x)
        np.testing.assert_allclose(MeanModel([Fake(.06),Fake(-.02)]).predict([1,2]), [.02,.02])

    def test_paired_bootstrap_uses_same_dates(self):
        book = pd.DataFrame({'equity': [1.,1.1,1.05,1.15]})
        result = paired_uncertainty(book, book.copy())
        self.assertEqual(result['interval95'], [0.,0.])
        self.assertFalse(result['positive_increment_supported'])
        with self.assertRaises(AssertionError):
            paired_uncertainty(book, book.iloc[1:])


if __name__ == '__main__':
    unittest.main()
