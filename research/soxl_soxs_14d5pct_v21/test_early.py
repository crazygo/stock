import unittest
import numpy as np
import pandas as pd
from early_training import (core, training_dates, forecast_routes, capital_decision, day_cutoff,
                            BASE_COLUMNS, REGIME_COLUMNS)


class EarlyTrainingTests(unittest.TestCase):
    def panel(self, n=190):
        days = pd.date_range('2025-01-01', periods=n)
        at = (days.tz_localize(core.ET) + pd.Timedelta(hours=9, minutes=25, seconds=30)).tz_convert('UTC')
        rng = np.random.default_rng(21)
        frame = pd.DataFrame(rng.normal(size=(n, 19)), index=at, columns=BASE_COLUMNS+REGIME_COLUMNS)
        frame['date'] = days; frame['feature_ready'] = True
        frame['C'] = rng.normal(0, .01, n); frame['D'] = rng.normal(0, .04, n)
        frame['M'] = abs(frame.D)
        frame['terminal_release'] = at + pd.Timedelta(hours=8)
        frame['release'] = at + pd.Timedelta(days=13, hours=8)
        return frame

    def test_no_later_completed_date_skips_pending(self):
        p = self.panel(8); p.loc[p.index[2], 'terminal_release'] = pd.NaT
        dates, meta = training_dates(p, day_cutoff(p.index[-1]))
        self.assertEqual(dates['early'], list(p.date.iloc[:2]))
        self.assertEqual(meta['known_but_waiting'], 4)
        self.assertFalse(meta['expired_unresolved_prefix'])
        _, stale = training_dates(p, p.index[-1] + pd.Timedelta(days=20))
        self.assertTrue(stale['expired_unresolved_prefix'])

    def test_release_equal_cutoff_not_admitted(self):
        p = self.panel(5); cutoff = day_cutoff(p.index[-1])
        p.loc[p.index[1], 'terminal_release'] = cutoff
        before, _ = training_dates(p, cutoff)
        after, _ = training_dates(p, cutoff + pd.Timedelta(seconds=1))
        self.assertEqual(len(before['early']), 1)
        self.assertEqual(len(after['early']), 4)

    def test_unavailable_truth_cannot_change_fit_or_raw_forecast(self):
        p = self.panel(); at = p.index[160]
        p.loc[p.index[145], 'terminal_release'] = pd.NaT
        old = {'raw': {'C': .01, 'D': .01, 'M': .02, 'q': .75, 'SOXL': .02, 'SOXS': 0.}}
        _, before, raw = forecast_routes(p, at, old, [], [])
        changed = p.copy(); changed.loc[changed.index[145:], ['C', 'D', 'M']] = 999.
        _, after, raw_after = forecast_routes(changed, at, old, [], [])
        for k in raw:
            np.testing.assert_allclose(raw[k], raw_after[k], atol=1e-12, rtol=1e-10)
        self.assertEqual(before['fit']['members'], after['fit']['members'])
        for k in before['fit']['heads']:
            for parameter in ['coefficients', 'intercept']:
                np.testing.assert_allclose(before['fit']['heads'][k][parameter],
                                           after['fit']['heads'][k][parameter], atol=1e-12, rtol=1e-10)

    def test_current_or_future_raw_predictions_cannot_calibrate_today(self):
        p = self.panel(); at = p.index[180]
        history = [{'at': t, 'date': r.date, 'raw': {'C': .001*i, 'D': .002*i}}
                   for i, (t, r) in enumerate(p.iterrows())]
        old = {'raw': {'C': .01, 'D': .01, 'M': .02, 'q': .75, 'SOXL': .02, 'SOXS': 0.}}
        _, a, _ = forecast_routes(p, at, old, history, history)
        altered = [dict(e, raw={'C': 1e9, 'D': -1e9}) if e['at'] >= at else e for e in history]
        _, b, _ = forecast_routes(p, at, old, altered, altered)
        self.assertEqual(a, b)

    def test_original_risk_coordinates_required(self):
        empty = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []]))
        at = pd.Timestamp('2026-01-05', tz='UTC')
        d = capital_decision(empty, at, at, {'ready': True, 'top_symbol': 'SOXL'})
        self.assertEqual(d['action'], 'CASH')
        self.assertEqual(d['reason'], 'original_risk_coordinates_unavailable')


if __name__ == '__main__':
    unittest.main()
