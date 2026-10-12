"""Causal daily learning and immutable prequential calibration contracts."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from walkforward import core, day_cutoff, eligible_index, fit_day, forecast
from daily_study import score


class DailyLearningTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((Path(__file__).parent / 'protocol.json').read_text())
        self.protocol.update(training_window_dates=80, minimum_training_dates=30,
                             calibration_window_dates=20, minimum_calibration_dates=10)
        self.protocol['forest']['n_estimators'] = 8
        dates = pd.bdate_range('2025-01-01', periods=180)
        index = pd.DatetimeIndex([pd.Timestamp(str(d.date()) + ' 09:25:30', tz=core.ET).tz_convert('UTC') for d in dates])
        self.features = pd.DataFrame({'date': dates, 'clock': '09:25:30',
                                     'x': np.sin(np.arange(180) / 7), 'z': np.arange(180) % 3}, index=index)
        self.tables = {s: pd.DataFrame({'complete_window': True, 'net_return': np.sin(np.arange(180) / 8 + n) * .04,
                      'label_available_at': index + pd.Timedelta(days=14)}, index=index) for n, s in enumerate(core.ETFS)}
        self.history = pd.DataFrame({'forecast_at': index, 'training_cutoff': [day_cutoff(at) for at in index],
                                    'SOXL_raw_ev': .002, 'SOXS_raw_ev': .001}, index=index)
        self.at = index[130]

    def fit(self, tables=None, features=None, history=None, at=None):
        return fit_day(features if features is not None else self.features,
                       tables if tables is not None else self.tables,
                       history if history is not None else self.history,
                       at if at is not None else self.at, self.protocol)

    def test_future_labels_features_and_forecasts_cannot_change_past_model(self):
        first, record = self.fit()
        tables = copy.deepcopy(self.tables)
        features = self.features.copy(); history = self.history.copy()
        for table in tables.values():
            unknown = table.label_available_at >= day_cutoff(self.at)
            table.loc[unknown, 'net_return'] = 999
        features.loc[features.index >= day_cutoff(self.at), 'x'] = 999
        history.loc[history.index >= day_cutoff(self.at), 'SOXL_raw_ev'] = 999
        second, second_record = self.fit(tables, features, history)
        self.assertEqual(record['model_id'], second_record['model_id'])
        self.assertEqual(forecast(first, self.features.loc[self.at], self.at, self.protocol),
                         forecast(second, self.features.loc[self.at], self.at, self.protocol))

    def test_new_mature_labels_update_later_model_and_keep_old_forecasts(self):
        history = self.history.copy(deep=True)
        _, early = self.fit()
        _, later = self.fit(at=self.features.index[155])
        self.assertGreater(later['train_last'], early['train_last'])
        self.assertNotEqual(later['model_id'], early['model_id'])
        self.assertEqual(later['training_dates'], 80)
        pd.testing.assert_frame_equal(history, self.history)

    def test_matured_labels_can_change_new_model(self):
        _, first = self.fit()
        tables = copy.deepcopy(self.tables)
        tables['SOXL'].loc[first['train_last'], 'net_return'] += .1
        _, second = self.fit(tables=tables)
        self.assertNotEqual(first['model_id'], second['model_id'])

    def test_unknown_and_exact_cutoff_are_excluded_symmetrically(self):
        cutoff = day_cutoff(self.at)
        eligible = eligible_index(self.features, self.tables, cutoff)
        last = eligible[-1]
        tables = copy.deepcopy(self.tables)
        tables['SOXL'].loc[last, 'label_available_at'] = cutoff
        self.assertNotIn(last, eligible_index(self.features, tables, cutoff))
        tables['SOXL'].loc[last, 'label_available_at'] = pd.NaT
        self.assertNotIn(last, eligible_index(self.features, tables, cutoff))
        tables = copy.deepcopy(self.tables)
        tables['SOXS'].loc[last, 'complete_window'] = False
        self.assertNotIn(last, eligible_index(self.features, tables, cutoff))

    def test_insufficient_calibration_abstains_but_emits_raw_forecast(self):
        bundle, record = self.fit(history=self.history.iloc[:0])
        result = forecast(bundle, self.features.loc[self.at], self.at, self.protocol)
        self.assertEqual(record['status'], 'insufficient_calibration')
        self.assertEqual(result['action'], 'CASH')
        self.assertTrue(np.isfinite(result['SOXL_raw_ev']))

    def test_revised_past_forecast_is_rejected_from_calibration(self):
        history = self.history.copy()
        history['forecast_at'] = self.at
        bundle, record = self.fit(history=history)
        self.assertEqual(record['calibration_dates'], 0)
        self.assertFalse(bundle['ready'])

    def test_future_model_cannot_rewrite_earlier_prediction(self):
        bundle, _ = self.fit()
        with self.assertRaises(ValueError):
            forecast(bundle, self.features.iloc[20], self.features.index[20], self.protocol)

    def test_intraday_analysis_reuses_same_day_training_boundary(self):
        _, morning = self.fit()
        _, afternoon = self.fit(at=self.at + pd.Timedelta(hours=5))
        self.assertEqual(morning['model_id'], afternoon['model_id'])

    def test_case_months_do_not_select_features_or_training_samples(self):
        _, record = self.fit()
        self.assertNotIn('date', record['feature_columns'])
        self.assertNotIn('clock', record['feature_columns'])
        self.assertTrue(all(at < day_cutoff(self.at) for at in record['train_members']))

    def test_cash_and_positive_mean_with_negative_month_fail(self):
        index = pd.to_datetime(['2026-05-29', '2026-07-31', '2026-09-30'])
        months = ['2026-05', '2026-07', '2026-09']
        curve = pd.DataFrame({'equity': [1., 1., 1.], 'mark_complete': True, 'symbol': 'CASH'}, index=index)
        self.assertFalse(score((curve, pd.DataFrame()), months)['primary_pass'])
        curve.equity = [1.4, 1.4 * 1.2, 1.4 * 1.2 * .9]
        result = score((curve, pd.DataFrame()), months)
        self.assertTrue(result['arithmetic_gate'])
        self.assertFalse(result['primary_pass'])


if __name__ == '__main__':
    unittest.main()
