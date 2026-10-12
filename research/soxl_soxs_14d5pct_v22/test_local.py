import unittest
import numpy as np
import pandas as pd
from local_training import similarity_weights, state_calibration, forecast_routes, day_cutoff, STATE_COLUMNS
import test_early


class LocalTrainingTests(unittest.TestCase):
    def panel(self):
        return test_early.EarlyTrainingTests().panel(210)

    def test_state_distance_does_not_use_outcomes_or_future_rows(self):
        p = self.panel(); pool = p.iloc[:120]; current = p.iloc[150]
        w, _ = similarity_weights(pool, current, True)
        changed = pool.copy(); changed[['C', 'D', 'M']] = 1e9
        w2, _ = similarity_weights(changed, current, True)
        np.testing.assert_allclose(w, w2, atol=1e-12, rtol=1e-10)
        self.assertAlmostEqual(sum(w), 120.)
        other = current.copy(); other[STATE_COLUMNS] = -other[STATE_COLUMNS]
        w3, _ = similarity_weights(pool, other, True)
        self.assertFalse(np.allclose(w, w3))

    def test_local_calibration_with_constant_states_matches_global_formula(self):
        p = self.panel(); p[STATE_COLUMNS] = 0.; ids = p.index[:100]; at = p.index[150]
        history = [{'date': p.loc[t, 'date'], 'at': t, 'raw': {'C': i*.001, 'D': i*.003}} for i, t in enumerate(ids)]
        cal = state_calibration(history, p, list(p.date.iloc[:120]), at)
        self.assertTrue(cal['ready']); self.assertAlmostEqual(cal['weights']['effective_dates'], 100.)
        for target in ['C', 'D']:
            x = np.array([h['raw'][target] for h in history]); y = p.loc[ids, target].to_numpy()
            z = (x-x.mean())/x.std(); coefficient = z@(y-y.mean())/(z@z+20)
            self.assertAlmostEqual(cal['maps'][target]['coefficient'], coefficient)

    def test_future_truth_and_raw_scores_cannot_change_current_forecast(self):
        p = self.panel(); at = p.index[180]
        histories = {k: [{'at': t, 'date': r.date, 'raw': {'C': .001*i, 'D': .002*i}}
                         for i, (t, r) in enumerate(p.iterrows())] for k in ['state', 'global']}
        _, before, raw = forecast_routes(p, at, histories)
        changed = p.copy(); changed.loc[changed.index >= at, ['C', 'D', 'M']] = 999.
        altered = {k: [dict(e, raw={'C': -1e9, 'D': 1e9}) if e['at'] >= at else e for e in h] for k, h in histories.items()}
        _, after, raw_after = forecast_routes(changed, at, altered)
        for mode in raw:
            for k in raw[mode]:
                np.testing.assert_allclose(raw[mode][k], raw_after[mode][k], atol=1e-12, rtol=1e-10)
        for n in before['forecasts']:
            for head in ['C', 'D']:
                for field in ['center', 'scale', 'intercept', 'coefficient']:
                    np.testing.assert_allclose(before['forecasts'][n]['calibration']['maps'][head][field],
                                               after['forecasts'][n]['calibration']['maps'][head][field],
                                               atol=1e-12, rtol=1e-10)

    def test_pending_dates_cannot_enter_local_models(self):
        p = self.panel(); at = p.index[150]
        p.loc[p.index[140], 'terminal_release'] = day_cutoff(at)
        _, record, _ = forecast_routes(p, at, {'state': [], 'global': []})
        self.assertEqual(record['fits']['state']['members'], list(p.index[:140]))
        self.assertEqual(record['fits']['global']['members'], list(p.index[:140]))

    def test_missing_current_state_and_small_history_abstain(self):
        p = self.panel(); p.loc[p.index[180], 'feature_ready'] = False
        models, record, _ = forecast_routes(p, p.index[180], {'state': [], 'global': []})
        self.assertIsNone(models['state'])
        self.assertTrue(all(not v['event']['ready'] for v in record['forecasts'].values()))
        models, _, _ = forecast_routes(p, p.index[90], {'state': [], 'global': []})
        self.assertIsNone(models['state']); self.assertIsNone(models['global'])

    def test_too_concentrated_base_fit_abstains(self):
        p = self.panel(); at = p.index[180]
        p[STATE_COLUMNS] = 100.
        p.loc[p.index[:40], STATE_COLUMNS] = 0.
        p.loc[at, STATE_COLUMNS] = 0.
        models, record, _ = forecast_routes(p, at, {'state': [], 'global': []})
        self.assertLess(record['fits']['state']['weights']['effective_dates'], 60)
        self.assertIsNone(models['state']); self.assertIsNotNone(models['global'])

    def test_independent_weighted_model_equations(self):
        from verify_local import state_model_equations
        p = self.panel(); at = p.index[180]
        models, record, raw = forecast_routes(p, at, {'state': [], 'global': []})
        info = record['fits']['state']
        expected, checks = state_model_equations(p, info['members'], at, models['state'], info)
        self.assertEqual(checks, 3)
        for k in raw['state']:
            np.testing.assert_allclose(raw['state'][k], expected[k], atol=1e-12, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
