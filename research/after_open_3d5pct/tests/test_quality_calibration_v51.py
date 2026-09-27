import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.quality_calibration_v51 import choose_inner_plan, probability_variant


class CalibrationIterationTests(unittest.TestCase):
    def source(self):
        rows = []
        for phase in ['selection', 'outer']:
            for candidate in ['A', 'B']:
                for i, target in enumerate([0, 1, 0, 1]):
                    raw = [.1, .9, .1, .9][i] if candidate == 'A' else .5
                    rows.append(dict(phase=phase, candidate=candidate, sample_id=f'{phase}-{i}',
                                     target=target, quality_eligible=True, p_raw=raw, p_touch=.5))
        return pd.DataFrame(rows)

    def test_outer_outcome_and_scores_cannot_change_selection(self):
        source = self.source()
        chosen, _ = choose_inner_plan(source, ['A', 'B'])
        self.assertEqual((chosen['candidate'], chosen['method']), ('A', 'identity'))
        changed = source.copy()
        ix = changed.phase == 'outer'
        changed.loc[ix, 'target'] = 1-changed.loc[ix, 'target']
        changed.loc[ix, ['p_raw', 'p_touch']] = np.nan
        self.assertEqual(chosen, choose_inner_plan(changed, ['A', 'B'])[0])

    def test_identity_recomputes_expected_return_and_recommendation(self):
        f = pd.DataFrame([dict(symbol='X', session_date='2026-01-01', cutoff_et='10:30',
            quality_eligible=True, p_raw=.2, p_touch=.8, failure_mean_gross=-.04,
            failure_q10_net=-.08, expected_net=.999, selected_quality_scenario=True)])
        c = dict(touch_gross_return=.05, buy_cost_rate=.0006, sell_cost_rate=.0006,
            development_scenario=dict(opportunity_budget_fraction_max_each_quality_date_hour=.05,
                candidate_calibrated_p_touch_min=.6, predicted_expected_net_min=0.,
                predicted_failure_q10_net_min=-.15))
        result = probability_variant(f, 'identity', c)
        self.assertLess(result.expected_net.iloc[0], 0)
        self.assertFalse(result.selected_quality_scenario.iloc[0])
        self.assertEqual(result.probability_estimate_kind.iloc[0], 'raw_uncalibrated_model_estimate')
        self.assertEqual(f.p_touch.iloc[0], .8)
        self.assertTrue(f.selected_quality_scenario.iloc[0])

    def test_population_mismatch_is_rejected(self):
        source = self.source()
        source = source[~((source.phase == 'selection') & (source.candidate == 'B') &
                          (source.sample_id == 'selection-0'))]
        with self.assertRaisesRegex(ValueError, 'different evaluation populations'):
            choose_inner_plan(source, ['A', 'B'])


if __name__ == '__main__':
    unittest.main()
