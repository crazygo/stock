"""Future-data exclusion, causal routing and entry-frozen risk contracts."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experts import (core, EXPERTS, RISKS, entry_rule, fit_predict, corrected_choice,
                     select_expert, eligible_index, day_cutoff)
from run_experts import score


class ExpertContracts(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((Path(__file__).parent / 'protocol.json').read_text())
        dates = pd.bdate_range('2025-01-02', periods=240)
        idx = pd.DatetimeIndex([pd.Timestamp(str(d.date()) + ' 09:25:30', tz=core.ET).tz_convert('UTC') for d in dates])
        self.features = pd.DataFrame({'date': dates, 'clock': '09:25:30', 'SOXX_ret20': np.sin(np.arange(240) / 9),
                      'QQQ_ret5': np.cos(np.arange(240) / 13), 'SOXL_atr14': .04, 'SOXS_atr14': .1}, index=idx)
        self.outcomes = {risk: {symbol: pd.DataFrame({'complete_window': True, 'label_available_at': idx + pd.Timedelta(days=14),
                        'net_return': .025 + .01 * np.sin(np.arange(240) / 6), 'held_sessions': 3}, index=idx)
                                  for symbol in core.ETFS} for risk in RISKS}
        self.histories = {expert: pd.DataFrame({'action': 'SOXL', 'SOXL_raw': .02, 'SOXS_raw': .02}, index=idx) for expert in EXPERTS}
        self.at = idx[200]

    def test_risk_width_is_known_at_entry_and_bounded(self):
        row = self.features.iloc[0]
        self.assertIsNone(entry_rule(row, 'SOXL', 'expiry14'))
        self.assertEqual(entry_rule(row, 'SOXL', 'atr_stop')['stop_fraction'], .08)
        self.assertEqual(entry_rule(row, 'SOXS', 'atr_protect')['stop_fraction'], .18)
        self.assertEqual(entry_rule(row, 'SOXS', 'atr_protect')['breakeven_trigger'], .02)

    def test_future_results_cannot_change_router(self):
        cutoff = day_cutoff(self.at)
        original = select_expert(self.histories, self.outcomes, self.features, cutoff, self.protocol)
        changed = copy.deepcopy(self.outcomes)
        for tables in changed.values():
            for table in tables.values():
                table.loc[table.label_available_at >= cutoff, 'net_return'] = 900
        after = select_expert(self.histories, changed, self.features, cutoff, self.protocol)
        self.assertEqual(original, after)
        self.assertTrue(all(t['last_label_available'] < cutoff for t in original[1]))

    def test_later_known_expert_failure_changes_selection(self):
        cutoff = day_cutoff(self.at)
        first, _ = select_expert(self.histories, self.outcomes, self.features, cutoff, self.protocol)
        self.histories[first]['action'] = 'SOXS'
        risk = first.split('__')[1]
        self.outcomes[risk]['SOXS']['net_return'] = -.1
        second, trials = select_expert(self.histories, self.outcomes, self.features, cutoff, self.protocol)
        self.assertNotEqual(first, second)
        self.assertFalse(next(x for x in trials if x['expert'] == first)['qualified'])

    def test_all_negative_experts_cause_cash(self):
        for tables in self.outcomes.values():
            for table in tables.values(): table['net_return'] = -.02
        selected, _ = select_expert(self.histories, self.outcomes, self.features, day_cutoff(self.at), self.protocol)
        self.assertIsNone(selected)

    def test_warmup_nan_forecasts_are_not_calibration_samples(self):
        table = self.outcomes['expiry14']
        eligible = eligible_index(self.features, table, day_cutoff(self.at))
        history = self.histories[EXPERTS[0]].copy()
        history[['SOXL_raw', 'SOXS_raw']] = np.nan
        history.loc[eligible[-5:], ['SOXL_raw', 'SOXS_raw']] = .01
        action, detail = corrected_choice(np.array([.03, .01]), history, table, eligible, self.protocol)
        self.assertEqual(detail['calibration_dates'], 5)
        self.assertEqual(action, 'CASH')

    def test_unknown_label_cannot_enter_training_or_calibration(self):
        table = self.outcomes['expiry14']
        eligible = eligible_index(self.features, table, day_cutoff(self.at))
        lost = eligible[-1]
        table['SOXL'].loc[lost, 'label_available_at'] = pd.NaT
        self.assertNotIn(lost, eligible_index(self.features, table, day_cutoff(self.at)))

    def test_future_data_cannot_change_any_algorithm_prediction(self):
        table = self.outcomes['expiry14']
        train = eligible_index(self.features, table, day_cutoff(self.at))[-120:]
        with threadpool_limits(limits=1):
            before, _ = fit_predict(self.features, train, table, self.features.loc[self.at])
            newer = self.features.copy()
            newer.loc[newer.index > self.at, 'SOXX_ret20'] = 999
            other = copy.deepcopy(table)
            for t in other.values(): t.loc[t.index > self.at, 'net_return'] = 999
            after, _ = fit_predict(newer, train, other, self.features.loc[self.at])
        for algorithm in before:
            np.testing.assert_array_equal(before[algorithm], after[algorithm])

    def test_pending_future_stop_is_not_counted_as_completed_stop(self):
        equity = pd.DataFrame({'equity': [1.], 'mark_complete': True, 'symbol': 'SOXL'}, index=pd.to_datetime(['2026-09-30']))
        trades = pd.DataFrame({'entry_date': pd.to_datetime(['2026-09-29']), 'exit_at': [pd.Timestamp('2026-10-01', tz=core.ET)],
                               'reason': ['risk_stop'], 'net_return': [-.06]})
        result = score((equity, trades), ['2026-09'])
        self.assertEqual(result['risk_stop_exits'], 0)
        self.assertEqual(result['pending_trades'], 1)


if __name__ == '__main__':
    unittest.main()
