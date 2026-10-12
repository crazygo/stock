"""Counterexamples for intraday lookahead, early exits and shadow-ledger selection."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from path_engine import core, build_context, feature_at, exit_label, rule_action
from learning import select_shadow, eligible_dates
from test_contract import market_fixture


class IntradayContracts(unittest.TestCase):
    def setUp(self):
        self.market = market_fixture()
        self.at = pd.Timestamp('2026-05-04 10:05:30', tz=core.ET).tz_convert('UTC')
        self.context = build_context(self.market)
        self.protocol = json.loads((Path(__file__).parent / 'protocol.json').read_text())

    def test_premarket_is_explicit_wait_without_future_opening_range(self):
        at = pd.Timestamp('2026-05-04 09:25:30', tz=core.ET).tz_convert('UTC')
        row, _ = feature_at(self.market, self.context, at)
        self.assertEqual(row['regular_context_ready'], 0)
        self.assertNotIn('x_range_high_distance', row)
        self.assertEqual(rule_action(row, 'range_breakout'), 'CASH')

    def test_future_price_and_volume_cannot_change_current_intraday_features(self):
        before, _ = feature_at(self.market, self.context, self.at)
        for f in self.market.bars.values():
            future = f.end_at > self.at
            f.loc[future, ['open', 'high', 'low', 'close', 'econ_open', 'econ_high', 'econ_low', 'econ_close', 'econ_volume']] *= 12
        after, _ = feature_at(self.market, build_context(self.market), self.at)
        self.assertEqual(before, after)

    def test_invalid_base_volume_is_missing_not_zero_feature(self):
        self.market.daily['SOXL']['volume'] = 0.
        with np.errstate(invalid='ignore', divide='ignore'):
            row, audit = feature_at(self.market, self.context, self.at)
        self.assertIsNone(row)
        self.assertEqual(audit['reason'], 'base_context_not_finite')

    def test_flat_close_has_symmetric_one_day_training_release(self):
        label = exit_label(self.market, self.context, 'SOXL', self.at, 'flat_close')
        self.assertEqual(label['reason'], 'session_close')
        self.assertEqual(label['exit_at'], pd.Timestamp('2026-05-04 16:00', tz=core.ET))
        self.assertEqual(label['label_available_at'], label['exit_at'] + pd.Timedelta(seconds=1))
        self.assertGreater(label['expiry_at'], label['exit_at'])
        self.assertTrue(label['complete_window'])

    def test_stop_before_target_in_same_bar(self):
        f = self.market.bars['SOXL']
        at = pd.Timestamp('2026-05-04 10:15', tz=core.ET)
        f.loc[at, ['high', 'econ_high']] = 106
        f.loc[at, ['low', 'econ_low']] = 97
        label = exit_label(self.market, self.context, 'SOXL', self.at, 'flat_close')
        self.assertEqual(label['reason'], 'risk_stop')
        self.assertLess(label['net_return'], 0)

    def test_trailing_stop_cannot_apply_before_observation_plus_delay(self):
        f = self.market.bars['SOXL']
        f.loc[pd.Timestamp('2026-05-04 10:15', tz=core.ET), ['high', 'econ_high']] = 103
        # The next bar starts at 10:15, earlier than 10:15:31; its dip must not trigger the new stop.
        f.loc[pd.Timestamp('2026-05-04 10:20', tz=core.ET), ['low', 'econ_low']] = 99
        label = exit_label(self.market, self.context, 'SOXL', self.at, 'trail_close')
        self.assertEqual(label['reason'], 'trailing_stop')
        self.assertEqual(label['exit_bar_start'], pd.Timestamp('2026-05-04 10:20', tz=core.ET))
        self.assertEqual(label['exit_at'], pd.Timestamp('2026-05-04 10:25', tz=core.ET))
        self.assertLess(label['net_return'], 0)  # Gap below a raised stop still loses fees/slippage.

    def test_cross_asset_exit_waits_for_joint_available_time_and_next_open(self):
        context = self.context.copy()
        trigger = pd.Timestamp('2026-05-04 10:15', tz=core.ET)
        context.loc[trigger, ['x_session_return', 'x_vwap_distance', 'q_session_return']] = -.01
        label = exit_label(self.market, context, 'SOXL', self.at, 'adaptive_overnight')
        self.assertEqual(label['reason'], 'trend_exit')
        self.assertEqual(label['exit_at'], pd.Timestamp('2026-05-04 10:20', tz=core.ET))
        self.assertGreater(label['label_available_at'], label['exit_at'] + pd.Timedelta(days=5))

    def test_missing_price_prefix_remains_unknown(self):
        self.market.bars['SOXL'] = self.market.bars['SOXL'].drop(pd.Timestamp('2026-05-04 10:20', tz=core.ET))
        label = exit_label(self.market, self.context, 'SOXL', self.at, 'flat_close')
        self.assertEqual(label['reason'], 'missing_prefix')
        self.assertFalse(label['complete_window'])
        self.assertIsNone(label['label_available_at'])

    def test_later_bar_arrival_delays_label_release(self):
        f = self.market.bars['SOXL']
        late = pd.Timestamp('2026-05-05 03:00', tz=core.ET)
        f.loc[pd.Timestamp('2026-05-04 09:40', tz=core.ET), 'available_at'] = late
        label = exit_label(self.market, self.context, 'SOXL', self.at, 'flat_close')
        self.assertEqual(label['label_available_at'], late)

    def test_shadow_selection_cannot_read_current_or_future_marks(self):
        days = pd.bdate_range('2025-01-01', periods=90)
        date = days[75]
        curve = pd.DataFrame({'equity': np.exp(np.arange(90) * .004), 'mark_complete': True}, index=days)
        trades = pd.DataFrame({'entry_date': days[30:75], 'exit_at': [pd.Timestamp(d, tz=core.ET) + pd.Timedelta(hours=16) for d in days[30:75]]})
        books = {'A': (curve, trades), 'B': (curve * 1.00001, trades)}
        first = select_shadow(books, date, self.protocol)
        changed = copy.deepcopy(books)
        for c, t in changed.values():
            c.loc[c.index >= date, 'equity'] *= .01
            t.loc[t.entry_date >= date, 'exit_at'] = pd.Timestamp('2100-01-01', tz=core.ET)
        self.assertEqual(first, select_shadow(changed, date, self.protocol))
        self.assertTrue(all(x['last_mark_date'] < date for x in first[1]))

    def test_unknown_one_clock_label_excludes_whole_training_date(self):
        index = pd.DatetimeIndex([self.at, self.at + pd.Timedelta(minutes=30)])
        features = pd.DataFrame({'date': pd.Timestamp('2026-05-04')}, index=index)
        tables = {s: pd.DataFrame({'label_available_at': pd.Timestamp('2026-05-04 16:00:01', tz=core.ET),
                                  'complete_window': True, 'net_return': .01}, index=index) for s in core.ETFS}
        cutoff = pd.Timestamp('2026-05-05', tz=core.ET)
        self.assertEqual(len(eligible_dates(features, tables, cutoff, self.protocol)), 1)
        tables['SOXS'].loc[index[-1], 'label_available_at'] = pd.NaT
        self.assertEqual(len(eligible_dates(features, tables, cutoff, self.protocol)), 0)


if __name__ == '__main__':
    unittest.main()
