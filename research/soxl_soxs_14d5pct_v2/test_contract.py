"""Counterexample-driven tests for causal timing, accounting and the user gate."""
import sys
from pathlib import Path
import unittest
import tempfile
import json

import numpy as np
import pandas as pd

import core
from ledger import replay, statistics
from position import manage

sys.path.insert(0, str(core.ROOT))
from scripts.model_history_calendar import calendar


def market_fixture():
    cal = calendar('2026-03-01', '2026-06-30')
    official = {pd.Timestamp(s['session_date']): s for s in cal['sessions']}
    days = pd.DatetimeIndex(sorted(official))
    ends = []
    for day in days:
        begin = pd.Timestamp(str(day.date()) + ' 04:00', tz=core.ET)
        close = pd.Timestamp(official[day]['close_at']).tz_convert(core.ET)
        ends.extend(pd.date_range(begin, close, freq='5min', inclusive='right').tz_convert('UTC'))
    ends = pd.DatetimeIndex(ends)
    local = ends.tz_convert(core.ET)
    dates = local.tz_localize(None).normalize()
    f = pd.DataFrame({'end_at': ends, 'start_at': ends - pd.Timedelta(minutes=5),
                       'available_at': ends + pd.Timedelta(seconds=1), 'date': dates,
                       'session_type': np.where(local.hour * 60 + local.minute > 570, 'regular', 'pre_market'),
                       'open': 100., 'high': 100., 'low': 100., 'close': 100., 'volume': 100.,
                       'share_factor': 1., 'econ_open': 100., 'econ_high': 100.,
                       'econ_low': 100., 'econ_close': 100., 'econ_volume': 100.}, index=ends)
    reg = f[f.session_type == 'regular']
    daily = reg.groupby('date').agg(open=('econ_open', 'first'), high=('econ_high', 'max'),
                                    low=('econ_low', 'min'), close=('econ_close', 'last'),
                                    volume=('econ_volume', 'sum'), available_at=('available_at', 'max'))
    daily['complete'] = True
    return core.Market({s: f.copy() for s in core.SYMBOLS}, {s: daily.copy() for s in core.SYMBOLS},
                       official, days, {s: [] for s in core.SYMBOLS},
                       {s: dict.fromkeys(days, True) for s in core.SYMBOLS}, {})


class ContractTests(unittest.TestCase):
    def test_position_before_and_after_expiry_outputs_exit_obligation(self):
        m = market_fixture()
        position = {'symbol': 'SOXL', 'entry_at': '2026-05-04T09:30:00-04:00', 'entry_fill_raw': 100.05, 'quantity': 10}
        before = manage(m, position, '2026-05-14T15:00:00-04:00')
        self.assertEqual(before['exit_action'], 'WAIT_FOR_LIMIT_OR_EXPIRY')
        self.assertEqual(before['take_profit_order']['limit_price'], 105.06)
        after = manage(m, position, '2026-05-15T16:00:00-04:00')
        self.assertEqual(after['action'], 'NO_ACTION')
        self.assertEqual(after['exit_action'], 'FORCE_SELL_AT_EXPIRY')
        self.assertTrue(after['force_sell_intent']['due'])
        self.assertTrue(after['force_sell_intent']['cancel_attached_take_profit_first'])
        self.assertTrue(after['force_sell_intent']['only_if_position_still_open'])
        self.assertFalse(after['force_sell_intent']['broker_submission'])

    def test_confirmed_take_profit_does_not_generate_second_sell(self):
        m = market_fixture()
        closed = manage(m, {'symbol': 'SOXL', 'take_profit_filled': True}, '2026-05-15T16:00:00-04:00')
        self.assertEqual(closed['exit_action'], 'NONE')

    def test_live_partial_current_day_does_not_replace_previous_close(self):
        m = market_fixture(); day = pd.Timestamp('2026-05-04')
        for frame in m.daily.values():
            frame.loc[day, 'complete'] = False
            frame.loc[day, 'available_at'] = pd.Timestamp('2026-05-04 10:35:01', tz=core.ET)
        row, audit = core.feature_row(m, pd.Timestamp('2026-05-04 10:55:30', tz=core.ET))
        self.assertIsNotNone(row, audit)

    def test_live_receipt_after_analysis_is_not_available(self):
        m = market_fixture(); at = pd.Timestamp('2026-05-04 08:55:30', tz=core.ET)
        for daily in m.daily.values():
            daily['live_available_at'] = daily.available_at
        for frame in m.bars.values():
            frame['live_available_at'] = frame.available_at
            frame.loc[frame.date == pd.Timestamp('2026-05-04'), 'live_available_at'] = at + pd.Timedelta(hours=1)
        historical, _ = core.feature_row(m, at)
        live, audit = core.feature_row(m, at, require_received=True)
        self.assertIsNotNone(historical)
        self.assertIsNone(live, audit)

    def test_loader_handles_microseconds_and_official_half_day(self):
        cal = calendar('2025-07-03', '2025-07-07')
        ends = pd.date_range('2025-07-03 09:35', '2025-07-03 16:00', freq='5min', tz=core.ET).tz_convert('UTC').as_unit('us')
        after = ends > pd.Timestamp('2025-07-03 13:00', tz=core.ET)
        frame = pd.DataFrame({'end_at': ends, 'start_at': ends - pd.Timedelta(minutes=5),
                               'available_at': ends + pd.Timedelta(seconds=1),
                               'session_date': '2025-07-03', 'session_type': 'regular',
                               'price_basis': 'NONE', 'open': 100., 'high': 100., 'low': 100.,
                               'close': np.where(after, 200., 100.), 'volume': 100.})
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); (base / 'bars').mkdir()
            (base / 'calendar.json').write_text(json.dumps(cal))
            for symbol in core.SYMBOLS:
                frame.to_parquet(base / 'bars' / f'{symbol}.parquet', index=False)
            loaded = core.load_market(base)
            for symbol in core.SYMBOLS:
                self.assertTrue(loaded.day_complete[symbol][pd.Timestamp('2025-07-03')])
                self.assertEqual(loaded.daily[symbol].loc[pd.Timestamp('2025-07-03'), 'close'], 100.)

    def test_premarket_and_arbitrary_analysis_times(self):
        m = market_fixture()
        for clock in ['08:55:30', '08:57:20', '10:02:19', '14:11:42']:
            at = pd.Timestamp('2026-05-04 ' + clock, tz=core.ET)
            features, audit = core.feature_row(m, at)
            self.assertIsNotNone(features, audit)
            self.assertTrue(all(pd.Timestamp(cut) < at for cut in audit['data_cutoffs'].values()))

    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            core.timestamp('2026-05-04 08:55:30')

    def test_future_bar_mutation_cannot_change_analysis(self):
        m = market_fixture(); at = pd.Timestamp('2026-05-04 08:55:30', tz=core.ET)
        old, _ = core.feature_row(m, at)
        for f in m.bars.values():
            f.loc[f.end_at > at, ['econ_open', 'econ_high', 'econ_low', 'econ_close']] *= 19
        new, _ = core.feature_row(m, at)
        self.assertEqual(old, new)

    def test_entry_is_after_decision_and_delay(self):
        m = market_fixture(); at = pd.Timestamp('2026-05-04 08:55:30', tz=core.ET)
        label = core.label_entry(m, 'SOXL', at)
        self.assertEqual(label['entry_at'].tz_convert(core.ET).strftime('%H:%M:%S'), '09:00:00')
        self.assertGreater(label['entry_at'], at + pd.Timedelta(seconds=30))
        self.assertEqual(label['attached_order']['limit_price'], 105.11)
        self.assertFalse(label['attached_order']['broker_submission'])

    def test_profit_before_entry_is_not_a_win(self):
        m = market_fixture(); at = pd.Timestamp('2026-05-04 08:55:30', tz=core.ET)
        f = m.bars['SOXL']
        f.loc[(f.date == pd.Timestamp('2026-05-04')) & (f.end_at <= at), 'high'] = 200
        label = core.label_entry(m, 'SOXL', at)
        self.assertEqual(label['reason'], 'expiry')
        self.assertEqual(label['target_hit'], 0)

    def test_fourteen_calendar_days_and_early_close(self):
        m = market_fixture()
        self.assertEqual(core.expiry_day(m, pd.Timestamp('2026-05-04')), pd.Timestamp('2026-05-15'))
        self.assertEqual(core.expiry_day(m, pd.Timestamp('2026-05-12')), pd.Timestamp('2026-05-22'))

    def test_hit_after_expiry_does_not_change_loss(self):
        m = market_fixture(); at = pd.Timestamp('2026-05-04 08:55:30', tz=core.ET)
        m.bars['SOXL'].loc[m.bars['SOXL'].date > pd.Timestamp('2026-05-15'), 'high'] = 200
        label = core.label_entry(m, 'SOXL', at)
        self.assertEqual(label['reason'], 'expiry')
        self.assertLess(label['net_return'], 0)

    def test_missing_prefix_stays_unresolved(self):
        m = market_fixture(); at = pd.Timestamp('2026-05-04 08:55:30', tz=core.ET)
        m.day_complete['SOXL'][pd.Timestamp('2026-05-05')] = False
        label = core.label_entry(m, 'SOXL', at)
        self.assertEqual(label['reason'], 'missing_prefix')
        self.assertIsNone(label['target_hit'])
        self.assertTrue(np.isnan(label['net_return']))

    def test_reverse_split_does_not_create_profit(self):
        m = market_fixture(); f = m.bars['SOXS']; day = pd.Timestamp('2026-05-05')
        later = f.date >= day
        f.loc[later, ['open', 'high', 'low', 'close']] *= 10
        f.loc[later, 'share_factor'] = .1
        label = core.label_entry(m, 'SOXS', pd.Timestamp('2026-05-04 08:55:30', tz=core.ET))
        self.assertEqual(label['reason'], 'expiry')
        self.assertLess(label['net_return'], 0)
        f.loc[(f.date == day) & (f.session_type == 'regular'), 'high'] = 1100
        label = core.label_entry(m, 'SOXS', pd.Timestamp('2026-05-04 08:55:30', tz=core.ET))
        self.assertEqual(label['reason'], 'target')
        self.assertAlmostEqual(label['exit_fill_economic'], 105.11)

    def test_no_same_day_reentry_and_all_costs_preserved(self):
        m = market_fixture()
        m.bars['SOXL'].loc[m.bars['SOXL'].session_type == 'regular', 'high'] = 106
        rows = []; labels = []; actions = {}
        for day in ['2026-05-04', '2026-05-05']:
            for clock in ['08:55:30', '10:55:30', '13:55:30']:
                at = pd.Timestamp(day + ' ' + clock, tz=core.ET).tz_convert('UTC')
                rows.append({'decision_at': at, 'date': pd.Timestamp(day), 'clock': clock})
                actions[at] = 'SOXL'
                labels.append(core.label_entry(m, 'SOXL', at))
        candidates = pd.DataFrame(rows).set_index('decision_at')
        equity, trades, decisions, orders = replay(m, candidates, labels, actions, '2026-05-04', '2026-05-05')
        self.assertEqual(len(trades), 2); self.assertEqual(len(orders), 2)
        self.assertAlmostEqual(equity.equity.iloc[-1], float(np.prod(1 + trades.net_return)))
        self.assertTrue((trades.net_return < .05).all())

    def test_pending_position_keeps_dividend_receivable(self):
        m = market_fixture(); f = m.bars['SOXL']
        m.bars['SOXL'] = f[f.date <= pd.Timestamp('2026-05-05')]
        m.actions['SOXL'] = [{'type': 'CashDividends', 'date': '20260505', 'payment_date': '20260508', 'value': '1'}]
        at = pd.Timestamp('2026-05-04 08:55:30', tz=core.ET).tz_convert('UTC')
        label = core.label_entry(m, 'SOXL', at)
        self.assertEqual(label['reason'], 'pending'); self.assertIsNone(label['target_hit'])
        candidates = pd.DataFrame([{'decision_at': at, 'date': pd.Timestamp('2026-05-04'), 'clock': '08:55:30'}]).set_index('decision_at')
        equity, trades, _, _ = replay(m, candidates, [label], {at: 'SOXL'}, '2026-05-04', '2026-05-05')
        self.assertGreater(equity.receivables.iloc[-1], 0)
        self.assertGreater(equity.position_value.iloc[-1], 0)
        self.assertEqual(len(trades), 1)

    def test_primary_gate_is_three_month_mean_not_each_month(self):
        days = pd.DatetimeIndex(['2026-05-29', '2026-06-30', '2026-07-31', '2026-08-31', '2026-09-30'])
        equity = pd.DataFrame({'equity': [1.2, 1.2, 1.08, 1.08, 1.404], 'mark_complete': True, 'symbol': 'CASH'}, index=days)
        result = statistics(equity, pd.DataFrame(), ['2026-05', '2026-07', '2026-09'])
        self.assertAlmostEqual(result['primary_average'], (.2 - .1 + .3) / 3)
        self.assertTrue(result['primary_pass'])
        equity.loc[pd.Timestamp('2026-07-31'), 'mark_complete'] = False
        self.assertFalse(statistics(equity, pd.DataFrame(), ['2026-05', '2026-07', '2026-09'])['primary_pass'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
