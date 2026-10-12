"""Time boundaries, target semantics and holding-management regressions."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd

from continuation import (core, OWN, SIGNED, UNSIGNED, ENTRY_COLUMNS, entry_state, review_state,
    immediate_sale, continuation_target, managed_exit, alive, joint_releases, label_key,
    eligible_dates, calibration, choose_entry, fit)


def fixture():
    at = pd.Timestamp('2026-05-04 09:25:30', tz=core.ET).tz_convert('UTC')
    entry_at = pd.Timestamp('2026-05-01 09:30', tz=core.ET).tz_convert('UTC')
    rows = []
    for time, session, price in [('2026-05-01 09:30', 'regular', 100.),
                                 ('2026-05-01 15:55', 'regular', 101.),
                                 ('2026-05-04 09:20', 'pre_market', 99.),
                                 ('2026-05-04 09:30', 'regular', 98.),
                                 ('2026-05-05 09:30', 'regular', 1000.)]:
        start = pd.Timestamp(time, tz=core.ET).tz_convert('UTC'); end = start + pd.Timedelta(minutes=5)
        rows.append({'start_at': start, 'end_at': end, 'available_at': end + pd.Timedelta(seconds=1),
            'date': start.tz_convert(core.ET).tz_localize(None).normalize(), 'open': price,
            'econ_open': price, 'econ_high': price + 1, 'econ_low': price - 1, 'econ_close': price,
            'share_factor': 1., 'session_type': session})
    bars = pd.DataFrame(rows).set_index('end_at', drop=False)
    label = {'symbol': 'SOXL', 'entry_date': pd.Timestamp('2026-05-01'), 'entry_at': entry_at,
             'decision_at': entry_at - pd.Timedelta(minutes=4, seconds=30), 'entry_fill_economic': 100.,
             'limit_economic': 105., 'fee_each_side': .0005, 'expiry_date': pd.Timestamp('2026-05-14'),
             'expiry_at': pd.Timestamp('2026-05-14 20:00Z'), 'exit_at': pd.Timestamp('2026-05-05 14:00Z'),
             'exit_fill_economic': 105., 'net_return': .049, 'dividends': [], 'complete_window': True,
             'label_available_at': pd.Timestamp('2026-05-14 20:00:01Z')}
    row = pd.Series({**{s + '_' + k: 0. for s in core.ETFS for k in OWN},
                     **{k: 0. for k in SIGNED + UNSIGNED}})
    market = SimpleNamespace(bars={'SOXL': bars}, sessions=pd.DatetimeIndex(['2026-05-01', '2026-05-04', '2026-05-05']),
        day_complete={'SOXL': {pd.Timestamp('2026-05-01'): True}})
    return market, row, label, at


class ContinuationTests(unittest.TestCase):
    def test_future_prices_do_not_enter_observed_state(self):
        market, row, label, at = fixture()
        before, audit = review_state(market, row, label, at)
        for col in ['econ_open', 'econ_high', 'econ_low', 'econ_close', 'open']:
            market.bars['SOXL'].loc[market.bars['SOXL'].available_at > at, col] *= 20
        after, _ = review_state(market, row, label, at)
        self.assertEqual(before, after)
        self.assertEqual(audit['observed_peak'], 102.)
        self.assertAlmostEqual(before['mark_return'], -.01)
        self.assertLess(audit['regular_peak_last_bar'], at)

    def test_review_day_future_missingness_does_not_enter_state(self):
        market, row, label, at = fixture()
        before, _ = review_state(market, row, label, at)
        market.day_complete['SOXL'][pd.Timestamp('2026-05-04')] = False
        self.assertEqual(before, review_state(market, row, label, at)[0])
        market.day_complete['SOXL'][pd.Timestamp('2026-05-01')] = False
        self.assertIsNone(review_state(market, row, label, at)[0])

    def test_continuation_target_and_dividend_cancellation(self):
        market, _, label, at = fixture()
        label['dividends'] = [{'ex': pd.Timestamp('2026-05-04'), 'economic_amount': 3.},
                              {'ex': pd.Timestamp('2026-05-05'), 'economic_amount': 2.}]
        sale = immediate_sale(market, label, at)
        got = continuation_target(label, sale, pd.Timestamp('2026-05-04'))
        self.assertAlmostEqual(got, np.log((105 * .9995 + 2) / (98 * .9995 ** 2)))
        changed = {**label, 'dividends': [{**d, 'economic_amount': 9000.} if d['ex'].day == 4 else d
                                       for d in label['dividends']]}
        self.assertEqual(got, continuation_target(changed, sale, pd.Timestamp('2026-05-04')))

    def test_exit_delay_gap_cap_and_full_window_maturity(self):
        market, _, label, at = fixture()
        market.bars['SOXL'].loc[market.bars['SOXL'].start_at == at + pd.Timedelta(minutes=4, seconds=30), 'open'] = 110.
        sale = immediate_sale(market, label, at)
        self.assertGreater(sale['exit_at'], at + pd.Timedelta(seconds=30))
        self.assertAlmostEqual(sale['exit_fill_economic'], 105 * .9995)
        exited = managed_exit(label, sale, at)
        self.assertEqual(exited['label_available_at'], label['label_available_at'])
        self.assertEqual(exited['reason'], 'continuation_exit')
        self.assertFalse(alive(exited, sale['exit_at'] + pd.Timedelta(seconds=1)))
        self.assertEqual(managed_exit(label, None, at), label)

    def test_full_window_joint_release_and_strict_cutoff(self):
        _, _, label, _ = fixture()
        at = label['decision_at']; date = label['entry_date']
        features = pd.DataFrame([{'date': date}], index=[at])
        pair = {label_key(at, s): deepcopy(label) for s in core.ETFS}
        pair[label_key(at, 'SOXS')]['net_return'] = -.2
        pair[label_key(at, 'SOXS')]['complete_window'] = False
        self.assertEqual(joint_releases(features, pair), {})
        pair[label_key(at, 'SOXS')]['complete_window'] = True
        release = joint_releases(features, pair)
        p = {'training_window_dates': 252}
        self.assertEqual(eligible_dates(release, label['label_available_at'], p), [])
        self.assertEqual(eligible_dates(release, label['label_available_at'] + pd.Timedelta(seconds=1), p), [date])

    def test_calibration_weights_dates_and_excludes_unreleased(self):
        d1, d2, future = map(pd.Timestamp, ['2026-04-01', '2026-04-02', '2026-09-01'])
        records = [{'id': str(i), 'entry_date': d1, 'raw': 0., 'target': .03} for i in range(8)]
        records += [{'id': 'b', 'entry_date': d2, 'raw': 0., 'target': -.01},
                    {'id': 'future', 'entry_date': future, 'raw': -100., 'target': 100.}]
        cal = calibration(records, [d1, d2], {'calibration_window_dates': 60, 'minimum_calibration_dates': 2})
        self.assertAlmostEqual(cal['correction'], .01)
        self.assertNotIn('future', cal['members'])
        self.assertEqual(choose_entry({'SOXL': -.02, 'SOXS': -.03}, cal)['action'], 'CASH')
        self.assertEqual(choose_entry({'SOXL': .02, 'SOXS': .02}, cal)['action'], 'SOXL')

    def test_pooled_training_date_weight_not_review_row_weight(self):
        _, row, _, _ = fixture(); state = entry_state(row, 'SOXL')
        rows = [{'id': str(i), 'entry_date': pd.Timestamp('2026-01-01'), 'state': state, 'target': 1.} for i in range(20)]
        rows += [{'id': 'other', 'entry_date': pd.Timestamp('2026-01-02'), 'state': state, 'target': -1.}]
        model, info = fit(rows, ENTRY_COLUMNS, {'minimum_training_dates': 2})
        self.assertAlmostEqual(info['weight_sum'], 2.)
        self.assertAlmostEqual(model.predict(np.array([[state[c] for c in ENTRY_COLUMNS]]))[0], 0.)

    def test_stress_fill_owns_remaining_state(self):
        market, row, label, at = fixture()
        stressed = {**label, 'entry_fill_economic': 102., 'limit_economic': 107.1}
        normal, _ = review_state(market, row, label, at)
        stress, _ = review_state(market, row, stressed, at)
        self.assertNotEqual(normal['mark_return'], stress['mark_return'])
        self.assertNotEqual(normal['stop_distance'], stress['stop_distance'])
        self.assertLess(immediate_sale(market, stressed, at, 2)['sale_net'], immediate_sale(market, label, at)['sale_net'])


if __name__ == '__main__':
    unittest.main()
