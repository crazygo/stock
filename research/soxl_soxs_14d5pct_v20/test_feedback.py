import unittest
import numpy as np
import pandas as pd
from resolved_feedback import core, prefix_receipt, feedback_pool, STATE_COLUMNS


class FeedbackTests(unittest.TestCase):
    def frame(self):
        at = pd.date_range('2026-01-05 14:25:30', periods=5, tz='UTC')
        f = pd.DataFrame(0., index=at, columns=[*STATE_COLUMNS, 'raw_C', 'raw_D', 'cal_C', 'cal_D', 'actual_C', 'actual_D'])
        f['date'] = at.tz_convert(core.ET).tz_localize(None).normalize()
        f['release'] = pd.Timestamp('2026-01-23', tz='UTC')
        f['terminal_release'] = pd.to_datetime(['2026-01-05 15:00', '2026-01-14 15:00', '2026-01-07 15:00', '2026-01-08 15:00', '2026-01-09 15:00'], utc=True)
        return f

    def test_known_later_dates_cannot_jump_pending_frontier(self):
        f = self.frame(); cutoff = pd.Timestamp('2026-01-12', tz='UTC')
        pool, info = feedback_pool(f, cutoff, 'frontier')
        self.assertEqual(list(pool.index), [f.index[0]])
        self.assertEqual(info['known_but_waiting_behind_pending'], 3)
        later, _ = feedback_pool(f, cutoff, 'available')
        self.assertEqual(len(later), 4)
        old, _ = feedback_pool(f, cutoff, 'full'); self.assertEqual(len(old), 0)

    def test_pending_values_and_same_time_release_are_not_training(self):
        f = self.frame(); cutoff = pd.Timestamp('2026-01-14 15:00', tz='UTC')
        before, _ = feedback_pool(f, cutoff, 'frontier')
        f.loc[f.index[1:], ['actual_C', 'actual_D']] = 1e10
        after, _ = feedback_pool(f, cutoff, 'frontier')
        pd.testing.assert_frame_equal(before, after)
        released, _ = feedback_pool(f, cutoff + pd.Timedelta(seconds=1), 'frontier')
        self.assertEqual(len(released), 5)

    def test_expired_missing_prefix_remains_explicit(self):
        f = self.frame(); f.loc[f.index[1], 'terminal_release'] = pd.NaT
        pool, info = feedback_pool(f, pd.Timestamp('2026-02-01', tz='UTC'), 'frontier')
        self.assertEqual(len(pool), 1); self.assertTrue(info['expired_unresolved_prefix'])

    def market(self):
        day = pd.Timestamp('2026-01-05'); start = pd.Timestamp('2026-01-05 14:30', tz='UTC')
        ends = pd.date_range(start + pd.Timedelta(minutes=5), periods=4, freq='5min')
        bars = pd.DataFrame({'start_at': ends - pd.Timedelta(minutes=5), 'end_at': ends,
                             'available_at': ends + pd.Timedelta(seconds=1), 'session_type': 'regular'}, index=ends)
        m = core.Market({'SOXL': bars}, {}, {day: {'open_at': str(start), 'close_at': str(ends[-1])}}, pd.DatetimeIndex([day]), {}, {}, {})
        label = {'symbol': 'SOXL', 'entry_at': start, 'exit_at': ends[2], 'reason': 'target',
                 'net_return': .049, 'dividends': [], 'complete_window': True,
                 'label_available_at': pd.Timestamp('2026-01-16 21:00:01', tz='UTC')}
        return m, label

    def test_terminal_prefix_and_future_suffix(self):
        market, label = self.market(); receipt = prefix_receipt(market, label)
        self.assertTrue(receipt['known']); self.assertEqual(receipt['bars'], 3)
        market.bars['SOXL'].loc[market.bars['SOXL'].index[-1], 'available_at'] += pd.Timedelta(days=100)
        self.assertEqual(receipt, prefix_receipt(market, label))
        market.bars['SOXL'] = market.bars['SOXL'].drop(market.bars['SOXL'].index[1])
        self.assertFalse(prefix_receipt(market, label)['known'])

    def test_delayed_bar_and_dividend_keep_waiting(self):
        market, label = self.market()
        late = pd.Timestamp('2026-01-06 06:00', tz='UTC')
        market.bars['SOXL'].loc[market.bars['SOXL'].index[1], 'available_at'] = late
        self.assertEqual(prefix_receipt(market, label)['release'], late)
        label['dividends'] = [{'economic_amount': 1.}]
        self.assertEqual(prefix_receipt(market, label)['release'], label['label_available_at'])
        label['complete_window'] = False
        self.assertFalse(prefix_receipt(market, label)['known'])


if __name__ == '__main__':
    unittest.main()
