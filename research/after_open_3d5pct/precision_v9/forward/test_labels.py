from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from ...timeaxis import add_regular_minutes, intervals
from .labels import evaluate_candidate, mature_signals
from .ledger import Ledger, _at, calendar_sessions, canonical, digest, file_sha, issue_once, register_cohort
from .test_ledger import fixture, prediction


def full_bars(spec, day, *, symbol='AVGO', hit_at=10):
    sessions = calendar_sessions(spec)
    entry = _at(day, 11, 35)
    end = add_regular_minutes(entry, 390 * 5, sessions)
    bars = []
    for i, (start, stop) in enumerate(intervals(entry, end, sessions, 5)):
        bars.append({'symbol': symbol, 'start_at': start.isoformat(),
                     'end_at': stop.isoformat(), 'available_at': (stop + timedelta(seconds=1)).isoformat(),
                     'received_at': (stop + timedelta(seconds=2)).isoformat(),
                     'open': 100., 'high': 106. if i == hit_at else 100.2,
                     'low': 99.8, 'close': 100., 'price_basis': 'NONE',
                     'source_id': f'fixture-bar-{i}'})
    return bars, end


def snapshot(bars):
    actions = {'coverage_start': '2025-01-01', 'coverage_end': '2027-12-31',
               'split_dates': [], 'halt_dates': [], 'coverage_verified': True}
    actions['source_sha256'] = digest(actions)
    return {'bars': bars, 'bars_sha256': digest(bars), 'source_sha256': digest(bars),
            'price_basis': 'NONE', 'corporate_actions': actions}


class LabelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dates, self.spec, self.route, self.cohort, self.artifacts = fixture(self.root)
        self.day = self.dates[0]
        self.bars, self.end = full_bars(self.spec, self.day)
        self.candidate = {'sample_id': 'AVGO:first', 'symbol': 'AVGO',
                          'decision_at': _at(self.day, 11, 30).isoformat(),
                          'planned_entry_at': _at(self.day, 11, 35).isoformat()}

    def tearDown(self):
        self.temp.cleanup()

    def test_nine_targets_full_maturity_and_first_touch_interval(self):
        result = evaluate_candidate(self.candidate, snapshot(self.bars),
                                    self.spec['calendar_sessions'], self.end + timedelta(seconds=3))
        self.assertTrue(result['all_nine_mature'])
        self.assertEqual(len(result['y']), 9)
        target = result['targets']['3d_5pct']
        self.assertEqual(target['status'], 'tp')
        self.assertEqual(target['entry_price'], 100.)
        self.assertEqual(target['entry_bar']['start_at'], self.candidate['planned_entry_at'])
        self.assertEqual(target['first_touch']['source_id'], 'fixture-bar-10')
        self.assertEqual(target['observed_rth_bars'], 234)
        self.assertEqual(result['targets']['5d_8pct']['status'], 'fp')
        self.assertEqual(result['targets']['5d_8pct']['observed_rth_bars'], 390)

    def test_early_hit_stays_pending_and_no_future_entry_leaks(self):
        result = evaluate_candidate(self.candidate, snapshot(self.bars),
                                    self.spec['calendar_sessions'], _at(self.day, 12, 35))
        for target in result['targets'].values():
            self.assertEqual(target['status'], 'pending')
            self.assertNotIn('entry_price', target)
            self.assertNotIn('first_touch', target)
        self.assertIsNone(result['y'])

    def test_missing_halt_split_and_changed_snapshot_are_not_zero_labels(self):
        missing = [b for i, b in enumerate(self.bars) if i != 3]
        result = evaluate_candidate(self.candidate, snapshot(missing),
                                    self.spec['calendar_sessions'], self.end + timedelta(days=1))
        self.assertEqual(result['targets']['3d_5pct']['status'], 'missing')
        self.assertIsNone(result['targets']['3d_5pct']['y'])
        halted = snapshot(self.bars)
        halted['corporate_actions']['halt_dates'] = [self.day]
        halted['corporate_actions']['source_sha256'] = digest({
            k: v for k, v in halted['corporate_actions'].items() if k != 'source_sha256'})
        self.assertEqual(evaluate_candidate(self.candidate, halted, self.spec['calendar_sessions'],
                                            self.end + timedelta(days=1))['targets']['3d_5pct']['status'], 'halts')
        split = snapshot(self.bars)
        split['corporate_actions']['split_dates'] = [self.day]
        split['corporate_actions']['source_sha256'] = digest({
            k: v for k, v in split['corporate_actions'].items() if k != 'source_sha256'})
        self.assertEqual(evaluate_candidate(self.candidate, split, self.spec['calendar_sessions'],
                                            self.end + timedelta(days=1))['targets']['3d_5pct']['status'], 'invalid')
        unknown = snapshot(self.bars)
        del unknown['corporate_actions']
        unknown_result = evaluate_candidate(self.candidate, unknown,
                                            self.spec['calendar_sessions'], self.end + timedelta(days=1))
        self.assertEqual(unknown_result['targets']['3d_5pct']['status'], 'evidence_unknown')
        self.assertIsNone(unknown_result['targets']['3d_5pct']['y'])
        changed = snapshot(self.bars)
        changed['bars'][0] = dict(changed['bars'][0], open=200.)
        with self.assertRaisesRegex(ValueError, 'snapshot hash'):
            evaluate_candidate(self.candidate, changed, self.spec['calendar_sessions'], self.end + timedelta(days=1))
        source_file = self.root / 'outcome.json'
        source_file.write_text(canonical(self.bars))
        external = snapshot(self.bars)
        external.update(path=str(source_file), source_sha256=file_sha(source_file))
        evaluate_candidate(self.candidate, external, self.spec['calendar_sessions'],
                           self.end + timedelta(days=1))
        source_file.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'source bytes changed'):
            evaluate_candidate(self.candidate, external, self.spec['calendar_sessions'],
                               self.end + timedelta(days=1))

    def test_buy_label_revisions_append_and_never_remove_buy(self):
        now = [_at(self.day, 8, 0)]
        ledger = Ledger(self.root / 'forward.sqlite', clock=lambda: now[0])
        try:
            register_cohort(self.cohort, ledger)
            now[0] = _at(self.day, 11, 30, 10)
            buy = issue_once(self.spec, self.route, self.cohort, 'r1',
                             [prediction('AVGO', self.day, self.route, self.artifacts)],
                             {'g2_eligible': True}, ledger)[0]
            self.assertEqual(buy['event_type'], 'buy')
            missing = [b for i, b in enumerate(self.bars) if i != 3]
            first = mature_signals([buy['event_id']], snapshot(missing), self.spec['calendar_sessions'],
                                   self.end + timedelta(days=1), ledger=ledger)
            self.assertEqual(len(first), 9)
            self.assertEqual(first[4]['payload']['outcome']['status'], 'missing')
            second = mature_signals([buy['event_id']], snapshot(self.bars), self.spec['calendar_sessions'],
                                    self.end + timedelta(days=2), ledger=ledger)
            self.assertEqual(len(second), 9)
            self.assertEqual(second[4]['payload']['supersedes'], first[4]['event_id'])
            self.assertEqual(len(ledger.events(event_type='buy')), 1)
        finally:
            ledger.close()


if __name__ == '__main__':
    unittest.main()
