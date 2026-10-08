"""Contract tests exercise coverage, causal cutoff, replay and atomic pair publication."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import digest, write
from data import Acquisition, coverage, cutoffs, normalize
from publish import TARGETS, publish
from views import at, MODEL
from run import verify_inputs


class ContractTests(unittest.TestCase):
    def test_missing_internal_day_is_not_hidden_by_endpoints(self):
        cal = {'start': '2026-10-01', 'days': ['2026-10-01', '2026-10-02', '2026-10-05']}
        rows = [['2026-10-01', 1, 2, 1, 2], ['2026-10-05', 2, 3, 2, 3]]
        result = coverage(rows, cal, '2026-10-01', '2026-10-05')
        self.assertEqual(result['status'], 'internal_gaps')
        self.assertEqual(result['internal_missing'], ['2026-10-02'])

    def test_weekend_does_not_create_missing_session(self):
        cal = {'start': '2026-10-01', 'days': ['2026-10-02', '2026-10-05']}
        rows = [['2026-10-02', 1, 2, 1, 2], ['2026-10-05', 2, 3, 2, 3]]
        self.assertEqual(coverage(rows, cal, '2026-10-02', '2026-10-05')['status'], 'complete')

    def test_open_session_not_used_as_complete_day(self):
        cal = {'US': {'start': '2026-10-01', 'end': '2026-10-15', 'days': ['2026-10-07', '2026-10-08'],
                      'early_close': [], 'closes': {'2026-10-08': '2026-10-08T16:00:00-04:00'}}}
        self.assertEqual(cutoffs(cal, 'auto', datetime(2026, 10, 8, 15, tzinfo=timezone.utc))['US'], '2026-10-07')
        self.assertEqual(cutoffs(cal, 'auto', datetime(2026, 10, 8, 21, tzinfo=timezone.utc))['US'], '2026-10-08')

    def test_expired_calendar_refuses_auto_latest(self):
        cal = {'US': {'start': '2026-10-01', 'end': '2026-10-02', 'days': [], 'early_close': []}}
        with self.assertRaises(ValueError):
            cutoffs(cal, 'auto', datetime(2026, 10, 8, 21, tzinfo=timezone.utc))

    def test_expired_live_calendar_allows_explicit_historical_review(self):
        cal = {'US': {'start': '2026-10-01', 'end': '2026-10-02',
                      'days': ['2026-10-02'], 'early_close': []}}
        self.assertEqual(cutoffs(cal, '2026-10-02', datetime(2026, 10, 8, 21, tzinfo=timezone.utc)),
                         {'US': '2026-10-02'})

    def test_half_day_closes_use_exchange_time(self):
        cal = {'US': {'start': '2026-11-25', 'end': '2026-12-31',
                      'days': ['2026-11-25', '2026-11-27'], 'early_close': ['2026-11-27']}}
        self.assertEqual(cutoffs(cal, 'auto', datetime(2026, 11, 27, 18, tzinfo=timezone.utc))['US'], '2026-11-27')

    def test_equal_cutoff_prefers_no_internal_gap(self):
        engine = Acquisition.__new__(Acquisition)
        engine.start = '2026-10-01'; engine.ends = {'US': '2026-10-05'}
        engine.cal = {'US': {'start': engine.start, 'days': ['2026-10-01', '2026-10-02', '2026-10-05']}}
        good = ('2026-10-05', 3, 0, None, [[d, 1, 1, 1, 1] for d in engine.cal['US']['days']])
        bad = ('2026-10-05', 2, 100, None, [good[4][0], good[4][-1]])
        self.assertGreater(engine.rank(good, {'code': 'US.TEST'}), engine.rank(bad, {'code': 'US.TEST'}))

    def test_future_and_prelisting_prices_excluded(self):
        frame = pd.DataFrame([{'time_key': d, 'open': 1, 'high': 2, 'low': 1, 'close': 2}
                              for d in ['2026-10-01', '2026-10-02', '2026-10-05']])
        result = normalize(frame, '2026-10-01', '2026-10-02', '2026-10-02')
        self.assertEqual([r[0] for r in result], ['2026-10-02'])

    def test_short_history_has_unknown_long_attributes(self):
        days = [d.strftime('%Y-%m-%d') for d in pd.bdate_range('2026-01-01', periods=65)]
        rows = [[d, 1, 2, 1, 1 + i * .001] for i, d in enumerate(days)]
        values, states, _ = at(rows, days, days[-1])
        self.assertIsNone(values[0]); self.assertIsNone(values[4])
        self.assertEqual(states[0], 'insufficient_history')
        self.assertIsNotNone(values[1])

    def test_adapter_matches_approved_model_on_identical_input(self):
        import numpy as np
        days = [d.strftime('%Y-%m-%d') for d in pd.bdate_range('2025-01-01', periods=160)]
        prices = np.exp(np.cumsum(np.sin(np.arange(160)) * .02 + .001))
        rows = [[d, p, p, p, p] for d, p in zip(days, prices)]
        values, states, _ = at(rows, days, days[-1])
        np.testing.assert_allclose(values, MODEL.measure(np.diff(np.log(prices))[-140:])[0], rtol=1e-12)
        self.assertEqual(states, ['scored'] * 5)

    def test_replay_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder);write(run / 'inputs/a.json', {'value': 1})
            write(run / 'inputs/data_manifest.json', {'datasets': [], 'input_files': {'inputs/a.json': digest(run / 'inputs/a.json')}})
            verify_inputs(run)
            write(run / 'inputs/a.json', {'value': 2})
            with self.assertRaises(ValueError):verify_inputs(run)

    def test_one_pointer_updates_both_and_corruption_preserves_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);store = root / '.cache/pair';store.mkdir(parents=True)
            for name in TARGETS:
                p = root / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('legacy')
            def make_run(name, content):
                run = store / 'runs' / name
                for target in TARGETS.values():
                    p = run / 'pages' / target;p.parent.mkdir(parents=True, exist_ok=True);p.write_text(content)
                write(run / 'validation/acceptance.json', {'status': 'passed'})
                write(run / 'checksums.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file()})
                return run
            first = make_run('a', 'first');publish(store, first, root)
            self.assertTrue(all((root / name).read_text() == 'first' for name in TARGETS))
            second = make_run('b', 'second');publish(store, second, root)
            self.assertTrue(all((root / name).read_text() == 'second' for name in TARGETS))
            corrupt = make_run('c', 'third');(corrupt / 'pages/ai_trend_quadrant_v2/index.html').write_text('changed')
            with self.assertRaises(ValueError):publish(store, corrupt, root)
            self.assertTrue(all((root / name).read_text() == 'second' for name in TARGETS))

    def test_failed_validation_never_installs_aliases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); store = root / '.cache/pair'; run = store / 'runs/failed'
            for name in TARGETS:
                p = root / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('legacy')
            write(run / 'validation/acceptance.json', {'status': 'candidate_only'})
            with self.assertRaises(ValueError):publish(store, run, root)
            self.assertFalse((store / 'current').exists())
            self.assertTrue(all((root / name).read_text() == 'legacy' for name in TARGETS))

    def test_historical_candidate_cannot_regress_published_cutoff(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); store = root / '.cache/pair'; run = store / 'runs/old'
            write(store / 'current/inputs/data_manifest.json', {'market_cutoffs': {'US': '2026-10-07'}})
            write(run / 'inputs/data_manifest.json', {'market_cutoffs': {'US': '2026-10-06'}, 'datasets': []})
            write(run / 'validation/acceptance.json', {'status': 'passed'})
            write(run / 'checksums.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file()})
            with self.assertRaisesRegex(ValueError, 'backwards'):publish(store, run, root)

    def test_foreign_alias_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);store = root / '.cache/pair';store.mkdir(parents=True)
            write(store / 'publication.json', {'root': str(root)})
            for name in TARGETS:
                p = root / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('another task')
            from publish import initialize
            with self.assertRaises(RuntimeError):initialize(store, root)
            self.assertTrue(all((root / name).read_text() == 'another task' for name in TARGETS))


if __name__ == '__main__':
    unittest.main()
