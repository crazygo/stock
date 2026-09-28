"""Offline R03 recovery contracts; no market-data request is made."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from . import acquire_batch as batch
from . import acquire_recovery as recovery
from .evaluate import sha


class Reply:
    def __init__(self, responses):
        self.responses = iter(responses)

    def request_history_kline(self, *args, **kwargs):
        return next(self.responses)


def raw_one():
    return pd.DataFrame([{'time_key': '2025-08-04 09:35:00', 'open': 10.,
                          'high': 10.2, 'low': 9.9, 'close': 10.1,
                          'volume': 100., 'turnover': 1000.}])


def part(root, frame, symbol='TEST', month='2025-08', complete=True):
    path = root / 'parts' / symbol / month
    path.mkdir(parents=True)
    frame.to_parquet(path / 'bars.parquet', index=False)
    (path / 'audit.json').write_text('[]\n')
    result = {'symbol': symbol, 'month': month, 'tier': 'futu',
              'pagination_complete': complete,
              'failure': None if complete else 'page_cap',
              'quality': batch.quality(frame, *recovery._month_bounds(month), batch.make_calendar()),
              'bars_sha256': sha(path / 'bars.parquet')}
    (path / 'result.json').write_text(json.dumps(result))
    return path


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_empty_success_keeps_raw_audit_and_missing_coverage(self):
        path = self.root / 'part'
        path.mkdir()
        raw = pd.DataFrame()  # OpenD may return an empty frame with no columns.
        reply = Reply([(0, raw, None)])
        pages, audit, complete, failure = recovery._fetch_pages(
            path, 'TEST', '2025-08-01', '2025-08-31',
            lambda: reply, sleeper=lambda _: None)
        self.assertTrue(complete)
        self.assertIsNone(failure)
        self.assertEqual(len(pages), 1)
        self.assertTrue(audit[0]['ok'])
        self.assertEqual(audit[0]['rows'], 0)
        self.assertTrue((path / 'raw_001.parquet').exists())
        frame = recovery.normalize(pd.concat(pages, ignore_index=True), 'TEST')
        self.assertEqual(list(frame.columns), list(recovery.CANONICAL_COLUMNS))
        self.assertTrue(frame.empty)
        result = recovery._new_result(path, 'TEST', '2025-08', 'futu', frame,
                                      complete, failure, audit, batch.make_calendar(), 0.)
        self.assertEqual(result['data_state'], 'provider_returned_no_bars_unknown_listing_state')
        self.assertEqual(result['quality']['missing_rth_bars'], result['quality']['expected_rth_bars'])
        self.assertGreater(result['quality']['missing_rth_bars'], 0)
        self.assertFalse(result['quality']['rth_complete'])
        self.assertEqual(recovery.verified_complete(path, 'TEST', '2025-08'), result)

    def test_inherit_complete_month_checks_hash_and_keeps_source(self):
        source = self.root / 'old'
        output = self.root / 'new'
        source_part = part(source, recovery.normalize(raw_one(), 'TEST'))
        original = (source_part / 'result.json').read_bytes()
        result = recovery._inherit(source_part, output / 'parts/TEST/2025-08',
                                   'TEST', '2025-08', output)
        self.assertEqual(result['bars_sha256'], sha(source_part / 'bars.parquet'))
        self.assertTrue((output / 'parts/TEST/2025-08').is_symlink())
        record = json.loads((output / 'inherited/TEST/2025-08.json').read_text())
        self.assertEqual(record['source_result_sha256'], sha(source_part / 'result.json'))
        self.assertEqual((source_part / 'result.json').read_bytes(), original)

    def test_changed_or_incomplete_checkpoint_rejected(self):
        frame = recovery.normalize(raw_one(), 'TEST')
        damaged = part(self.root / 'damaged', frame)
        (damaged / 'bars.parquet').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'bars missing or changed'):
            recovery.verified_complete(damaged, 'TEST', '2025-08')
        incomplete = part(self.root / 'incomplete', frame, complete=False)
        with self.assertRaisesRegex(ValueError, 'incomplete checkpoint'):
            recovery.verified_complete(incomplete, 'TEST', '2025-08')

    def test_failed_pagination_stays_unresolved_and_other_part_can_complete(self):
        failed = self.root / 'failed'
        failed.mkdir()
        reply = Reply([(1, 'quota', None)] * 3)
        pages, audit, complete, failure = recovery._fetch_pages(
            failed, 'TEST', '2025-08-01', '2025-08-31',
            lambda: reply, sleeper=lambda _: None)
        self.assertFalse(complete)
        self.assertFalse(pages)
        self.assertEqual(len(audit), 3)
        first = recovery._new_result(failed, 'TEST', '2025-08', 'futu', None,
                                     complete, failure, audit, batch.make_calendar(), 0.)
        self.assertFalse(first['pagination_complete'])
        self.assertEqual(first['data_state'], 'unresolved_failure')
        with self.assertRaisesRegex(ValueError, 'incomplete checkpoint'):
            recovery.verified_complete(failed, 'TEST', '2025-08')
        good = self.root / 'good'
        good.mkdir()
        second = recovery._new_result(good, 'NEXT', '2025-09', 'futu',
                                      recovery.canonical_empty(), True, None,
                                      [{'ok': True, 'rows': 0}], batch.make_calendar(), 0.)
        summary = recovery._progress(self.root / 'out', [first, second], 2)
        self.assertEqual(summary['status'], 'incomplete_unresolved')
        self.assertEqual(summary['complete_parts'], 1)
        self.assertEqual(summary['empty_successful_parts'], 1)
        self.assertEqual(summary['unresolved_failures'], 1)


if __name__ == '__main__':
    unittest.main()
