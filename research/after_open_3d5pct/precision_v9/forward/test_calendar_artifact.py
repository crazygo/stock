"""Offline replay of the captured Nasdaq/NYSE 2026 response bytes."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from .calendar_artifact import (CALENDAR_ID, _nasdaq_dates, _nyse_dates,
                                build_from_sources, require_coverage,
                                validate_calendar, verify_artifact)
from .ledger import _at, calendar_sessions, canonical, file_sha

RUN = Path(__file__).resolve().parent / 'calendar_2026_official_v1'
EXPECTED_CLOSED = {'2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03',
                   '2026-05-25', '2026-06-19', '2026-07-03', '2026-09-07',
                   '2026-11-26', '2026-12-25'}
EXPECTED_EARLY = {'2026-11-27', '2026-12-24'}


class OfficialCalendarTest(unittest.TestCase):
    def setUp(self):
        self.doc = verify_artifact(RUN)

    def _copy_sources(self):
        temporary = tempfile.TemporaryDirectory()
        dest = Path(temporary.name)
        for name in ('nasdaq_response.html', 'nyse_response.html', 'sources.json'):
            shutil.copyfile(RUN / name, dest / name)
        return temporary, dest

    def test_cross_source_dates_and_ledger_feature_shape(self):
        nasdaq = _nasdaq_dates((RUN / 'nasdaq_response.html').read_bytes())
        nyse = _nyse_dates((RUN / 'nyse_response.html').read_bytes())
        self.assertEqual(nasdaq, nyse)
        self.assertEqual(nasdaq, (EXPECTED_CLOSED, EXPECTED_EARLY))
        self.assertEqual(self.doc['calendar_id'], CALENDAR_ID)
        self.assertEqual(len(self.doc['sessions']), 251)
        self.assertEqual(len(calendar_sessions({'calendar_sessions': self.doc['sessions']})), 251)
        by_day = {row['session_date']: row for row in self.doc['sessions']}
        self.assertNotIn('2026-07-03', by_day)
        self.assertNotIn('2026-11-26', by_day)
        self.assertEqual(by_day['2026-11-27']['duration_minutes'], 210)
        self.assertEqual(by_day['2026-12-24']['close_at'],
                         '2026-12-24T13:00:00-05:00')
        self.assertEqual(by_day['2026-03-06']['open_at'],
                         '2026-03-06T09:30:00-05:00')
        self.assertEqual(by_day['2026-03-09']['open_at'],
                         '2026-03-09T09:30:00-04:00')
        self.assertEqual(by_day['2026-10-30']['open_at'],
                         '2026-10-30T09:30:00-04:00')
        self.assertEqual(by_day['2026-11-02']['open_at'],
                         '2026-11-02T09:30:00-05:00')
        self.assertEqual(by_day['2026-11-27']['close_at'],
                         '2026-11-27T13:00:00-05:00')

    def test_coverage_sixty_sessions_and_1950_minutes_refuses_2027(self):
        dates = [r['session_date'] for r in self.doc['sessions']]
        valid = require_coverage(self.doc, dates[:60], _at(dates[59], 11, 35))
        self.assertEqual(valid['cohort_sessions'], 60)
        self.assertTrue(valid['coverage_sufficient'])
        with self.assertRaisesRegex(ValueError, 'ends before final'):
            require_coverage(self.doc, dates[-60:], _at(dates[-1], 11, 35))
        with self.assertRaisesRegex(ValueError, 'exceeds 2026'):
            require_coverage(self.doc, dates[-59:] + ['2027-01-04'],
                             _at('2027-01-04', 11, 35))
        with self.assertRaisesRegex(ValueError, 'frozen 11:35'):
            require_coverage(self.doc, dates[:60], _at(dates[59], 11, 30))

    def test_duplicate_reorder_dst_and_source_tamper_rejected(self):
        duplicate = deepcopy(self.doc)
        duplicate['sessions'].insert(1, duplicate['sessions'][0])
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            validate_calendar(duplicate)
        changed = deepcopy(self.doc)
        row = next(r for r in changed['sessions'] if r['session_date'] == '2026-03-09')
        row['open_at'] = '2026-03-09T09:30:00-05:00'
        with self.assertRaisesRegex(ValueError, 'clock differs'):
            validate_calendar(changed)
        temporary, dest = self._copy_sources()
        try:
            build_from_sources(dest)
            self.assertEqual(len(verify_artifact(dest)['sessions']), 251)
            with self.assertRaises(FileExistsError):
                build_from_sources(dest)
            builder = dest / 'builder_snapshot.py'
            original_builder = builder.read_bytes()
            builder.write_bytes(original_builder + b'\n# changed')
            with self.assertRaisesRegex(ValueError, 'code hash changed'):
                verify_artifact(dest)
            builder.write_bytes(original_builder)
            (dest / 'nasdaq_response.html').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'raw response bytes changed'):
                verify_artifact(dest)
        finally:
            temporary.cleanup()

    def test_official_disagreement_fails_before_artifact_write(self):
        temporary, dest = self._copy_sources()
        try:
            path = dest / 'nasdaq_response.html'
            text = path.read_text()
            self.assertIn('November 27, 2026</td><td>Early Close* - U.S.</td><td>1:00 p.m.', text)
            text = text.replace('November 27, 2026</td><td>Early Close* - U.S.</td><td>1:00 p.m.',
                                'November 27, 2026</td><td>Early Close* - U.S.</td><td>Closed')
            path.write_text(text)
            source = json.loads((dest / 'sources.json').read_text())
            record = next(r for r in source['sources'] if r['name'] == 'nasdaq')
            record['raw_sha256'] = file_sha(path)
            record['raw_size'] = path.stat().st_size
            (dest / 'sources.json').write_text(canonical(source) + '\n')
            with self.assertRaisesRegex(ValueError, 'official 2026 dates disagree'):
                build_from_sources(dest)
            self.assertFalse((dest / 'sessions.json').exists())
        finally:
            temporary.cleanup()


if __name__ == '__main__':
    unittest.main()
