from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo

import pandas as pd

from .ledger import (Ledger, _at, _prediction_hash, digest, file_sha, issue_once,
                     reconcile_session, register_cohort, route_contract)

ET = ZoneInfo('America/New_York')


def fixture(root: Path):
    dates = [d.date().isoformat() for d in pd.bdate_range('2025-12-01', periods=130)
             if d.date().isoformat() not in ('2025-12-25', '2026-01-01')]
    calendar = []
    for day in dates:
        close_hour = 13 if day == '2025-12-24' else 16
        calendar.append({'open_at': _at(day, 9, 30).isoformat(),
                         'close_at': _at(day, close_hour, 0).isoformat()})
    paths = {role: root / f'{role}.bin' for role in
             ('model', 'source', 'schema', 'calibration', 'snapshot')}
    for role, path in paths.items():
        path.write_bytes(f'frozen-{role}'.encode())
    artifacts = {'model_sha256': digest([file_sha(paths['model'])]),
                 'schema_sha256': file_sha(paths['schema']),
                 'calibration_sha256': file_sha(paths['calibration']),
                 'source_code_sha256': digest([file_sha(paths['source'])]),
                 'files': {role: ([{'path': str(path), 'sha256': file_sha(path)}]
                                  if role in ('model', 'source') else
                                  {'path': str(path), 'sha256': file_sha(path)})
                           for role, path in paths.items()}}
    spec = {'calendar_sessions': calendar,
            'groups': {'chips': ['AVGO', 'AMD', 'NVDA', 'INTC', 'QCOM'],
                       'optics': ['AVGO'], 'storage': ['MU']}}
    route = {'route_id': 'B_group', 'model_version': 'frozen-v1',
             'score_column': 'raw_3d_5pct',
             'thresholds': {'chips': .6, 'optics': .9, 'storage': None},
             'artifacts': artifacts}
    cohort = {'cohort_id': 'fixture-60', 'session_dates': dates[:60],
              'calendar_sessions': calendar,
              'routes': ['B_group', 'B_no_group', 'C_group', 'C_no_group', 'C_no_daily'],
              'groups': ['chips', 'optics', 'storage'],
              'last_session_close_at': calendar[59]['close_at'],
              'evidence_mode': 'fixture', 'bootstrap_repeats': 40}
    cohort['route_contracts'] = {name: route_contract(route)
                                 for name in cohort['routes']}
    return dates, spec, route, cohort, artifacts


def prediction(symbol, day, route, artifacts, *, score=.8, received=None):
    cutoff = _at(day, 11, 30).isoformat()
    scores = {name: score for name in (
        '1d_3pct', '1d_5pct', '1d_8pct', '3d_3pct', '3d_5pct', '3d_8pct',
        '5d_3pct', '5d_5pct', '5d_8pct')}
    pred = {'route_id': route['route_id'], 'model_version': route['model_version'],
            'sample_id': f'{symbol}:{day}', 'symbol': symbol, 'session_date': day,
            'cutoff_at': cutoff, 'decision_at': cutoff,
            'raw': scores, 'processed': dict(scores), 'score_column': 'raw_3d_5pct',
            'lineage': {'source_ids': ['fixture-bar-1'], 'group_version_ids': ['fixture-group'],
                        'latest_available_at': _at(day, 11, 30).isoformat(),
                        'latest_received_at': received or _at(day, 11, 30).isoformat(),
                        'receipt_verified': True,
                        'snapshot_sha256': artifacts['files']['snapshot']['sha256'],
                        'mode': 'fixture'},
            'quality': {'g1_ready': True, 'g2_eligible': True,
                        'publication_eligible': True, 'rejections': [],
                        'artifacts': artifacts}}
    pred['quality']['prediction_sha256'] = _prediction_hash(pred)
    return pred


class LedgerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dates, self.spec, self.route, self.cohort, self.artifacts = fixture(self.root)
        self.now = [_at(self.dates[0], 8, 0)]
        self.path = self.root / 'forward.sqlite'
        self.ledger = Ledger(self.path, clock=lambda: self.now[0])
        register_cohort(self.cohort, self.ledger)

    def tearDown(self):
        self.ledger.close()
        self.temp.cleanup()

    def _issue(self, symbol='AVGO', day=None, run_id='r1', ledger=None, score=.8):
        day = day or self.dates[0]
        return issue_once(self.spec, self.route, self.cohort, run_id,
                          [prediction(symbol, day, self.route, self.artifacts, score=score)],
                          {'g2_eligible': True}, ledger or self.ledger)[0]

    def test_cross_run_restart_cooldown_and_group_trigger(self):
        self.now[0] = _at(self.dates[0], 11, 30, 10)
        first = self._issue()
        self.assertEqual(first['event_type'], 'buy')
        self.assertEqual(first['payload']['trigger_groups'], ['chips'])
        self.assertEqual(self._issue(run_id='retry')['event_id'], first['event_id'])
        other = Ledger(self.path, clock=lambda: self.now[0])
        try:
            self.assertEqual(self._issue(run_id='restart', ledger=other)['event_id'], first['event_id'])
            self.now[0] = _at(self.dates[1], 11, 30, 10)
            blocked = self._issue(day=self.dates[1], run_id='r2', ledger=other)
            self.assertEqual(blocked['payload']['reason'], 'cooldown')
            self.now[0] = _at(self.dates[5], 11, 30, 10)
            later = self._issue(day=self.dates[5], run_id='r3', ledger=other)
            self.assertEqual(later['event_type'], 'buy')
            self.assertEqual(self.ledger.verify_chain(), 7)  # cohort + three decision pairs
        finally:
            other.close()

    def test_concurrent_same_symbol_only_one_buy(self):
        self.now[0] = _at(self.dates[0], 11, 30, 10)
        def call(index):
            db = Ledger(self.path, clock=lambda: self.now[0])
            try:
                return self._issue(run_id=f'parallel-{index}', ledger=db)
            finally:
                db.close()
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(call, (1, 2)))
        self.assertEqual(receipts[0]['event_id'], receipts[1]['event_id'])
        self.assertEqual(len(self.ledger.events(event_type='buy')), 1)

    def test_cooldown_crosses_half_day_holiday_and_new_year(self):
        first_day = '2025-12-23'
        self.now[0] = _at(first_day, 11, 30, 10)
        first = self._issue(day=first_day)
        self.assertEqual(first['payload']['cooldown_end'], _at('2025-12-29', 14, 35).isoformat())
        self.now[0] = _at('2025-12-29', 11, 30, 10)
        self.assertEqual(self._issue(day='2025-12-29', run_id='blocked')['payload']['reason'], 'cooldown')
        self.now[0] = _at('2025-12-30', 11, 30, 10)
        second = self._issue(day='2025-12-30', run_id='new-year')
        self.assertEqual(second['event_type'], 'buy')
        self.assertEqual(second['payload']['cooldown_end'], _at('2026-01-05', 11, 35).isoformat())
        self.now[0] = _at('2026-01-05', 11, 30, 10)
        self.assertEqual(self._issue(day='2026-01-05', run_id='blocked-2')['payload']['reason'], 'cooldown')

    def test_cooldown_survives_new_cohort_and_model_version(self):
        second = dict(self.cohort, cohort_id='fixture-following',
                      session_dates=self.dates[40:100],
                      last_session_close_at=self.spec['calendar_sessions'][99]['close_at'])
        newer_route = dict(self.route, model_version='frozen-v2')
        second['route_contracts'] = dict(self.cohort['route_contracts'],
                                         B_group=route_contract(newer_route))
        register_cohort(second, self.ledger)
        self.now[0] = _at(self.dates[39], 11, 30, 10)
        self.assertEqual(self._issue(day=self.dates[39])['event_type'], 'buy')
        self.now[0] = _at(self.dates[40], 11, 30, 10)
        result = issue_once(self.spec, newer_route, second, 'new-version',
                            [prediction('AVGO', self.dates[40], newer_route, self.artifacts)],
                            {'g2_eligible': True}, self.ledger)[0]
        self.assertEqual(result['payload']['reason'], 'cooldown')

    def test_same_cohort_route_version_threshold_or_hash_change_rejected(self):
        self.now[0] = _at(self.dates[0], 11, 30, 10)
        for changed in (dict(self.route, model_version='other'),
                        dict(self.route, thresholds=dict(self.route['thresholds'], chips=.7)),
                        dict(self.route, artifacts=dict(self.artifacts, schema_sha256='f' * 64))):
            with self.assertRaisesRegex(ValueError, 'frozen cohort contract'):
                issue_once(self.spec, changed, self.cohort, 'changed',
                           [prediction('AVGO', self.dates[0], changed, self.artifacts)],
                           {'g2_eligible': True}, self.ledger)
        self.assertFalse(self.ledger.events(event_type='buy'))

    def test_session_reconciliation_keeps_stopped_day_in_denominator(self):
        self.now[0] = _at(self.dates[1], 17, 0)
        event = reconcile_session(self.spec, self.cohort, 'B_group', self.dates[0], self.ledger)
        self.assertEqual(event['payload']['status'], 'missing_run')
        self.assertEqual(reconcile_session(self.spec, self.cohort, 'B_group', self.dates[0],
                                           self.ledger)['event_id'], event['event_id'])

    def test_empty_run_is_distinct_from_stopped_day(self):
        self.now[0] = _at(self.dates[0], 11, 30, 10)
        empty = issue_once(self.spec, self.route, self.cohort, 'empty-run', [],
                           {'session_date': self.dates[0], 'g2_eligible': True}, self.ledger)[0]
        self.assertEqual(empty['event_type'], 'run_empty')
        self.now[0] = _at(self.dates[1], 17, 0)
        row = reconcile_session(self.spec, self.cohort, 'B_group', self.dates[0], self.ledger)
        self.assertEqual(row['payload']['status'], 'ran_no_buy')

    def test_failure_rolls_back_and_retry_does_not_duplicate(self):
        self.now[0] = _at(self.dates[0], 11, 30, 10)
        original = self.ledger.append
        def crash(fields, payload):
            if fields['event_type'] == 'buy':
                raise OSError('simulated commit path failure')
            return original(fields, payload)
        self.ledger.append = crash
        with self.assertRaisesRegex(OSError, 'simulated'):
            self._issue()
        self.ledger.append = original
        self.assertEqual(len(self.ledger.events()), 1)
        self.assertEqual(self._issue()['event_type'], 'buy')
        self.assertEqual(len(self.ledger.events(event_type='buy')), 1)

    def test_early_publication_and_not_yet_received_input_refuse(self):
        self.now[0] = _at(self.dates[0], 11, 29, 59)
        early = self._issue('AMD')
        self.assertEqual(early['event_type'], 'refusal')
        self.assertEqual(early['payload']['reason_detail'], 'published_before_decision')
        self.now[0] = _at(self.dates[1], 11, 30, 10)
        future_receipt = prediction('NVDA', self.dates[1], self.route, self.artifacts,
                                    received=_at(self.dates[1], 11, 30, 20).isoformat())
        row = issue_once(self.spec, self.route, self.cohort, 'future-receipt',
                         [future_receipt], {'g2_eligible': True}, self.ledger)[0]
        self.assertEqual(row['event_type'], 'refusal')
        self.assertEqual(row['payload']['reason_detail'], 'input_not_observed_at_publication')
        self.assertEqual(len(self.ledger.events(event_type='buy')), 0)

    def test_expired_late_receipt_and_bad_prediction_hash_refuse(self):
        self.now[0] = _at(self.dates[0], 11, 30, 31)
        self.assertEqual(self._issue()['payload']['reason'], 'expired')
        self.now[0] = _at(self.dates[1], 11, 30, 10)
        late = prediction('AMD', self.dates[1], self.route, self.artifacts,
                          received=_at(self.dates[1], 11, 30, 31).isoformat())
        row = issue_once(self.spec, self.route, self.cohort, 'late', [late],
                         {'g2_eligible': True}, self.ledger)[0]
        self.assertEqual(row['payload']['reason'], 'data_unverified')
        bad = prediction('NVDA', self.dates[1], self.route, self.artifacts)
        bad['raw']['3d_5pct'] = .99  # bytes changed after prediction hash
        row = issue_once(self.spec, self.route, self.cohort, 'bad-hash', [bad],
                         {'g2_eligible': True}, self.ledger)[0]
        self.assertEqual(row['payload']['reason'], 'data_unverified')
        future = prediction('INTC', self.dates[1], self.route, self.artifacts)
        future['y'] = 1
        future['quality']['prediction_sha256'] = _prediction_hash(future)
        row = issue_once(self.spec, self.route, self.cohort, 'future-label', [future],
                         {'g2_eligible': True}, self.ledger)[0]
        self.assertEqual(row['payload']['reason'], 'data_unverified')
        no_g2 = prediction('QCOM', self.dates[1], self.route, self.artifacts)
        row = issue_once(self.spec, self.route, self.cohort, 'no-g2', [no_g2],
                         {'g2_eligible': False}, self.ledger)[0]
        self.assertEqual(row['payload']['reason'], 'data_unverified')
        self.assertFalse(self.ledger.events(event_type='buy'))

    def test_live_clock_cannot_be_injected_or_open_fixture_db(self):
        with self.assertRaisesRegex(ValueError, 'injected clock'):
            Ledger(self.root / 'live.sqlite', mode='live', clock=lambda: self.now[0])
        with self.assertRaisesRegex(ValueError, 'mode mismatch'):
            Ledger(self.path, mode='live')


if __name__ == '__main__':
    unittest.main()
