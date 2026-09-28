"""Fixture-only checks for the all-candidate prior store."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from .features import _prior
from .ledger import _at, canonical, digest, file_sha
from .prior_store import (PriorStore, read_prior_asof, register_candidates,
                          update_candidate_outcomes, verify_prior_view)
from .test_labels import full_bars, snapshot
from .test_ledger import fixture


def proven(payload):
    return dict(payload, source_sha256=digest(payload))


class PriorStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dates, self.spec, *_ = fixture(self.root)
        self.day = self.dates[0]
        self.clock = [_at(self.day, 8, 0)]
        self.path = self.root / 'prior.sqlite'
        self.store = PriorStore(self.path, clock=lambda: self.clock[0])
        self.universe = proven({'universe_version': 'all-v1',
                                'members': [{'symbol': 'AVGO', 'role': 'candidate'},
                                            {'symbol': 'AMD', 'role': 'candidate'},
                                            {'symbol': 'QQQ', 'role': 'benchmark'}]})

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def manifest(self, day, *, symbols=('AVGO', 'AMD')):
        return proven({'label_contract': 'v9_11_35_390_rth_none',
                       'rows': [{'symbol': symbol, 'sample_id': f'{symbol}:{day}',
                                 'eligible': True, 'reason': 'frozen_universe'}
                                for symbol in symbols]})

    def session(self, day):
        return next(dict(row, session_date=day)
                    for row in self.spec['calendar_sessions']
                    if row['open_at'][:10] == day)

    def register(self, day=None):
        day = day or self.day
        return register_candidates(self.universe, self.session(day),
                                   self.manifest(day), self.store)

    def mature(self, day=None, *, hit_at=10):
        day = day or self.day
        bars_a, end = full_bars(self.spec, day, symbol='AVGO', hit_at=hit_at)
        bars_b, _ = full_bars(self.spec, day, symbol='AMD', hit_at=-1)
        source = {'AVGO': snapshot(bars_a), 'AMD': snapshot(bars_b)}
        observed = end + timedelta(seconds=3)
        self.clock[0] = observed
        return source, observed

    def test_full_manifest_restart_idempotence_and_non_buy_labels(self):
        incomplete = self.manifest(self.day, symbols=('AVGO',))
        with self.assertRaisesRegex(ValueError, 'omits'):
            register_candidates(self.universe, self.session(self.day), incomplete, self.store)
        registered = self.register()
        self.assertEqual(registered['coverage'], {'registered': 2, 'eligible': 2})
        self.assertEqual(len(self.register()['candidate_ids']), 2)
        other = PriorStore(self.path, clock=lambda: self.clock[0])
        try:
            self.assertEqual(register_candidates(self.universe, self.session(self.day),
                                                 self.manifest(self.day), other), registered)
            self.assertEqual(other.verify_chain(), 3)
        finally:
            other.close()
        source, observed = self.mature()
        versions = update_candidate_outcomes(registered['candidate_ids'], source,
                                             self.spec['calendar_sessions'], observed, self.store)
        self.assertEqual(len(versions), 2)
        self.assertTrue(all(v['payload']['result']['all_nine_mature'] for v in versions))
        outcome_cut = self.store.verify_chain()
        self.assertEqual(update_candidate_outcomes(registered['candidate_ids'], source,
                                                   self.spec['calendar_sessions'], observed,
                                                   self.store), [])
        restarted = PriorStore(self.path, clock=lambda: self.clock[0])
        try:
            self.assertEqual(update_candidate_outcomes(registered['candidate_ids'], source,
                                                       self.spec['calendar_sessions'], observed,
                                                       restarted), [])
        finally:
            restarted.close()
        self.assertEqual(self.store.verify_chain(), outcome_cut)
        # There is no BUY table here. AMD was never issued and still enters the prior.
        view = read_prior_asof(['AMD', 'AVGO'], observed + timedelta(seconds=1),
                               self.store.verify_chain(), self.store)
        self.assertEqual(view['coverage']['registered'], 2)
        self.assertEqual(view['coverage']['mature_selected'], 2)
        self.assertEqual({r['symbol'] for r in view['rows']}, {'AMD', 'AVGO'})
        self.assertEqual(view['rows'][0]['label_available_at'], observed.isoformat())
        self.assertEqual(verify_prior_view(view['path'], self.store)['view_sha256'],
                         view['view_sha256'])

    def test_early_touch_pending_then_revision_old_view_unchanged(self):
        ids = self.register()['candidate_ids']
        source, observed = self.mature()
        early = _at(self.day, 12, 35)
        self.clock[0] = early
        first = update_candidate_outcomes(ids, source, self.spec['calendar_sessions'],
                                          early, self.store)
        self.assertFalse(all(e['payload']['result']['all_nine_mature'] for e in first))
        pending = read_prior_asof(['AVGO'], early + timedelta(seconds=1),
                                  self.store.verify_chain(), self.store)
        self.assertEqual(pending['rows'], [])
        self.assertEqual(pending['coverage']['mature_selected'], 0)
        self.clock[0] = observed
        second = update_candidate_outcomes(ids, source, self.spec['calendar_sessions'],
                                           observed, self.store)
        self.assertEqual({e['payload']['supersedes'] for e in second},
                         {e['event_id'] for e in first})
        old_cut = self.store.verify_chain()
        old = read_prior_asof(['AVGO'], observed + timedelta(seconds=1), old_cut, self.store)
        old_bytes = Path(old['path']).read_bytes()
        changed_bars, _ = full_bars(self.spec, self.day, symbol='AVGO', hit_at=-1)
        changed = dict(source, AVGO=snapshot(changed_bars))
        later = observed + timedelta(days=1)
        self.clock[0] = later
        third = update_candidate_outcomes(ids, changed, self.spec['calendar_sessions'],
                                          later, self.store)
        self.assertEqual(len(third), 2)
        replay = read_prior_asof(['AVGO'], observed + timedelta(seconds=1),
                                 old_cut, self.store)
        self.assertEqual(replay['rows'], old['rows'])
        self.assertEqual(replay['selected_versions'], old['selected_versions'])
        self.assertEqual(Path(old['path']).read_bytes(), old_bytes)
        new = read_prior_asof(['AVGO'], later + timedelta(seconds=1),
                              self.store.verify_chain(), self.store)
        self.assertNotEqual(new['rows'][0]['y'], old['rows'][0]['y'])
        self.assertEqual(new['rows'][0]['label_available_at'], later.isoformat())

    def test_tamper_future_cut_and_source_change_fail_closed(self):
        ids = self.register()['candidate_ids']
        with self.assertRaisesRegex(ValueError, 'future or invalid'):
            read_prior_asof(['AVGO'], _at(self.day, 12, 0), 999, self.store)
        source, observed = self.mature()
        update_candidate_outcomes(ids, source, self.spec['calendar_sessions'], observed, self.store)
        view = read_prior_asof(['AVGO'], observed + timedelta(seconds=1),
                               self.store.verify_chain(), self.store)
        Path(view['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            verify_prior_view(view['path'], self.store)
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            read_prior_asof(['AVGO'], observed + timedelta(seconds=1),
                            view['store_cut'], self.store)

    def test_source_proof_and_transaction_rollback(self):
        bad = dict(self.manifest(self.day), source_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'source proof'):
            register_candidates(self.universe, self.session(self.day), bad, self.store)
        leaked = self.manifest(self.day)
        leaked['rows'][0]['raw'] = {'3d_5pct': .99}
        with self.assertRaisesRegex(ValueError, 'may not inspect'):
            register_candidates(self.universe, self.session(self.day), leaked, self.store)
        routed = self.manifest(self.day)
        routed['rows'][0]['route_id'] = 'B_group'
        with self.assertRaisesRegex(ValueError, 'may not inspect'):
            register_candidates(self.universe, self.session(self.day), routed, self.store)
        self.assertEqual(self.store.verify_chain(), 0)
        self.register()
        changed = self.manifest(self.day)
        changed['rows'][0]['eligible'] = False
        changed['source_sha256'] = digest({k: v for k, v in changed.items() if k != 'source_sha256'})
        with self.assertRaisesRegex(ValueError, 'changed'):
            register_candidates(self.universe, self.session(self.day), changed, self.store)
        self.assertEqual(self.store.verify_chain(), 3)

    def test_late_receipt_controls_effective_availability(self):
        ids = self.register()['candidate_ids']
        source, observed = self.mature()
        receipt = observed + timedelta(days=1)
        source['AVGO']['received_at'] = receipt.isoformat()
        update_candidate_outcomes(ids, source, self.spec['calendar_sessions'],
                                  observed, self.store)
        cut = self.store.verify_chain()
        early = read_prior_asof(['AVGO'], observed + timedelta(seconds=1), cut, self.store)
        self.assertEqual(early['rows'], [])
        self.clock[0] = receipt + timedelta(seconds=1)
        later = read_prior_asof(['AVGO'], receipt + timedelta(seconds=1), cut, self.store)
        self.assertEqual(later['rows'][0]['label_available_at'], receipt.isoformat())

    def test_registered_source_bytes_revalidated_on_read(self):
        source_file = self.root / 'universe.json'
        body = {k: v for k, v in self.universe.items() if k != 'source_sha256'}
        source_file.write_text(canonical(body))
        universe = dict(body, path=str(source_file), source_sha256=file_sha(source_file))
        register_candidates(universe, self.session(self.day), self.manifest(self.day),
                            self.store)
        source_file.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'universe candidate source bytes changed'):
            read_prior_asof(['AVGO'], _at(self.day, 12, 0),
                            self.store.verify_chain(), self.store)

    def test_capture_at_113005_excludes_112030_revision_before_deadline(self):
        ids = self.register()['candidate_ids']
        source, observed = self.mature()
        update_candidate_outcomes(ids, source, self.spec['calendar_sessions'],
                                  observed, self.store)
        forecast_day = self.dates[6]
        self.clock[0] = _at(forecast_day, 11, 30, 5)
        deadline = _at(forecast_day, 11, 30, 30)
        original = read_prior_asof(['AVGO'], deadline, self.store.verify_chain(), self.store)
        self.assertEqual(original['observed_through_at'], self.clock[0].isoformat())
        self.assertEqual(original['information_deadline_at'], deadline.isoformat())
        self.assertEqual(len(original['rows']), 1)
        changed_bars, _ = full_bars(self.spec, self.day, symbol='AVGO', hit_at=-1)
        self.clock[0] = _at(forecast_day, 11, 30, 20)
        update_candidate_outcomes(ids, dict(source, AVGO=snapshot(changed_bars)),
                                  self.spec['calendar_sessions'], self.clock[0], self.store)
        old = verify_prior_view(original['path'], self.store)
        self.assertEqual(old['rows'], original['rows'])
        revised = read_prior_asof(['AVGO'], deadline, self.store.verify_chain(), self.store)
        self.assertNotEqual(revised['rows'][0]['y'], original['rows'][0]['y'])
        self.assertEqual(revised['observed_through_at'], self.clock[0].isoformat())

    def test_last_63_beta_prior_and_unique_candidate_ids(self):
        # Synthetic mature outcome events isolate view selection and the frozen Beta rule.
        universe = proven({'id': 'one-stock', 'members': ['AVGO']})
        self.clock[0] = _at(self.dates[0], 8, 0)
        ids = []
        for day in self.dates[:64]:
            manifest = proven({'rows': [{'symbol': 'AVGO', 'sample_id': f'AVGO:{day}',
                                         'eligible': True}]})
            ids.extend(register_candidates(universe, self.session(day), manifest,
                                           self.store)['candidate_ids'])
        self.clock[0] = _at('2026-06-01', 12, 0)
        with self.store.transaction():
            for i, (day, candidate_id) in enumerate(zip(self.dates[:64], ids)):
                result = {'sample_id': f'AVGO:{day}', 'symbol': 'AVGO',
                          'decision_at': _at(day, 11, 30).isoformat(),
                          'label_end_at': _at(day, 16, 0).isoformat(),
                          'label_available_at': _at(day, 16, 0).isoformat(),
                          'y': [i % 2] * 9, 'all_nine_mature': True, 'targets': {}}
                self.store.append('outcome_version', candidate_id, 'AVGO', day,
                                  {'result': result, 'source_sha256': 'a' * 64,
                                   'action_sha256': 'b' * 64,
                                   'calendar_sha256': 'c' * 64})
        view = read_prior_asof(['AVGO'], _at('2026-06-02', 12, 0),
                               self.store.verify_chain(), self.store)
        self.assertEqual(len(view['rows']), 64)
        prior, selected = _prior('AVGO', _at('2026-06-03', 11, 30, 30), view['rows'])
        self.assertEqual(len(selected), 63)
        self.assertEqual(selected[0], f'AVGO:{self.dates[1]}')
        self.assertAlmostEqual(float(prior[0]), (1 + 32) / (2 + 63))

    def test_history_import_cannot_backdate_or_inject_clock(self):
        with self.assertRaisesRegex(ValueError, 'cannot inject'):
            PriorStore(self.root / 'historical.sqlite', mode='historical_fixture',
                       clock=lambda: _at(self.day, 8, 0))
        history = PriorStore(self.root / 'historical.sqlite', mode='historical_fixture')
        try:
            view = read_prior_asof(['AVGO'], _at('2030-01-01', 12, 0), 0, history)
            self.assertEqual(view['rows'], [])
            self.assertEqual(view['observed_through_at'], view['captured_at'])
        finally:
            history.close()


if __name__ == '__main__':
    unittest.main()
