"""Offline adapter tests: real historical refusal plus synthetic denominator."""
from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .evaluate import evaluate_cohort
from .inference import _sha
from .ledger import Ledger, _at, canonical, digest, issue_once, register_cohort, route_contract
from .labels import mature_signals
from .local_adapter import load_local_bundle, run_local_e2e
from . import local_adapter
from .prior_store import PriorStore, read_prior_asof, register_candidates
from .test_inference import SUMMARY, V8, manifest as route_manifest
from .test_labels import full_bars, snapshot
from .test_ledger import fixture, prediction


def save_json(path, value):
    path.write_text(canonical(value))
    return {'path': str(path), 'sha256': _sha(path)}


class LocalAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dates, self.spec, self.route, self.cohort, self.artifacts = fixture(self.root)
        self.day = '2026-06-01'

    def tearDown(self):
        self.temp.cleanup()

    def _historical_bundle(self):
        source_calendar = V8 / 'inputs/market_data/calendars/nasdaq_sessions_2026_v1.json'
        sessions = json.loads(source_calendar.read_text())['sessions']
        official = next(x for x in sessions if x['session_date'] == self.day)
        universe_body = {'universe_version': 'adapter-test-universe',
                         'members': [{'symbol': 'AMD', 'role': 'candidate'},
                                     {'symbol': 'COHR', 'role': 'candidate'},
                                     {'symbol': 'QQQ', 'role': 'benchmark'}]}
        universe_ref = save_json(self.root / 'prior_universe.json', universe_body)
        universe = dict(universe_body, path=universe_ref['path'],
                        source_sha256=universe_ref['sha256'])
        eligibility_body = {'rows': [
            {'symbol': symbol, 'sample_id': f'{symbol}:{self.day}', 'eligible': True}
            for symbol in ('AMD', 'COHR')]}
        eligibility_ref = save_json(self.root / 'eligibility.json', eligibility_body)
        eligibility = dict(eligibility_body, path=eligibility_ref['path'],
                           source_sha256=eligibility_ref['sha256'])
        prior = PriorStore(self.root / 'prior.sqlite', mode='historical_fixture')
        register_candidates(universe, official, eligibility, prior)
        view = read_prior_asof(['AMD'], _at(self.day, 11, 30, 30),
                               prior.verify_chain(), prior)
        forecast_ref = save_json(self.root / 'forecast_candidates.json',
                                 {'rows': [{'symbol': 'AMD',
                                            'sample_id': f'AMD:{self.day}',
                                            'eligible': True}]})
        dependency_ref = save_json(self.root / 'required_market_dependencies.json',
                                   {'routes': {'B_no_group': {'AMD': ['AMD']}}})
        action_ref = save_json(self.root / 'amd_actions.json',
                               {'coverage_start': '2026-01-01',
                                'coverage_end': '2026-12-31',
                                'split_dates': [], 'halt_dates': [],
                                'coverage_verified': False})
        bar_path = V8 / 'inputs/market_data/us_5m/AMD/2026.parquet'
        input_doc = {'mode': 'historical_fixture', 'session_date': self.day,
                     'feature_calendar_start': '2026-01-01',
                     'prior_universe': universe_ref,
                     'forecast_candidates': forecast_ref,
                     'required_market_dependencies': dependency_ref,
                     'calendar': {'path': str(source_calendar),
                                  'sha256': _sha(source_calendar)},
                     'bars': {'AMD': {'path': str(bar_path), 'sha256': _sha(bar_path)}},
                     'corporate_actions': {'AMD': action_ref}}
        input_ref = save_json(self.root / 'input_manifest.json', input_doc)
        return prior, view, input_ref, sessions, input_doc

    def test_group_universe_matches_full_registered_candidates(self):
        prior, view, _, _, input_doc = self._historical_bundle()
        try:
            for name, candidates in (
                    ('complete', ['AMD', 'COHR']),
                    ('missing_peer', ['AMD']),
                    ('extra', ['AMD', 'COHR', 'XYZ']),
                    ('duplicate', ['AMD', 'COHR', 'COHR']),
                    ('benchmark', ['AMD', 'COHR', 'QQQ'])):
                with self.subTest(name=name):
                    groups = {kind: save_json(self.root / f'{name}_{kind}.json', value)
                              for kind, value in (
                                  ('versions', []), ('memberships', []),
                                  ('universe_symbols', candidates))}
                    manifest = dict(input_doc, groups=groups)
                    ref = save_json(self.root / f'{name}_manifest.json', manifest)
                    if name == 'complete':
                        bundle = load_local_bundle(ref, view['path'],
                                                   'historical_fixture', prior)
                        self.assertEqual(bundle['metadata']['universe_symbols'],
                                         ['AMD', 'COHR'])
                        self.assertEqual({r['symbol'] for r in bundle['forecast_rows']},
                                         {'AMD'})
                        self.assertNotIn('QQQ', bundle['metadata']['universe_symbols'])
                    else:
                        with self.assertRaisesRegex(ValueError,
                                                    'group universe differs'):
                            load_local_bundle(ref, view['path'],
                                              'historical_fixture', prior)
        finally:
            prior.close()

    def test_three_lists_and_real_amd_history_rejects_publication(self):
        prior, view, input_ref, sessions, input_doc = self._historical_bundle()
        try:
            bundle = load_local_bundle(input_ref, view['path'], 'historical_fixture', prior)
            self.assertEqual(len(bundle['forecast_rows']), 1)
            self.assertEqual({x['symbol'] for x in bundle['forecast_rows']}, {'AMD'})
            self.assertEqual(len([e for e in prior.events(event_type='candidate_registered')]), 2)
            self.assertEqual(bundle['dependencies']['B_no_group']['AMD'], ['AMD'])
            self.assertEqual(bundle['prior_view']['rows'], [])  # imported now, not in June
            self.assertFalse(bundle['evidence']['real_receipt_verified'])
            model_dir = Path(SUMMARY['B_no_group']['incumbent_paths'][0])
            frozen = route_manifest('B_no_group', model_dir)
            route_ref = save_json(self.root / 'route.json', frozen)
            dates = [s['session_date'] for s in sessions if s['session_date'] >= self.day][:60]
            self.assertEqual(len(dates), 60)
            spec = {'calendar_sessions': sessions,
                    'groups': {'chips': ['AMD'], 'optics': ['AMD'], 'storage': ['AMD']}}
            route = {'route_id': frozen['route_id'], 'model_version': frozen['model_version'],
                     'score_column': frozen['score_column'],
                     'thresholds': frozen['thresholds'],
                     'artifacts': {name: frozen[name] for name in
                                   ('model_sha256', 'schema_sha256',
                                    'calibration_sha256', 'source_code_sha256')}}
            cohort = {'cohort_id': 'local-adapter-fixture', 'session_dates': dates,
                      'calendar_sessions': sessions,
                      'routes': ['B_group', 'B_no_group', 'C_group', 'C_no_group',
                                 'C_no_daily'],
                      'groups': ['chips', 'optics', 'storage'],
                      'last_session_close_at': next(s['close_at'] for s in sessions
                                                    if s['session_date'] == dates[-1]),
                      'evidence_mode': 'fixture', 'bootstrap_repeats': 40}
            cohort['route_contracts'] = {name: route_contract(route)
                                         for name in cohort['routes']}
            now = [_at('2026-05-29', 12, 0)]
            ledger = Ledger(self.root / 'fixture_forward.sqlite', clock=lambda: now[0])
            try:
                register_cohort(cohort, ledger)
                now[0] = _at(self.day, 11, 30, 10)
                plan = {'mode': 'historical_fixture', 'session_date': self.day,
                        'input_manifest': input_ref, 'prior_view_path': view['path'],
                        'route_manifest': route_ref, 'spec': spec, 'route': route,
                        'cohort': cohort, 'run_id': 'historical-amd-fixture'}
                report = run_local_e2e(plan, self.root / 'output', ledger, prior)
                self.assertEqual(report['forecast_denominator'], 1)
                self.assertEqual(report['counts'], {'ledger_refusal': 1})
                self.assertFalse(report['gates']['g2_real_receipts'])
                self.assertFalse(report['gates']['g1_numerical_parity'])
                self.assertEqual(report['input_hashes_before'], report['input_hashes_after'])
                self.assertIn('actions:AMD', report['input_hashes_after'])
                self.assertEqual(json.loads((self.root / 'output/ledger_audit.json').read_text())
                                 [0]['ledger_reason'], 'data_unverified')
                self.assertEqual(len(ledger.events(event_type='buy')), 0)
                (self.root / 'route.json').write_text('{}')
                unavailable = run_local_e2e(plan, self.root / 'model_damaged_output',
                                            ledger, prior)
                self.assertEqual(unavailable['counts'], {'rejected': 1})
                rejected = json.loads((self.root / 'model_damaged_output/preledger_audit.json').read_text())
                self.assertEqual(rejected[0]['rejections'], ['model_unavailable'])
                save_json(self.root / 'route.json', frozen)
                bad_source = dict(input_doc, bars={'AMD': dict(input_doc['bars']['AMD'],
                                                                sha256='0' * 64)})
                bad_ref = save_json(self.root / 'bad_input_manifest.json', bad_source)
                bad_plan = dict(plan, input_manifest=bad_ref)
                no_bar = run_local_e2e(bad_plan, self.root / 'bar_damaged_output',
                                       ledger, prior)
                self.assertEqual(no_bar['counts'], {'rejected': 1})
                damaged_audit = json.loads((self.root / 'bar_damaged_output/preledger_audit.json').read_text())
                self.assertEqual(damaged_audit[0]['rejections'],
                                 ['required_market_source_missing'])
                self.assertEqual(len(ledger.events(event_type='run_empty')), 0)
                qqq_path = V8 / 'inputs/market_data/us_5m/QQQ/2026.parquet'
                qqq_action = save_json(self.root / 'qqq_actions.json',
                                       json.loads((self.root / 'amd_actions.json').read_text()))
                extra_deps = save_json(self.root / 'extra_deps.json',
                                       {'routes': {'B_no_group': {'AMD': ['AMD', 'QQQ']}}})
                extra_doc = dict(input_doc,
                                 required_market_dependencies=extra_deps,
                                 bars=dict(input_doc['bars'],
                                           QQQ={'path': str(qqq_path), 'sha256': _sha(qqq_path)}),
                                 corporate_actions=dict(input_doc['corporate_actions'],
                                                        QQQ=qqq_action))
                extra_ref = save_json(self.root / 'extra_input_manifest.json', extra_doc)
                extra_plan = dict(plan, input_manifest=extra_ref)
                excess = run_local_e2e(extra_plan, self.root / 'extra_dependency_output',
                                       ledger, prior)
                self.assertEqual(excess['counts'], {'rejected': 1})
                excess_audit = json.loads((self.root / 'extra_dependency_output/preledger_audit.json').read_text())
                self.assertEqual(excess_audit[0]['rejections'],
                                 ['required_market_dependency_mismatch'])
                action_path = self.root / 'amd_actions.json'
                original_action = action_path.read_bytes()
                original_issue = local_adapter.issue_once

                def changed_after_ledger(*args, **kwargs):
                    result = original_issue(*args, **kwargs)
                    action_path.write_text('{}')
                    return result

                with patch.object(local_adapter, 'issue_once', changed_after_ledger):
                    with self.assertRaisesRegex(ValueError, 'actions:AMD.*changed'):
                        run_local_e2e(plan, self.root / 'postledger_damage', ledger, prior)
                fatal = json.loads((self.root / 'postledger_damage/fatal.json').read_text())
                self.assertNotEqual(fatal['input_hashes_before']['actions:AMD']['sha256'],
                                    fatal['input_hashes_after']['actions:AMD']['sha256'])
                action_path.write_bytes(original_action)
                original_seal_sha = prior.db.execute(
                    "SELECT payload_sha FROM events WHERE event_type='view_created'").fetchone()[0]

                def damaged_seal_after_ledger(*args, **kwargs):
                    result = original_issue(*args, **kwargs)
                    prior.db.execute('DROP TRIGGER no_prior_update')
                    prior.db.execute("UPDATE events SET payload_sha=? WHERE event_type='view_created'",
                                     ('0' * 64,))
                    return result

                with patch.object(local_adapter, 'issue_once', damaged_seal_after_ledger):
                    with self.assertRaisesRegex(ValueError, 'seal or hash chain'):
                        run_local_e2e(plan, self.root / 'prior_seal_damage', ledger, prior)
                sealed_failure = json.loads((self.root / 'prior_seal_damage/fatal.json').read_text())
                self.assertEqual(sealed_failure['input_hashes_before']['prior_view']['sha256'],
                                 sealed_failure['input_hashes_after']['prior_view']['sha256'])
                self.assertIn('seal_error', sealed_failure['input_hashes_after']['prior_view'])
                prior.db.execute("UPDATE events SET payload_sha=? WHERE event_type='view_created'",
                                 (original_seal_sha,))
            finally:
                ledger.close()
            # A market-data dependency is separate from the forecast list.
            altered = json.loads((self.root / 'required_market_dependencies.json').read_text())
            altered['routes']['B_no_group']['AMD'].append('QQQ')
            save_json(self.root / 'required_market_dependencies.json', altered)
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                load_local_bundle(input_ref, view['path'], 'historical_fixture', prior)
        finally:
            prior.close()

    def test_five_synthetic_buys_full_labels_and_sixty_session_denominator(self):
        day = self.dates[0]
        now = [_at(day, 8, 0)]
        ledger = Ledger(self.root / 'synthetic_forward.sqlite', clock=lambda: now[0])
        try:
            register_cohort(self.cohort, ledger)
            now[0] = _at(day, 11, 30, 10)
            symbols = ['AVGO', 'AMD', 'NVDA', 'INTC', 'QCOM']
            buys = issue_once(self.spec, self.route, self.cohort, 'synthetic-five',
                              [prediction(s, day, self.route, self.artifacts)
                               for s in symbols], {'g2_eligible': True}, ledger)
            self.assertTrue(all(b['event_type'] == 'buy' for b in buys))
            sources = {}
            end = None
            for index, symbol in enumerate(symbols):
                bars, end = full_bars(self.spec, day, symbol=symbol,
                                      hit_at=10 if index < 2 else -1)
                if index == 3:
                    future = end + timedelta(days=2)
                    bars[200] = dict(bars[200], available_at=future.isoformat(),
                                     received_at=future.isoformat())
                if index == 4:
                    bars.pop(3)
                sources[symbol] = snapshot(bars)
            observed = end + timedelta(seconds=3)
            outcomes = mature_signals([b['event_id'] for b in buys], sources,
                                      self.spec['calendar_sessions'], observed,
                                      ledger=ledger)
            self.assertEqual(len(outcomes), 45)
            now[0] = _at(self.dates[61], 17, 0)
            report = evaluate_cohort(self.cohort['cohort_id'], ledger.verify_chain(),
                                     self.cohort, ledger=ledger)
            cell = report['cells']['B_group:chips']['main']
            self.assertEqual(report['official_sessions'], 60)
            self.assertEqual((cell['issued'], cell['tp'], cell['fp'], cell['unresolved']),
                             (5, 2, 1, 2))
            self.assertEqual(cell['unresolved_by_reason'], {'pending': 1, 'missing': 1})
            self.assertFalse(report['final_target_pass'])
        finally:
            ledger.close()


if __name__ == '__main__':
    unittest.main()
