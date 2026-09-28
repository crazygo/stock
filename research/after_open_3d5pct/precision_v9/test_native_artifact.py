"""Bounded native-artifact counterexamples and existing-engine interface replay."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

# The formal inference module must be imported before any torch-dependent code.
from .forward.inference import load_route, predict_route
from .forward.schema import build_schema
from . import native_artifact as native


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENGINE = HERE.parent / 'runs/precision_v9_baseline_engine_smoke_20260927'
SUMMARY = json.loads((HERE.parent / 'runs/focus_v8_20260927/round_10/summary.json').read_text())


class NativeArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='native-artifact-tests-')
        cls.root = Path(cls.tmp.name)
        cls.exports = {}
        for route in ('B_no_group', 'C_no_group'):
            run = ENGINE / f'{route}_dev1'
            cls.exports[route] = native.export_posthoc_engineering(
                run, cls.root / f'{route}_export')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_five_route_three_arm_schema_identity(self):
        for route in native.ROUTES:
            for arm in native.ARMS:
                semantics = native.peer_semantics(route, arm)
                recipe = SUMMARY[route]['incumbent_recipe']
                schema = build_schema(route, recipe, semantics)
                self.assertEqual(schema['route_id'], route)
                self.assertEqual(schema['support_rule'],
                                 'checkpoint_buffers' if recipe.get('support') else 'none')
                self.assertEqual(schema['tabular_width'],
                                 765 if route in native.GROUP_ROUTES else 585)
                if route in native.GROUP_ROUTES:
                    self.assertEqual(schema['group_peer_semantics'], semantics)
                else:
                    self.assertNotIn('group_peer_semantics', schema)
        with self.assertRaisesRegex(ValueError, 'exclude'):
            build_schema('B_no_group', SUMMARY['B_no_group']['incumbent_recipe'], native.CAUSAL)
        with self.assertRaisesRegex(ValueError, 'explicit'):
            build_schema('B_group', SUMMARY['B_group']['incumbent_recipe'], 'excluded')

    def test_actual_posthoc_engineering_child_replays_original_eval_and_singleton(self):
        for route in ('B_no_group', 'C_no_group'):
            with self.subTest(route=route):
                selection = native.write_new(
                    self.root / f'{route}_selected.json',
                    {'sample_ids': ['AMD:2026-06-01:11:30:v6'],
                     'selection_rule': 'fixed_registered_old_engine_anchor'})
                final = native.verify_in_child(
                    self.exports[route], ENGINE/f'{route}_dev1', selection['path'],
                    self.root/f'{route}_child')
                actual = json.loads((self.root/f'{route}_child/child_result.json').read_text())
                self.assertTrue(final['pass'])
                self.assertEqual(final['source_drift'], [])
                self.assertTrue(actual['pass'])
                self.assertEqual(actual['eval_rows'], 1975)
                self.assertEqual(len(actual['singleton']), 1)
                self.assertFalse(actual['singleton'][0]['publication_eligible'])

    def test_manifest_recipe_threshold_calibration_and_causal_claim_cannot_be_rewritten(self):
        source = json.loads(self.exports['C_no_group'].read_text())
        wrong = deepcopy(source)
        wrong['thresholds']['chips'] = .8
        with self.assertRaisesRegex(ValueError, 'threshold'):
            load_route(wrong)
        wrong = deepcopy(source)
        wrong['recipe']['curves'] = True
        with self.assertRaisesRegex(ValueError, 'recipe'):
            load_route(wrong)
        with tempfile.TemporaryDirectory() as d:
            calibrated = json.loads(Path(source['calibration_file']['path']).read_text())
            calibrated['parameters'][0][0] += .1
            other = native.write_new(Path(d)/'calibration.json', calibrated)
            wrong = deepcopy(source)
            wrong['calibration_file'] = other
            wrong['calibration_sha256'] = other['sha256']
            with self.assertRaisesRegex(ValueError, 'calibration'):
                load_route(wrong)
        wrong = deepcopy(source)
        wrong['schema']['group_peer_semantics'] = native.CAUSAL
        with self.assertRaisesRegex(ValueError, 'schema'):
            load_route(wrong)

    def test_loaded_manifest_seal_and_caller_made_dict_cannot_bypass_provenance(self):
        source = json.loads(self.exports['C_no_group'].read_text())
        loaded = load_route(source)
        loaded['manifest']['thresholds']['chips'] = .8
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            predict_route(loaded, {})
        forged = {'manifest': deepcopy(source), 'model': object(), 'artifacts': []}
        forged['manifest']['thresholds']['chips'] = .8
        with self.assertRaisesRegex(ValueError, 'threshold'):
            predict_route(forged, {})

    def test_rehashed_posthoc_identity_cannot_change_fold_or_seed(self):
        source = json.loads(self.exports['B_no_group'].read_text())
        original = json.loads(Path(source['native_provenance_file']['path']).read_text())
        for field, value in (('fold', 'dev2'), ('seed', 1234)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                altered = deepcopy(original)
                altered['identity'][field] = value
                ref = native.write_new(Path(d)/'provenance.json', altered)
                wrong = deepcopy(source)
                wrong['native_provenance_file'] = ref
                wrong['native_provenance_sha256'] = ref['sha256']
                with self.assertRaisesRegex(ValueError, 'posthoc'):
                    load_route(wrong)

    def test_rehashed_alternative_causal_membership_cannot_replace_prefit_source(self):
        with tempfile.TemporaryDirectory() as d:
            original = native.write_new(Path(d)/'registered_membership.json',
                                        [{'symbol': 'AMD', 'group_ids': ['trend_15:up']}])
            other = native.write_new(Path(d)/'other_real_membership.json',
                                     [{'symbol': 'AMD', 'group_ids': ['trend_15:down']}])
            contract = {'selected_dataset': {'dataset_sha256': 'a'*64},
                        'causal_sources': {'membership_sha256': original['sha256'],
                                           'references': {'native_group_memberships': original},
                                           'source_hashes': {'membership': original}}}
            accepted = {'dataset_sha256': 'a'*64,
                        'membership_sha256': original['sha256'],
                        'references': {'native_group_memberships': original},
                        'source_hashes': {'membership': original}}
            self.assertEqual(native._verify_causal_export(contract, accepted), accepted)
            replaced = deepcopy(accepted)
            replaced['membership_sha256'] = other['sha256']
            replaced['references']['native_group_memberships'] = other
            replaced['source_hashes']['membership'] = other
            self.assertEqual(native.verify_record(other), other)
            with self.assertRaisesRegex(ValueError, 'pre-fit training source contract'):
                native._verify_causal_export(contract, replaced)

    def test_model_and_copied_source_bytes_are_verified(self):
        source = json.loads(self.exports['B_no_group'].read_text())
        wrong = deepcopy(source)
        wrong['model_files'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'model'):
            load_route(wrong)
        with tempfile.TemporaryDirectory() as d:
            artifact = native.export_posthoc_engineering(
                ENGINE/'B_no_group_dev1', Path(d)/'copy')
            manifest = json.loads(artifact.read_text())
            copied = Path(manifest['source_files'][0]['path'])
            copied.write_bytes(copied.read_bytes() + b'changed')
            with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                load_route(manifest)
            selection = native.write_new(Path(d)/'selected.json',
                                         {'sample_ids': ['AMD:2026-06-01:11:30:v6']})
            with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
                native.verify_in_child(artifact, ENGINE/'B_no_group_dev1',
                                       selection['path'], Path(d)/'failed_child')
            final = json.loads((Path(d)/'failed_child/child_final_status.json').read_text())
            self.assertFalse(final['pass'])
            self.assertEqual(final['stage'], 'preflight')

    def test_failed_preflight_keeps_terminal_state_without_launch(self):
        with self.assertRaisesRegex(ValueError, 'reference run'):
            native.verify_in_child(self.exports['B_no_group'], ENGINE/'C_no_group_dev1',
                                   self.exports['C_no_group'], self.root/'wrong_run_child')
        final = json.loads((self.root/'wrong_run_child/child_final_status.json').read_text())
        self.assertFalse(final['pass'])
        self.assertEqual(final['stage'], 'preflight')
        self.assertIsNone(final['exit_code'])
        self.assertFalse((self.root/'wrong_run_child/stdout.txt').exists())

    def test_prior_parquet_list_cells_normalize_without_ambiguous_array_comparison(self):
        path = ENGINE/'C_no_group_dev1/prior_history_ids.parquet'
        original = pd.read_parquet(path)
        rows = native.prior_history_rows(path)
        selected = [i for i, n in enumerate(rows['counts']) if n >= 2][:2]
        self.assertEqual(len(selected), 2)
        for i in selected:
            self.assertEqual(rows['history_ids'][i], list(original.prior_history_ids.iloc[i]))
            self.assertIsInstance(rows['history_ids'][i], list)

    def _sparse_causal_fixture(self, base):
        selected, old = base/'selected', base/'old'
        inputs = selected/'inputs'
        group = inputs/'market_data/groups_v9'
        group.mkdir(parents=True)
        (inputs/'market_data/universe').mkdir(parents=True)
        (inputs/'market_data/calendars').mkdir(parents=True)
        (selected/'dataset').mkdir()
        (old/'inputs/market_data/universe').mkdir(parents=True)
        candidates = [f'S{i:03d}' for i in range(108)]
        universe = {'members': [{'symbol': name, 'role': 'candidate'} for name in candidates] +
                               [{'symbol': 'QQQ', 'role': 'benchmark'}]}
        old_universe = old/'inputs/market_data/universe/pool.json'
        native.write_new(old_universe, universe)
        native.write_new(old/'config.json', {'universe': 'market_data/universe/pool.json'})
        universe_path = inputs/'market_data/universe/pool.json'
        native.write_new(universe_path, universe)
        calendar = inputs/'market_data/calendars/calendar.json'
        native.write_new(calendar, {'calendar_id': 'official_fixture_v1',
            'sources': ['https://nasdaqtrader.com/Trader.aspx?id=Calendar'],
            'sessions': [
                {'session_date': '2026-05-29', 'open_at': '2026-05-29T13:30:00+00:00',
                 'close_at': '2026-05-29T20:00:00+00:00', 'duration_minutes': 390},
                {'session_date': '2026-06-01', 'open_at': '2026-06-01T13:30:00+00:00',
                 'close_at': '2026-06-01T20:00:00+00:00', 'duration_minutes': 390}]})
        versions = [{'week_id': '2026-W23', 'strategy_id': 'trend_15',
                     'version_id': 'v9:trend_15:2026-W23',
                     'membership_basis': 'causal_weekly_reconstruction',
                     'first_session': '2026-06-01',
                     'effective_from': '2026-06-01T13:30:00+00:00',
                     'effective_to': '2026-06-08T13:30:00+00:00',
                     'feature_cutoff_at': '2026-06-01T13:30:00+00:00'}]
        members = [{'week_id': '2026-W23', 'strategy_id': 'trend_15',
                    'symbol': candidates[0], 'version_id': versions[0]['version_id'],
                    'group_ids': ['trend_15:up'],
                    'facts': {'history_end': '2026-05-29'}}]
        native.write_new(group/'versions.json', versions)
        native.write_new(group/'memberships.json', members)
        native.write_new(group/'groups.json', [{'group_id': 'trend_15:up'}])
        config = {'group_run': 'market_data/groups_v9',
                  'universe': 'market_data/universe/pool.json',
                  'calendar': 'market_data/calendars/calendar.json',
                  'feature_start': '2026-06-01', 'data_end': '2026-06-01'}
        native.write_new(selected/'config.json', config)
        native.write_new(selected/'source_provenance.json', [])
        paths = [group/'versions.json', group/'memberships.json', group/'groups.json',
                 universe_path, calendar]
        source = {str(path.relative_to(inputs)): native.sha(path) for path in paths}
        manifest = {'builder_sha256': native.sha(HERE/'data.py'),
                    'group_quality': 'causal_weekly_reconstruction_on_current_universe_not_original_PIT',
                    'source_hashes': source}
        def refresh():
            (selected/'dataset/manifest.json').write_bytes(native.canonical(manifest))
            complete = {'builder_sha': native.sha(HERE/'data.py'),
                        'prepare_sha': native.sha(HERE/'prepare.py'),
                        'manifest_sha': native.sha(selected/'dataset/manifest.json')}
            (selected/'complete.json').write_bytes(native.canonical(complete))
        refresh()
        plan = {'old_dataset_root': str(old),
                'native_group_versions': str(group/'versions.json'),
                'native_group_memberships': str(group/'memberships.json'),
                'native_candidate_universe': str(universe_path),
                'native_builder_source': str(HERE/'data.py'),
                'native_source_root': str(inputs)}
        return selected, plan, manifest, members, universe, versions, calendar, refresh

    def test_sparse_real_builder_contract_and_wrong_pool_or_group_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            selected, plan, manifest, members, universe, versions, calendar, refresh = \
                self._sparse_causal_fixture(Path(d))
            accepted = native._causal_sources(plan, selected)
            self.assertEqual(accepted['candidate_count'], 108)
            self.assertEqual(accepted['membership_count'], 1)
            self.assertEqual(accepted['unclassified_count'], 107)
            missing = dict(plan); missing.pop('native_group_versions')
            with self.assertRaisesRegex(ValueError, 'lacks actual source paths'):
                native._causal_sources(missing, selected)
            wrong = dict(plan); wrong['native_source_root'] = str(Path(d))
            with self.assertRaisesRegex(ValueError, 'native_source_root'):
                native._causal_sources(wrong, selected)
            universe['members'][0]['symbol'] = 'ZZZZ'
            Path(plan['native_candidate_universe']).write_bytes(native.canonical(universe))
            key = str(Path(plan['native_candidate_universe']).relative_to(selected/'inputs'))
            manifest['source_hashes'][key] = native.sha(plan['native_candidate_universe'])
            refresh()
            with self.assertRaisesRegex(ValueError, 'frozen 108-member'):
                native._causal_sources(plan, selected)
            universe['members'][0]['symbol'] = 'S000'
            Path(plan['native_candidate_universe']).write_bytes(native.canonical(universe))
            manifest['source_hashes'][key] = native.sha(plan['native_candidate_universe'])
            members[0]['group_ids'] = ['trend_15:invented']
            Path(plan['native_group_memberships']).write_bytes(native.canonical(members))
            key = str(Path(plan['native_group_memberships']).relative_to(selected/'inputs'))
            manifest['source_hashes'][key] = native.sha(plan['native_group_memberships'])
            refresh()
            with self.assertRaisesRegex(ValueError, 'category'):
                native._causal_sources(plan, selected)

    def test_causal_clock_requires_calendar_session_and_version_fields(self):
        for defect in ('empty_calendar', 'missing_cutoff', 'sunday_history'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as d:
                selected, plan, manifest, members, universe, versions, calendar, refresh = \
                    self._sparse_causal_fixture(Path(d))
                if defect == 'empty_calendar':
                    doc = json.loads(calendar.read_text()); doc['sessions'] = []
                    calendar.write_bytes(native.canonical(doc))
                    changed = calendar
                elif defect == 'missing_cutoff':
                    versions[0].pop('feature_cutoff_at')
                    changed = Path(plan['native_group_versions'])
                    changed.write_bytes(native.canonical(versions))
                else:
                    members[0]['facts']['history_end'] = '2026-05-31'
                    changed = Path(plan['native_group_memberships'])
                    changed.write_bytes(native.canonical(members))
                key = str(changed.relative_to(selected/'inputs'))
                manifest['source_hashes'][key] = native.sha(changed)
                refresh()
                with self.assertRaisesRegex(ValueError, 'calendar|version|history_end'):
                    native._causal_sources(plan, selected)


if __name__ == '__main__':
    unittest.main()
