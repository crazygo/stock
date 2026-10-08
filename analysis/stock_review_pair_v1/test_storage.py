"""Real files verify deduplication, revision isolation, corruption and portability."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from common import digest, fingerprint, read, write
from storage import SharedData, hydrate, price_bars, resolve_ref, verify_market
from export import export_run
from run import verify_inputs
from views import trend_record
from data import Acquisition


class StorageTests(unittest.TestCase):
    def capture(self, run, close=2):
        bars = [['2026-10-01', 1., float(close), 1., float(close), 100.]]
        raw = run / 'inputs/market/US.TEST.parquet'
        raw.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(bars, columns=['day', 'open', 'high', 'low', 'close', 'volume']).to_parquet(raw)
        write(run / 'inputs/members.json', {'records': []})
        entry = {'security_id': 'US.TEST', 'price_basis': 'QFQ_full_series_single_capture',
                 'dataset_id': fingerprint(['US.TEST', bars]), 'bars': bars,
                 'path': 'inputs/market/US.TEST.parquet', 'sha256': digest(raw)}
        manifest = {'datasets': [entry], 'input_files': {'inputs/members.json': digest(run / 'inputs/members.json')},
                    'data_run_id': 'same-analysis-data'}
        write(run / 'inputs/data_manifest.json', manifest)
        return manifest

    def test_replay_deduplicates_and_runs_contain_no_price_files(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder);legacy = store / 'runs/legacy'
            original = self.capture(legacy);before = digest(legacy / 'inputs/data_manifest.json')
            first = store / 'runs/first';a = SharedData(store);m = a.replay(legacy, first, original)
            write(first / 'inputs/data_manifest.json', m);verify_inputs(first)
            self.assertNotIn('bars', m['datasets'][0]);self.assertNotIn('path', m['datasets'][0])
            self.assertEqual(list(first.rglob('*.parquet')), [])
            self.assertTrue((first / 'inputs/members.json').is_symlink())
            second = store / 'runs/second';b = SharedData(store);n = b.replay(first, second, m)
            write(second / 'inputs/data_manifest.json', n);verify_inputs(second)
            self.assertEqual(b.metrics['objects_created'], 0)
            self.assertEqual(b.metrics['bytes_created'], 0)
            self.assertEqual(hydrate(second, n)['datasets'][0]['bars'], original['datasets'][0]['bars'])
            self.assertEqual(digest(legacy / 'inputs/data_manifest.json'), before)
            self.assertEqual(n['data_run_id'], original['data_run_id'])

    def test_revision_keeps_previous_price_version(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder);a = store / 'runs/a';b = store / 'runs/b'
            one, two = self.capture(a, 2), self.capture(b, 3)
            pool = SharedData(store)
            for run, old, name in [(a, one, 'one'), (b, two, 'two')]:
                target = store / 'runs' / name;m = pool.replay(run, target, old)
                write(target / 'inputs/data_manifest.json', m)
            self.assertNotEqual(read(store / 'runs/one/inputs/data_manifest.json')['datasets'][0]['object_ref'],
                                read(store / 'runs/two/inputs/data_manifest.json')['datasets'][0]['object_ref'])
            self.assertEqual(price_bars(store / 'runs/one', read(store / 'runs/one/inputs/data_manifest.json')['datasets'][0])[0][4], 2.)
            self.assertEqual(len(list((store / 'data/daily').rglob('*.json'))), 2)

    def test_corruption_stops_replay_without_repairing_old_object(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder);legacy = store / 'runs/legacy';old = self.capture(legacy)
            run = store / 'runs/new';pool = SharedData(store);m = pool.replay(legacy, run, old)
            write(run / 'inputs/data_manifest.json', m)
            obj = resolve_ref(run, m['datasets'][0]['object_ref']);obj.chmod(0o644);obj.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):verify_inputs(run)
            with self.assertRaisesRegex(ValueError, 'corrupt'):pool.replay(legacy, store / 'runs/retry', old)
            self.assertEqual(obj.read_text(), '{}')

    def test_export_survives_missing_shared_library_and_manifest_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder) / 'store';legacy = store / 'runs/legacy';old = self.capture(legacy)
            run = store / 'runs/new';m = SharedData(store).replay(legacy, run, old)
            write(run / 'inputs/data_manifest.json', m)
            write(run / 'checksums.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file()})
            exported = export_run(run, Path(folder) / 'portable')
            (store / 'data').rename(store / 'data-unavailable')
            verify_inputs(exported)
            self.assertEqual(digest(exported / 'inputs/data_manifest.json'), digest(run / 'inputs/data_manifest.json'))
            self.assertEqual(hydrate(exported, m)['datasets'][0]['bars'], old['datasets'][0]['bars'])
            self.assertTrue(all(not p.is_symlink() for p in exported.rglob('*')))

    def test_force_recompute_does_not_take_cached_values(self):
        with tempfile.TemporaryDirectory() as folder:
            entry = {'security_id': 'US.TEST', 'dataset_id': 'version', 'bars': []}
            with patch('views.at', return_value=([None]*5, ['insufficient_history']*5, None)) as calculate:
                trend_record(entry, [], '2026-10-01', Path(folder));calculate.reset_mock()
                _, cached = trend_record(entry, [], '2026-10-01', Path(folder));self.assertTrue(cached);calculate.assert_not_called()
                _, cached = trend_record(entry, [], '2026-10-01', Path(folder), recompute=True)
                self.assertFalse(cached);calculate.assert_called_once()

    def test_acquisition_reuses_complete_local_series_and_saves_only_references(self):
        from types import SimpleNamespace
        import time
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder);legacy = store / 'runs/legacy';old = self.capture(legacy)
            args = SimpleNamespace(store=store, offline=True, no_opend=True)
            cal = {'US': {'start': '2026-01-01', 'days': ['2026-10-01']}}
            run = store / 'runs/new'
            engine = Acquisition(args, run, cal, {'US': '2026-10-01'}, time.monotonic()+1)
            try:
                with patch.object(engine, 'r2', side_effect=AssertionError('unexpected R2 call')), \
                     patch.object(engine, 'opend', side_effect=AssertionError('unexpected OpenD call')):
                    entry = engine.acquire({'code': 'US.TEST', 'listing_date': '2026-10-01'})
                self.assertEqual(entry['coverage']['status'], 'calendar_partial')
                self.assertEqual(price_bars(run, entry), old['datasets'][0]['bars'])
                self.assertEqual(list(run.rglob('*.parquet')), [])
            finally:
                engine.cleanup()

    def test_price_reference_cannot_claim_another_security(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Path(folder);legacy = store / 'runs/legacy';old = self.capture(legacy)
            run = store / 'runs/new';m = SharedData(store).replay(legacy, run, old)
            m['datasets'][0]['security_id'] = 'US.OTHER'
            with self.assertRaisesRegex(ValueError, 'identity'):verify_market(run, m)

    def test_publisher_checks_shared_objects_before_switching_pages(self):
        from publish import TARGETS, publish
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);store = root / '.cache/store';legacy = store / 'runs/legacy';old = self.capture(legacy)
            run = store / 'runs/new';m = SharedData(store).replay(legacy, run, old)
            write(run / 'inputs/data_manifest.json', m)
            for name, target in TARGETS.items():
                p = root / name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('previous')
                p = run / 'pages' / target;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('candidate')
            write(run / 'validation/acceptance.json', {'status': 'passed'})
            write(run / 'checksums.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file()})
            obj = resolve_ref(run, m['datasets'][0]['object_ref']);obj.chmod(0o644);obj.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum'):publish(store, run, root)
            self.assertTrue(all((root/name).read_text() == 'previous' for name in TARGETS))


if __name__ == '__main__':
    unittest.main()
