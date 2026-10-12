"""Verify classification, missing-data and immutable-input behavior offline."""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

SCRIPT = Path(__file__).with_name('build_context.py')
SPEC = importlib.util.spec_from_file_location('insight_context', SCRIPT)
CTX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CTX)
REPO = SCRIPT.resolve().parents[4]
DAYS = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(70)]


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fixture(root, shared=False, errors=None):
    run = root / 'runs/batch'
    run.mkdir(parents=True)
    members = [
        {'code': 'US.HELD', 'kind': 'STOCK', 'groups': ['holdings', 'watchlist'], 'chains': ['compute']},
        {'code': 'US.WATCH', 'kind': 'STOCK', 'groups': ['watchlist', 'special'], 'chains': ['compute']},
        {'code': 'US.AIONLY', 'kind': 'STOCK', 'groups': ['ai_a'], 'chains': ['compute']},
        {'code': 'US.SPY', 'kind': 'ETF', 'groups': ['watchlist'], 'chains': []},
    ]
    entries = []
    for m in members:
        bars = [[d, 100 + i, 100 + i, 100 + i, 100 + i, 10] for i, d in enumerate(DAYS)]
        entry = {'security_id': m['code'], 'market': 'US', 'dataset_id': hashed([m['code'], bars]),
                 'price_basis': 'QFQ', 'coverage': {'actual_end': DAYS[-1], 'status': 'complete'}}
        if shared:
            obj = root / 'data/daily' / (m['code'] + '.json')
            obj.parent.mkdir(parents=True, exist_ok=True)
            obj.write_text(json.dumps({'security_id': m['code'], 'bars': bars}))
            entry['object_ref'] = {'path': str(obj.relative_to(root / 'data')), 'sha256': CTX.sha(obj)}
        else:
            entry['bars'] = bars
        entries.append(entry)
    files = {
        'run.json': {'status': 'partial', 'run_id': 'batch', 'data_run_id': 'data',
                     'report_run_id': 'report', 'information_cutoff': '2026-03-11T22:00:00Z'},
        'manifest.json': {'run_id': 'batch', 'data_run_id': 'data'},
        'inputs/data_manifest.json': {'data_run_id': 'data', 'datasets': entries,
                                     'market_cutoffs': {'US': DAYS[-1]}, 'auxiliary_sources': {'news': 'not_connected'}},
        'inputs/members.json': {'observed_at': '2026-03-11T21:00:00Z', 'records': members,
                               'capture': {'ok': True, 'holdings': [{'code': 'US.HELD'}], 'errors': errors or []}},
        'inputs/calendars.json': {'US': {'days': DAYS}},
        'inputs/value_chain.json': {'groups': [{'id': 'compute', 'name': '计算'}], 'rows': []},
        'results/trend/results.json': {'version': 'five_traits_v2', 'data_run_id': 'data', 'keys': ['G'],
                                      'records': [{'code': m['code'], 'v': [0.1], 'status': ['scored']} for m in members]},
        'results/behavior.json': {'data_run_id': 'data', 'records': []},
    }
    checksums = {}
    for name, value in files.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        checksums[name] = CTX.sha(path)
    (run / 'checksums.json').write_text(json.dumps(checksums))
    return run


class ContextTests(unittest.TestCase):
    def test_completed_windows_ignore_future_and_require_every_session(self):
        bars = [[d, 1, 1, 1, 100 + i] for i, d in enumerate(DAYS)]
        result = CTX.window_returns(bars + [['2099-01-01', 1, 1, 1, 1e9]], DAYS, DAYS[-1])
        self.assertAlmostEqual(result['5']['value'], 169 / 164 - 1)
        self.assertEqual(result['60']['status'], 'scored')
        broken = CTX.window_returns([b for b in bars if b[0] != DAYS[-3]], DAYS, DAYS[-1])
        self.assertIsNone(broken['5']['value'])
        stale = CTX.window_returns(bars[:-1], DAYS, DAYS[-1])
        self.assertIsNone(stale['20']['value'])
        unknown = CTX.window_returns(bars, DAYS[:-1], DAYS[-1])
        self.assertEqual(unknown['5']['status'], 'unknown_calendar_or_target')
        self.assertIsNone(CTX.group_summary([])['windows']['5']['positive_share'])

    def test_pools_and_baseline_do_not_manufacture_trade_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture(Path(tmp))
            before = {p: CTX.sha(p) for p in run.rglob('*') if p.is_file()}
            result = CTX.build(run, REPO, run)
            self.assertEqual(result['counts']['holding'], 1)
            self.assertEqual(result['counts']['watch_only'], 2)
            self.assertEqual(result['counts']['background'], 1)
            self.assertTrue(result['baseline']['same_evaluation'])
            rows = {r['code']: r for r in result['securities']}
            self.assertEqual(rows['US.HELD']['role'], 'holding')
            self.assertEqual(rows['US.AIONLY']['role'], 'background')
            self.assertEqual(rows['US.HELD']['trait_raw_value_delta'], {'G': 0.0})
            self.assertTrue(all(r['news'] == 'not_requested' and r['action_assessment'] == 'not_computed'
                                for r in rows.values()))
            self.assertEqual(before, {p: CTX.sha(p) for p in run.rglob('*') if p.is_file()})

    def test_failed_holdings_query_prevents_confirmed_unheld_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture(Path(tmp), errors=[{'group': 'holdings', 'reason': 'query_failed'}])
            result = CTX.build(run, REPO)
            self.assertEqual(result['counts']['watch_only'], 0)
            self.assertEqual(result['counts']['watch_membership_unconfirmed'], 2)
            self.assertFalse(result['membership']['holdings_snapshot_query_succeeded'])

    def test_checksums_reject_modified_member_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture(Path(tmp))
            path = run / 'inputs/members.json'
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                CTX.build(run, REPO)

    def test_shared_objects_verify_content_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture(Path(tmp), shared=True)
            self.assertEqual(CTX.build(run, REPO)['counts']['holding'], 1)
            obj = Path(tmp) / 'data/daily/US.HELD.json'
            obj.write_text(obj.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                CTX.build(run, REPO)

    def test_shared_fact_symlink_requires_verified_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = fixture(root, shared=True)
            fact = run / 'inputs/members.json'
            shared = root / 'data/facts/members.json'
            shared.parent.mkdir(parents=True)
            shared.write_bytes(fact.read_bytes())
            fact.unlink()
            fact.symlink_to(shared)
            dm = run / 'inputs/data_manifest.json'
            manifest = CTX.read(dm)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                CTX.build(run, REPO)
            manifest['input_refs'] = {'inputs/members.json': {'path': 'facts/members.json', 'sha256': CTX.sha(shared)}}
            dm.write_text(json.dumps(manifest))
            checksums = CTX.read(run / 'checksums.json')
            checksums['inputs/data_manifest.json'] = CTX.sha(dm)
            (run / 'checksums.json').write_text(json.dumps(checksums))
            self.assertEqual(CTX.build(run, REPO)['counts']['holding'], 1)
            shared.write_text(shared.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                CTX.build(run, REPO)

    def test_relocated_export_uses_only_its_packaged_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = fixture(root / 'original', shared=True)
            export = root / 'offline-package'
            shutil.copytree(run, export)
            shutil.copytree(root / 'original/data', export / 'shared_data')
            (export / 'data_location.json').write_text(json.dumps({'root': 'shared_data'}))
            shutil.rmtree(root / 'original')
            result = CTX.build(export, REPO)
            self.assertEqual(result['counts']['holding'], 1)
            self.assertEqual(result['base_run_path'], str(export.resolve()))

    def test_cli_cannot_write_into_source_or_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = fixture(root)
            command = [sys.executable, str(SCRIPT), '--repo', str(REPO), '--run', str(run), '--output']
            forbidden = run / 'context.json'
            result = subprocess.run(command + [str(forbidden)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(forbidden.exists())
            target = root / 'insight/context.json'
            first = subprocess.run(command + [str(target)], capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            digest = CTX.sha(target)
            second = subprocess.run(command + [str(target)], capture_output=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(CTX.sha(target), digest)


if __name__ == '__main__':
    unittest.main()
