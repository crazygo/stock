"""Publish verified derived audit files, preserving immutable source evidence."""
from pathlib import Path
from datetime import datetime, timezone
import shutil
from error_audit import ROOT, SOURCE, CACHE, read, dump, sha


def package():
    run = CACHE / 'audit'
    v = read(run / 'verification.json')
    assert v['passed'] and v['result_sha256'] == sha(run / 'results.json')
    assert v['inputs_sha256'] == sha(run / 'inputs.json')
    registration = read(SOURCE / 'evidence/analysis_registration.json')
    for f, h in registration['code_sha256'].items():
        assert sha(ROOT / f) == h, f
    for f, h in read(run / 'inputs.json').items():
        assert sha(ROOT / f) == h, f
    result = read(run / 'results.json')
    assert not result['goal_complete'] and not result['strategy_changed'] and not result['new_cases_run']
    assert not any(p.is_dir() and 'case' in p.name.lower() for p in CACHE.iterdir())
    for f in ['candidate_rows.csv', 'model_rows.csv', 'executed_rows.csv', 'results.json', 'inputs.json', 'verification.json']:
        shutil.copyfile(run / f, SOURCE / 'evidence' / f)
    assert sha(CACHE / 'tests.log') == registration['tests_sha256']
    shutil.copyfile(CACHE / 'tests.log', SOURCE / 'evidence/tests.log')
    dump(SOURCE / 'evidence/final_verification.json', dict(
        passed=True, created_at=datetime.now(timezone.utc).isoformat(),
        result_sha256=sha(run / 'results.json'), verification_sha256=sha(run / 'verification.json'),
        tests_sha256=sha(CACHE / 'tests.log'), registered_code_files=len(registration['code_sha256']),
        verified_input_files=len(read(run / 'inputs.json')), new_strategy=False,
        new_cases_run=False, goal_complete=False, no_raw_market_or_models_copied=True))
    files = {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'delivery_manifest.json'}
    dump(SOURCE / 'evidence/delivery_manifest.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(), delivery_files_sha256=files,
        cache_root=str(run.relative_to(ROOT)),
        cache_files_sha256={str(p.relative_to(run)): sha(p) for p in run.rglob('*') if p.is_file()},
        input_files_sha256=read(run / 'inputs.json'), input_root='.',
        additional_cache_files_sha256={'tests.log': sha(CACHE/'tests.log'), 'verification.log': sha(CACHE/'verification.log')},
        new_strategy=False, new_buy_eligible=False, goal_complete=False))
    print({'delivery_files': len(files), 'source_records_verified': len(read(run/'inputs.json'))})


if __name__ == '__main__':
    package()
