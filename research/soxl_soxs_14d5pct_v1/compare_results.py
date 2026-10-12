"""Compare recomputed research artifacts to the immutable imported batch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import pandas as pd

from paths import IMPORT_ROOT, writable_results


def compare_json(actual, expected, path='$'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f'{path}: keys differ')
        for key in expected:
            compare_json(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f'{path}: list length differs')
        for i, value in enumerate(expected):
            compare_json(actual[i], value, f'{path}[{i}]')
    elif isinstance(expected, Real) and not isinstance(expected, bool):
        if not isinstance(actual, Real) or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
            raise AssertionError(f'{path}: {actual!r} != {expected!r}')
    elif actual != expected:
        raise AssertionError(f'{path}: {actual!r} != {expected!r}')


def compare(run):
    run = writable_results(run)
    manifest = json.loads((IMPORT_ROOT / 'MANIFEST.json').read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((IMPORT_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Imported snapshot changed: {name}')
    checked = []
    # verification.json has deliberately stronger coverage/exit-status checks.
    for baseline in sorted((IMPORT_ROOT / 'results').iterdir()):
        if baseline.name == 'verification.json' or baseline.suffix not in {'.csv', '.json'}:
            continue
        actual = run / baseline.name
        if baseline.suffix == '.json':
            compare_json(json.loads(actual.read_text()), json.loads(baseline.read_text()))
        elif baseline.read_bytes() == b'\n':
            if actual.read_bytes() != baseline.read_bytes():
                raise AssertionError(f'{baseline.name}: expected empty trades')
        else:
            old, new = pd.read_csv(baseline), pd.read_csv(actual)
            extra = set(new.columns) - set(old.columns)
            if extra != ({'all_expected_bars_present'} if baseline.name == 'hourly_trade_audit.csv' else set()):
                raise AssertionError(f'{baseline.name}: unexpected columns: {extra}')
            pd.testing.assert_frame_equal(new[old.columns], old, check_exact=False,
                                          rtol=1e-11, atol=1e-12)
        checked.append(baseline.name)
    verification = json.loads((run / 'verification.json').read_text())
    if (verification['unit_tests']['failures'] or verification['unit_tests']['errors']
            or verification['disagreements']
            or not verification['all_checked_paths_have_full_hourly_sessions']):
        raise AssertionError('Implementation verification failed')
    report = {
        'pass': True, 'manifest_checked_files': len(manifest),
        'compared_artifacts': checked, 'numeric_relative_tolerance': 1e-11,
        'numeric_absolute_tolerance': 1e-12,
        'excluded_binary_artifacts': ['frozen_models.pkl', 'equity.png'],
        'binary_exclusion_reason': 'No unpickling; model outputs compared numerically; rendering is platform-dependent',
        'verification': verification,
        'effectiveness_gate': 'FAILED; successful reproduction does not change model acceptance',
    }
    (run / 'reproduction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    compare(parser.parse_args().results)
