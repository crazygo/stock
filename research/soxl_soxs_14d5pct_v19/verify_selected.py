"""Verify selected case replay and immutable pre-May blend predictions."""
from pathlib import Path
import argparse
from selection import validate_selection, core, sha, read
from verify_blend import verify


def check(run):
    authorization = validate_selection(); run = Path(run); verify(run)
    old = core.ROOT / '.cache/soxl_soxs_14d5pct_v18/runs/blend_20261012'
    before = read(old / 'predictions.json'); now = read(run / 'predictions.json')
    # The parent-primary no-risk and stress comparisons intentionally change.
    unchanged = [n for n in before if n not in ['blend_no_risk', 'double_cost']]
    count = 0
    for name in unchanged:
        assert before[name] == now[name][:len(before[name])], name
        count += len(before[name])
    proof = {'passed': True, 'pre_may_events_unchanged': count, 'routes': unchanged,
             'intentional_primary_based_controls': ['blend_no_risk', 'double_cost'],
             'source_prediction_sha256': sha(old / 'predictions.json'),
             'case_prediction_sha256': sha(run / 'predictions.json'),
             'selection_authorization': authorization}
    core.dump(run / 'historical_prefix_verification.json', proof); print(proof)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', type=Path, required=True)
    check(p.parse_args().run)
