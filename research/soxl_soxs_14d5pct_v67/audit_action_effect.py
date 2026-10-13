"""Post-result descriptive counts; never selects or rewrites a strategy."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def audit(run):
    run = Path(run)
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = read(run / 'results.json')
    verified = read(run / 'verification.json')
    assert verified['passed'] and verified['results_sha256'] == sha(run / 'results.json')
    frozen = read(run / 'decisions_frozen.json')
    assert frozen['predictions_sha256'] == sha(run / 'predictions.json')
    fresh = {x['at']: x for x in frozen['routes']['fresh_point']}
    old = {x['at']: x for x in frozen['routes']['conditional_point']}
    assert len(fresh) == len(frozen['routes']['fresh_point'])
    assert len(old) == len(frozen['routes']['conditional_point']) and fresh.keys() == old.keys()
    changed = [at for at in fresh if fresh[at]['action'] != old[at]['action']]
    transitions = collections.Counter(old[at]['action'] + ' -> ' + fresh[at]['action'] for at in changed)
    assert sum(transitions.values()) == len(changed)
    by_month = collections.Counter(fresh[at]['date'][:7] for at in changed)
    assert sum(by_month.values()) == len(changed)
    entries = {}
    for name in ['fresh_point', 'conditional_point']:
        with (run / (name + '_trades.csv')).open() as stream:
            rows = list(csv.DictReader(stream))
        entries[name] = {(x['decision_at'], x['symbol']) for x in rows}
        assert len(entries[name]) == len(rows) == result['routes'][name]['entries']
        events = fresh if name == 'fresh_point' else old
        assert all(events[at]['action'] == symbol for at, symbol in entries[name])
    out = {
        'passed': True,
        'post_result_diagnostic': True,
        'registered_before_capital_results': False,
        'strategy_changed': False,
        'queries': len(fresh),
        'changed_proposals': len(changed),
        'changed_dates': len({fresh[at]['date'] for at in changed}),
        'changed_by_old_to_new': dict(transitions),
        'changed_by_month': dict(by_month),
        'fresh_entries': len(entries['fresh_point']),
        'legacy_entries': len(entries['conditional_point']),
        'common_entry_events': len(entries['fresh_point'] & entries['conditional_point']),
        'scope': 'Proposal and actual-entry counts only; neither a causal gain estimate nor independent samples.',
        'results_sha256': sha(run / 'results.json'),
        'verification_sha256': sha(run / 'verification.json'),
        'decisions_frozen_sha256': sha(run / 'decisions_frozen.json'),
        'source_sha256': sha(Path(__file__)),
    }
    (run / 'action_effect_audit.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
