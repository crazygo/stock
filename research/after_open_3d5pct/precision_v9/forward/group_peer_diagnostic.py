"""Read-only frozen AMD group-map diagnosis; print one canonical JSON record."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from ... import v6_data as legacy
from .test_features import schema
from .test_group_parity import V8, _fixture


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _saved_features():
    source = Path(__file__).with_name('evidence_baseline') / 'features.py'
    name = __package__ + '._saved_prepatch_features'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    features = _saved_features()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        snapshot, decision, metadata, store, _, frozen, _, fixture = _fixture()
        captured = {}
        original = features.d._group_state

        def capture(symbol, day, dates, sessions, views, own, peers):
            captured[day] = (symbol, dates, sessions, views, own, peers)
            return original(symbol, day, dates, sessions, views, own, peers)

        with patch.object(features.d, '_group_state', capture):
            features.build_features_asof(snapshot, decision, schema('B_group'), metadata, store)

    config = json.loads((V8 / 'config.json').read_text())
    # This is the actual frozen builder call: candidate symbols to _group_map,
    # with QQQ loaded as a separate benchmark view. The 109-symbol comparison
    # deliberately passes benchmark QQQ into that map and must differ.
    candidate = sorted(metadata['universe_symbols'])
    legacy.ROOT = V8 / 'inputs'
    args = next(iter(captured.values()))
    old_map, old_peers = legacy._group_map(config, args[2], candidate)
    benchmark_map, benchmark_peers = legacy._group_map(config, args[2], candidate + ['QQQ'])
    result = {
        'case': 'AMD:2026-06-01:11:30:v8_frozen',
        'candidate_count': len(candidate),
        'candidate_contains_qqq': 'QQQ' in candidate,
        'candidate_plus_qqq_count': len(set(candidate) | {'QQQ'}),
        'actual_fixture_dependency_count': fixture['dependency_symbols'],
        'actual_fixture_dependencies_contain_qqq': 'QQQ' in snapshot['bars'],
        'source_sha256': {
            'v6_data.py': _hash(Path(legacy.__file__)),
            'precision_v9/data.py': _hash(Path(features.d.__file__)),
            'forward/features.py': _hash(Path(features.__file__)),
            'forward/test_group_parity.py': _hash(Path(__file__).with_name('evidence_baseline') / 'test_group_parity.py'),
            'versions.json': _hash(V8 / 'inputs' / config['group_run'] / 'versions.json'),
            'memberships.json': _hash(V8 / 'inputs' / config['group_run'] / 'memberships.json'),
        },
        'call': {
            'frozen_builder': 'v6_data._group_map(config, dates, candidate_symbols)',
            'wrong_comparison': 'v6_data._group_map(config, dates, candidate_symbols + [QQQ])',
            'function_argument_order': 'symbol, day_index, dates, sessions, views, own_map, peers',
            'same_views_each_comparison': True,
        },
        'weeks': [],
    }
    for j, (ix, call) in enumerate(captured.items()):
        symbol, dates, sessions, views, forward_map, forward_peers = call
        date = dates[ix]
        iso = pd.Timestamp(date).isocalendar()
        week = f'{iso.year}-W{iso.week:02d}'
        old = legacy._group_state(symbol, ix, dates, sessions, views, old_map, old_peers)
        with_qqq = legacy._group_state(symbol, ix, dates, sessions, views, benchmark_map, benchmark_peers)
        forward_legacy = legacy._group_state(symbol, ix, dates, sessions, views, forward_map, forward_peers)
        forward_v9 = original(symbol, ix, dates, sessions, views, forward_map, forward_peers)
        frozen_week = frozen['group_seq'][0, j]
        record = {
            'week': week, 'representative_date': date,
            'frozen_candidate_only_max_abs': float(np.max(np.abs(old - frozen_week))),
            'candidate_plus_qqq_vs_forward_legacy_max_abs': float(np.max(np.abs(with_qqq - forward_legacy))),
            'candidate_plus_qqq_vs_frozen_max_abs': float(np.max(np.abs(with_qqq - frozen_week))),
            'forward_v9_vs_frozen_max_abs': float(np.max(np.abs(forward_v9 - frozen_week))),
            'strategies': [],
        }
        for slot, strategy in enumerate(legacy.STRATEGIES):
            own = old_map.get((week, strategy, symbol))
            if own is None:
                continue
            group = own[0]
            older, newer = old_peers[(week, group)], forward_peers[(week, group)]
            record['strategies'].append({
                'strategy': strategy,
                'old_peer_count': len(older), 'forward_peer_count': len(newer),
                'old_only': sorted(older - newer), 'forward_only': sorted(newer - older),
                'old_issuers': len({legacy._issuer(p) for p in older if legacy._issuer(p) != legacy._issuer(symbol)}),
                'forward_issuers': len({legacy._issuer(p) for p in newer if legacy._issuer(p) != legacy._issuer(symbol)}),
                'qqq_prefix_return': float(views['QQQ'].cutoff_return[ix]),
                'qqq_history_end': (forward_map.get((week, strategy, 'QQQ')) or (None, {}))[1].get('history_end'),
                'frozen_candidate_only_max_abs': float(np.max(np.abs(old[slot] - frozen_week[slot]))),
                'legacy_benchmark_inclusion_max_abs': float(np.max(np.abs(with_qqq[slot] - frozen_week[slot]))),
                'v9_forward_max_abs': float(np.max(np.abs(forward_v9[slot] - frozen_week[slot]))),
            })
        result['weeks'].append(record)
    assert not result['candidate_contains_qqq'] and result['candidate_count'] == 108
    assert result['candidate_plus_qqq_count'] == 109
    assert all(x['frozen_candidate_only_max_abs'] == 0 for x in result['weeks'])
    assert all(x['candidate_plus_qqq_vs_forward_legacy_max_abs'] == 0 for x in result['weeks'])
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
