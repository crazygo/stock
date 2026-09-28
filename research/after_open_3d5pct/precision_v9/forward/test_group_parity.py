"""Read-only G1 pairing: real frozen peer bars -> feature-only grouped routes."""
from __future__ import annotations

import json
from pathlib import Path
import time
import unittest
import warnings

import numpy as np
import pandas as pd

from .features import CAUSAL_GROUP_PEERS, LABEL_COLUMNS, build_features_asof
from .inference import predict_route
from .test_features import V8, real_fixture, row_from_npz, schema


ROUTES = ('B_group', 'C_group', 'C_no_daily')
DATE = '2026-06-01'
SYMBOL = 'AMD'
BAR_COLUMNS = ['session_date', 'start_at_et', 'end_at', 'available_at',
               'session_type', 'price_basis', 'open', 'high', 'low', 'close',
               'volume', 'turnover']
STRATEGIES = {'trend_15', 'trend_63', 'trend_126', 'volatility', 'liquidity'}


def _dependencies(sessions, versions, memberships, universe, deadline):
    """Union of the actual six weekly peer groups, using frozen membership IDs."""
    representatives = {}
    for ix in range(len(sessions) - 1, -1, -1):
        iso = pd.Timestamp(sessions[ix]['session_date']).isocalendar()
        week = f'{iso.year}-W{iso.week:02d}'
        representatives.setdefault(week, ix)
        if len(representatives) == 6:
            break
    assert len(representatives) == 6
    eligible_versions = {v['version_id'] for v in versions
                         if v['strategy_id'] in STRATEGIES and
                         pd.Timestamp(v['feature_cutoff_at']) <= deadline and
                         pd.Timestamp(v['effective_from']) <= deadline}
    own_groups = {(m['week_id'], m['group_ids'][0]) for m in memberships
                  if m['symbol'] == SYMBOL and m['week_id'] in representatives and
                  m['strategy_id'] in STRATEGIES and len(m['group_ids']) == 1 and
                  m['version_id'] in eligible_versions}
    dependencies = {m['symbol'] for m in memberships
                    if m['strategy_id'] in STRATEGIES and len(m['group_ids']) == 1 and
                    (m['week_id'], m['group_ids'][0]) in own_groups and
                    m['version_id'] in eligible_versions and m['symbol'] in universe}
    dependencies |= {SYMBOL, 'QQQ'}
    first = min(representatives.values())
    peer_start = sessions[max(0, first - 20)]['session_date']
    return sorted(dependencies), peer_start, sorted(representatives)


def _split_days(path):
    actions = pd.read_parquet(path)
    ratio = pd.to_numeric(actions.get('split_ratio', pd.Series(np.nan, index=actions.index)), errors='coerce')
    base = pd.to_numeric(actions.get('split_base', pd.Series(np.nan, index=actions.index)), errors='coerce')
    return sorted(actions.loc[((ratio.notna()) & (ratio != 0) & (ratio != 1)) |
                              ((base.notna()) & (ratio.notna()) & (base != ratio)),
                              'ex_div_date'].astype(str))


def _fixture():
    started = time.perf_counter()
    snapshot, decision, row_ix, rows = real_fixture(SYMBOL, DATE)
    inputs = V8 / 'inputs'
    config = json.loads((V8 / 'config.json').read_text())
    group = inputs / config['group_run']
    versions = json.loads((group / 'versions.json').read_text())
    memberships = json.loads((group / 'memberships.json').read_text())
    universe = {m['symbol'] for m in json.loads((inputs / config['universe']).read_text())['members']
                if m['role'] == 'candidate'}
    deadline = pd.Timestamp(decision['information_deadline_at'])
    dependencies, peer_start, weeks = _dependencies(
        snapshot['sessions'], versions, memberships, universe, deadline)
    assert len(dependencies) == 107 and len(universe) == 108
    root = inputs / config['source_dir']
    snapshot['bars'][SYMBOL] = snapshot['bars'][SYMBOL].loc[
        snapshot['bars'][SYMBOL].session_date <= DATE, BAR_COLUMNS].copy()
    for symbol in dependencies:
        if symbol == SYMBOL:
            continue
        path = root / symbol / '2026.parquet'
        frame = pd.read_parquet(path, columns=BAR_COLUMNS,
                                filters=[('session_date', '>=', peer_start),
                                         ('session_date', '<=', DATE)])
        clock = frame.start_at_et.str[11:16]
        snapshot['bars'][symbol] = frame.loc[(frame.session_type == 'regular') &
                                             (clock >= '09:30') & (clock < '11:30')].copy()
    assert set(snapshot['bars']) == set(dependencies)
    assert all(not (LABEL_COLUMNS & set(frame)) for frame in snapshot['bars'].values())
    actions = root.parent / 'corporate_actions' / f'{SYMBOL}.parquet'
    metadata = {'split_days': {SYMBOL: _split_days(actions)},
                'versions': versions, 'memberships': memberships,
                'universe_symbols': sorted(universe),
                'mature_prior_scope': 'all_candidates'}
    # Read the small nine-label array once. Only rows with every outcome mature
    # strictly before this decision enter the feature builder's prior store.
    with np.load(V8 / 'dataset/features.npz') as archive:
        labels = archive['y'].reshape(-1, 9)
    own = np.flatnonzero((rows.symbol == SYMBOL).to_numpy())
    eligible = [i for i in own if i != row_ix and
                pd.Timestamp(rows.iloc[i].decision_at) < deadline and
                pd.Timestamp(rows.iloc[i].label_end_at) < deadline and
                pd.Timestamp(rows.iloc[i].label_available_at) < deadline]
    store = [{'sample_id': rows.iloc[i].sample_id, 'symbol': SYMBOL,
              'decision_at': rows.iloc[i].decision_at,
              'label_end_at': rows.iloc[i].label_end_at,
              'label_available_at': rows.iloc[i].label_available_at,
              'y': labels[i].tolist()} for i in eligible]
    assert store and all(len(item['y']) == 9 for item in store)
    assert not any('entry_price' in item for item in store)
    prior = ((labels[eligible[-63:]].sum(axis=0) + 1) /
             (min(len(eligible), 63) + 2)).astype(np.float32)
    frozen = {name: row_from_npz(V8 / 'dataset/features.npz', name, row_ix)[None]
              for name in ('x5', 'x60', 'xday', 'group_seq')}
    return snapshot, decision, metadata, store, prior, frozen, rows.iloc[row_ix], {
        'dependency_symbols': len(dependencies), 'candidate_universe_symbols': len(universe),
        'peer_prefix_start': peer_start, 'representative_weeks': weeks,
        'eligible_mature_prior_rows': len(store), 'fixture_seconds': time.perf_counter() - started}


def _max_abs(left, right):
    a, b = np.asarray(left), np.asarray(right)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    return float(np.max(np.abs(np.nan_to_num(a - b, nan=0.0)))) if a.size else 0.0


class GroupParityTests(unittest.TestCase):
    def test_frozen_legacy_metadata_rejected_before_group_prediction(self):
        snapshot, decision, metadata, store, _, _, _, metrics = _fixture()
        self.assertEqual(metrics['dependency_symbols'], 107)
        self.assertEqual(metrics['eligible_mature_prior_rows'], 58)
        # An adversarial declaration of the new semantics must not cause the
        # old membership rows to pass: their liquidity history_end is absent.
        claimed_new_metadata = {**metadata, 'group_peer_semantics': CAUSAL_GROUP_PEERS}
        for route in ROUTES:
            with self.subTest(route=route):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    built = build_features_asof(snapshot, decision,
                        schema(route, group_peer_semantics=CAUSAL_GROUP_PEERS),
                        claimed_new_metadata, store)
                self.assertIsNone(built['X'])
                self.assertEqual(built['rejections'], ['group_metadata_incompatible'])
                self.assertFalse(built['lineage']['g2_eligible'])
                self.assertEqual(built['lineage']['group_peer_semantics'], CAUSAL_GROUP_PEERS)
                self.assertEqual(len(built['lineage']['group_metadata_sha256']), 64)
                detail = built['lineage']['group_metadata_rejection']
                self.assertEqual(detail['reason'], 'classified_member_time_evidence_invalid')
                self.assertGreaterEqual(detail['issue_count'], 568)
                self.assertTrue(any(x['reason'] == 'missing_history_end' and
                                    x['strategy_id'] == 'liquidity' for x in detail['issues']))
                with self.assertRaisesRegex(ValueError, 'feature build rejected'):
                    predict_route({'manifest': {'route_id': route}}, built)


if __name__ == '__main__':
    unittest.main()
