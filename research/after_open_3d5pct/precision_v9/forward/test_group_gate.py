"""Small causal group fixtures; no frozen training tensor enters feature construction."""
from __future__ import annotations

from copy import deepcopy
import warnings
import unittest

import numpy as np
import pandas as pd

from .features import CAUSAL_GROUP_PEERS, LEGACY_GROUP_PEERS, build_features_asof
from .test_features import schema


def _fixture():
    date = '2026-06-01'
    opening = pd.Timestamp('2026-06-01 09:30', tz='America/New_York')
    session = {'session_date': date, 'open_at': opening.isoformat(),
               'close_at': pd.Timestamp('2026-06-01 16:00', tz='America/New_York').isoformat(),
               'duration_minutes': 390}
    bars = {}
    for symbol, base in [('AMD', 100.), ('COHR', 120.), ('QQQ', 500.)]:
        rows = []
        for j in range(24):
            start = opening + pd.Timedelta(minutes=5*j)
            end = start + pd.Timedelta(minutes=5)
            price = base + j*.1
            rows.append({'session_date': date, 'start_at_et': start.isoformat(),
                         'end_at': end.isoformat(),
                         'available_at': (end+pd.Timedelta(seconds=1)).isoformat(),
                         'session_type': 'regular', 'price_basis': 'NONE',
                         'open': price, 'high': price+.1, 'low': price-.1,
                         'close': price+.05, 'volume': 100., 'turnover': 10000.})
        bars[symbol] = pd.DataFrame(rows)
    snapshot = {'bars': bars, 'sessions': [session], 'mode': 'historical_fixture'}
    decision = {'symbol': 'AMD', 'session_date': date,
                'cutoff_at': '2026-06-01T11:30:00-04:00',
                'decision_at': '2026-06-01T11:30:00-04:00',
                'information_deadline_at': '2026-06-01T11:30:30-04:00'}
    version = {'week_id': '2026-W23', 'strategy_id': 'trend_15',
               'version_id': 'trend_15:2026-W23:known',
               'membership_basis': 'causal_weekly_reconstruction',
               'feature_cutoff_at': '2026-05-29T20:00:00+00:00',
               'effective_from': '2026-05-29T20:00:00+00:00',
               'effective_to': '2026-06-08T14:30:00+00:00'}
    def member(symbol):
        return {'symbol': symbol, 'week_id': '2026-W23',
                'strategy_id': 'trend_15', 'version_id': version['version_id'],
                'group_ids': ['trend_15:up'], 'facts': {'history_end': '2026-05-29'}}
    metadata = {'split_days': {'AMD': []}, 'versions': [version],
                'memberships': [member('AMD'), member('COHR'), member('QQQ')],
                # QQQ has a classified row, but is deliberately not a candidate.
                'universe_symbols': ['AMD', 'COHR'],
                'group_peer_semantics': CAUSAL_GROUP_PEERS}
    return snapshot, decision, metadata


def _build(snapshot, decision, metadata, route='B_group'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return build_features_asof(snapshot, decision,
            schema(route, group_peer_semantics=CAUSAL_GROUP_PEERS), metadata, [])


class GroupGateTests(unittest.TestCase):
    def test_valid_classified_peer_and_benchmark_separation(self):
        snapshot, decision, metadata = _fixture()
        built = _build(snapshot, decision, metadata)
        self.assertEqual(built['rejections'], [])
        self.assertEqual(built['lineage']['group_peer_semantics'], CAUSAL_GROUP_PEERS)
        self.assertEqual(len(built['lineage']['group_metadata_sha256']), 64)
        group = built['X']['group_seq'][-1]
        self.assertEqual(group[0, 9], 1.)
        self.assertAlmostEqual(group[0, 7], np.log1p(1)/5, places=7)
        self.assertEqual(group[5, 9], 1.)  # QQQ remains a benchmark view.
        # Even a malformed candidate list must not promote the benchmark to peer.
        with_qqq = deepcopy(metadata)
        with_qqq['universe_symbols'].append('QQQ')
        revised = _build(snapshot, decision, with_qqq)
        np.testing.assert_array_equal(revised['X']['group_seq'], built['X']['group_seq'])

    def test_classified_own_and_peer_bad_history_rejected_with_lineage(self):
        snapshot, decision, metadata = _fixture()
        bad = [(None, 'missing_history_end'), ('2026-02-30', 'malformed_history_end'),
               (float('nan'), 'malformed_history_end'), (12345, 'malformed_history_end'),
               ('2026-06-01', 'nonprior_history_end'),
               ('2026-06-02', 'nonprior_history_end')]
        for symbol in ('AMD', 'COHR'):
            for value, reason in bad:
                with self.subTest(symbol=symbol, value=value):
                    changed = deepcopy(metadata)
                    facts = next(x['facts'] for x in changed['memberships'] if x['symbol']==symbol)
                    if value is None:
                        facts.pop('history_end')
                    else:
                        facts['history_end'] = value
                    result = _build(snapshot, decision, changed)
                    self.assertIsNone(result['X'])
                    self.assertEqual(result['rejections'], ['group_metadata_incompatible'])
                    detail = result['lineage']['group_metadata_rejection']
                    self.assertEqual(detail['issue_count'], 1)
                    self.assertEqual(detail['issues'][0]['symbol'], symbol)
                    self.assertEqual(detail['issues'][0]['strategy_id'], 'trend_15')
                    self.assertEqual(detail['issues'][0]['week_id'], '2026-W23')
                    self.assertEqual(detail['issues'][0]['representative_date'], '2026-06-01')
                    self.assertEqual(detail['issues'][0]['version_id'],
                                     'trend_15:2026-W23:known')
                    self.assertEqual(detail['issues'][0]['reason'], reason)
                    self.assertEqual(len(result['lineage']['group_metadata_sha256']), 64)

    def test_unclassified_insufficient_history_masks_group(self):
        snapshot, decision, metadata = _fixture()
        own = metadata['memberships'][0]
        own['group_ids'] = []
        own['classification_status'] = 'insufficient_history'
        result = _build(snapshot, decision, metadata)
        self.assertEqual(result['rejections'], [])
        self.assertEqual(float(np.max(result['X']['group_seq'][-1, 0])), 0.)
        self.assertEqual(result['X']['group_seq'][-1, 5, 9], 1.)

    def test_classified_group_id_must_match_strategy(self):
        snapshot, decision, metadata = _fixture()
        for member in metadata['memberships']:
            member['group_ids'] = ['liquidity:high']
        result = _build(snapshot, decision, metadata)
        self.assertEqual(result['rejections'], ['group_metadata_incompatible'])
        self.assertEqual(result['lineage']['group_metadata_rejection']['issues'][0]['reason'],
                         'inconsistent_group_or_version')

    def test_classified_category_must_be_registered(self):
        snapshot, decision, metadata = _fixture()
        for member in metadata['memberships']:
            member['group_ids'] = ['trend_15:unregistered_category']
        result = _build(snapshot, decision, metadata)
        self.assertIsNone(result['X'])
        self.assertEqual(result['rejections'], ['group_metadata_incompatible'])
        issue = result['lineage']['group_metadata_rejection']['issues'][0]
        self.assertEqual((issue['symbol'], issue['strategy_id'], issue['week_id']),
                         ('AMD', 'trend_15', '2026-W23'))
        self.assertEqual((issue['field'], issue['reason']),
                         ('group_ids', 'inconsistent_group_or_version'))
        self.assertEqual(len(result['lineage']['group_metadata_sha256']), 64)

    def test_classified_own_or_peer_unknown_version_rejected_before_mask(self):
        for symbol, role in (('AMD', 'own'), ('COHR', 'peer')):
            with self.subTest(symbol=symbol):
                snapshot, decision, metadata = _fixture()
                next(m for m in metadata['memberships'] if m['symbol'] == symbol)[
                    'version_id'] = 'missing_version'
                result = _build(snapshot, decision, metadata)
                self.assertIsNone(result['X'])
                self.assertEqual(result['rejections'], ['group_metadata_incompatible'])
                issue = result['lineage']['group_metadata_rejection']['issues'][0]
                self.assertEqual((issue['symbol'], issue['role'], issue['strategy_id'],
                                  issue['week_id'], issue['version_id']),
                                 (symbol, role, 'trend_15', '2026-W23', 'missing_version'))
                self.assertEqual((issue['field'], issue['reason']),
                                 ('version_id', 'missing_or_mismatched_version'))
                self.assertEqual(len(result['lineage']['group_metadata_sha256']), 64)

    def test_future_own_version_and_unrelated_classified_row_keep_mask_semantics(self):
        snapshot, decision, metadata = _fixture()
        unrelated = deepcopy(next(m for m in metadata['memberships']
                                  if m['symbol'] == 'COHR'))
        unrelated['group_ids'] = ['trend_15:range']
        unrelated['version_id'] = 'unrelated_missing_version'
        metadata['memberships'].append(unrelated)
        self.assertEqual(_build(snapshot, decision, metadata)['rejections'], [])
        future = deepcopy(metadata['versions'][0])
        future['version_id'] = 'registered_future'
        future['feature_cutoff_at'] = '2026-06-02T20:00:00+00:00'
        future['effective_from'] = '2026-06-02T20:00:00+00:00'
        metadata['versions'].append(future)
        metadata['memberships'][0]['version_id'] = future['version_id']
        result = _build(snapshot, decision, metadata)
        self.assertEqual(result['rejections'], [])
        self.assertEqual(float(np.max(result['X']['group_seq'][-1, 0])), 0.)

    def test_legacy_or_undeclared_semantics_and_no_group(self):
        snapshot, decision, metadata = _fixture()
        old_schema = schema('B_group', group_peer_semantics=LEGACY_GROUP_PEERS)
        old = build_features_asof(snapshot, decision, old_schema, metadata, [])
        self.assertEqual(old['rejections'], ['group_metadata_incompatible'])
        self.assertEqual(old['lineage']['group_metadata_rejection']['reason'],
                         'legacy_static_schema_is_not_feature_only')
        untagged = deepcopy(metadata)
        untagged.pop('group_peer_semantics')
        result = _build(snapshot, decision, untagged)
        self.assertEqual(result['rejections'], ['group_metadata_incompatible'])
        self.assertEqual(result['lineage']['group_metadata_rejection']['reason'],
                         'group_source_semantics_missing_or_legacy')
        # No-group route neither reads nor requires group metadata.
        no_group = _build(snapshot, decision, {'split_days': {'AMD': []}}, 'B_no_group')
        self.assertEqual(no_group['rejections'], [])


if __name__ == '__main__':
    unittest.main()
