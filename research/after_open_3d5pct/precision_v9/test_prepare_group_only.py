"""Targeted real-source checks before the one-shot R03M full reconstruction."""
from dataclasses import replace
import json
import unittest

import numpy as np
import pandas as pd

from . import prepare_group_only as p


class RealClassificationPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _, cls.manifest = p._frozen_identity()
        cls.hashes = p._input_sources(p.V8 / 'inputs', cls.manifest)
        cls.sessions, cls.dates, cls.weeks = p._sessions(cls.config)
        cls.versions = [v for v in p._versions(cls.sessions, cls.weeks)
                        if v['week_id'] == '2026-W23']
        cls.definitions = p._definitions()
        cls.old = {(m['symbol'], m['week_id'], m['strategy_id']): m for m in
                   json.loads((p.V8 / 'inputs' / cls.config['group_run'] /
                               'memberships.json').read_text())}
        cls.original = p.rebuild_original_symbol('AAPL', p.V8 / 'inputs',
                         cls.sessions, cls.dates, cls.versions, cls.definitions, cls.hashes)
        old_root = p.vd.ROOT
        try:
            p.vd.ROOT = p.V8 / 'inputs'
            cls.aapl_view = p.vd.load_symbol('AAPL', cls.sessions, cls.config)
            cls.cohr_view = p.vd.load_symbol('COHR', cls.sessions, cls.config)
            cls.added = p.rebuild_added_symbol('COHR', cls.cohr_view,
                         p.vd._split_days('COHR'), cls.sessions, cls.dates,
                         cls.versions, cls.hashes)
        finally:
            p.vd.ROOT = old_root

    def test_float64_original_and_float32_added_match_frozen_categories(self):
        for symbol, rebuilt, path in (('AAPL', self.original, 'original_float64'),
                                      ('COHR', self.added, 'added_float32')):
            self.assertEqual(len(rebuilt), 5)
            for member in rebuilt:
                old = self.old.get((symbol, '2026-W23', member['strategy_id']))
                if old is not None:
                    self.assertEqual(member['group_ids'], old['group_ids'])
                    self.assertEqual(member['status'], old['status'])
                facts = member['facts']
                self.assertEqual(facts['classifier_path'], path)
                self.assertEqual(facts['bars_sha256'], self.hashes[f'market_data/us_5m/{symbol}/2026.parquet'])
                if member['group_ids']:
                    self.assertEqual(facts['history_end'], '2026-05-29')
                    self.assertLess(facts['max_available_at'],
                                    next(v['feature_cutoff_at'] for v in self.versions
                                         if v['strategy_id'] == member['strategy_id']))
                    if path == 'added_float32':
                        self.assertEqual(facts['classifier_value'], old['facts']['v8_classifier_value'])
                else:
                    self.assertEqual(facts['reason'], 'insufficient_history')

    def test_float32_window_rejects_unavailable_bar_and_action(self):
        version = next(v for v in self.versions if v['strategy_id'] == 'liquidity')
        ids, dates, need = p._window(self.sessions, self.dates, version['first_session'], 'liquidity')
        self.assertEqual(len(ids), need)
        late = self.cohr_view.effective_available.copy()
        late[ids[-1], 143] = np.datetime64(pd.Timestamp(version['feature_cutoff_at']).tz_localize(None))
        modified = replace(self.cohr_view, effective_available=late)
        result = p.rebuild_added_symbol('COHR', modified, set(), self.sessions,
                                        self.dates, [version], self.hashes)[0]
        self.assertEqual(result['group_ids'], [])
        self.assertEqual(result['facts']['reason'], 'history_not_available')
        blocked = p.rebuild_added_symbol('COHR', self.cohr_view, {dates[-1]},
                                         self.sessions, self.dates, [version], self.hashes)[0]
        self.assertEqual(blocked['group_ids'], [])
        self.assertEqual(blocked['facts']['reason'], 'corporate_action_in_lookback')

    def test_causal_group_state_rejects_future_history_end(self):
        members = [m for m in self.original + self.added if m['strategy_id'] == 'liquidity']
        own, peers = p._candidate_map([v for v in self.versions if v['strategy_id'] == 'liquidity'],
                                      members, ['AAPL', 'COHR'])
        day = self.dates.index('2026-06-01')
        valid = p.v9._group_state('AAPL', day, self.dates, self.sessions,
                                  {'AAPL': self.aapl_view, 'COHR': self.cohr_view}, own, peers)
        self.assertEqual(valid[4, 9], 1)
        key = ('2026-W23', 'liquidity', 'COHR')
        gid, facts, version = own[key]
        own[key] = (gid, {**facts, 'history_end': '2026-06-01'}, version)
        excluded = p.v9._group_state('AAPL', day, self.dates, self.sessions,
                                     {'AAPL': self.aapl_view, 'COHR': self.cohr_view}, own, peers)
        self.assertEqual(excluded[4, 9], 0)

    def test_source_manifest_refuses_changed_hash(self):
        corrupted = dict(self.manifest)
        corrupted['source_hashes'] = dict(self.manifest['source_hashes'])
        key = 'market_data/us_5m/AAPL/2026.parquet'
        corrupted['source_hashes'][key] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'frozen manifest source changed'):
            p._input_sources(p.V8 / 'inputs', corrupted)

    def test_iso_week_one_uses_december_official_cutoff(self):
        week_one = [v for v in p._versions(self.sessions, self.weeks)
                    if v['week_id'] == '2026-W01']
        self.assertEqual({v['first_session'] for v in week_one}, {'2025-12-29'})
        rebuilt = p.rebuild_original_symbol('AAPL', p.V8 / 'inputs', self.sessions,
                                             self.dates, week_one, self.definitions, self.hashes)
        for member in rebuilt:
            old = self.old[('AAPL', '2026-W01', member['strategy_id'])]
            self.assertEqual((member['group_ids'], member['status']),
                             (old['group_ids'], old['status']))

    def test_old_benchmark_metadata_is_counted_but_candidate_cannot_disappear(self):
        version = next(v for v in self.versions if v['strategy_id'] == 'liquidity')
        aapl = self.old[('AAPL', '2026-W23', 'liquidity')]
        qqq = {**aapl, 'symbol': 'QQQ'}
        lookup, extra = p._old_candidate_lookup([aapl, qqq], ['AAPL', 'COHR'], [version])
        self.assertEqual(set(lookup), {('AAPL', '2026-W23', 'liquidity')})
        self.assertEqual([m['symbol'] for m in extra], ['QQQ'])
        with self.assertRaisesRegex(ValueError, 'original candidate membership missing'):
            p._old_candidate_lookup([qqq], ['AAPL', 'COHR'], [version])
        with self.assertRaisesRegex(ValueError, 'duplicate old candidate membership'):
            p._old_candidate_lookup([aapl, aapl, qqq], ['AAPL', 'COHR'], [version])


if __name__ == '__main__':
    unittest.main()
