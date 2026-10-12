#!/usr/bin/env python3
"""Read a frozen stock-review-pair run; prepare facts, never buy/sell signals."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

HORIZONS = (5, 20, 60)
SECTORS = {
    'US.XLB': '材料', 'US.XLC': '通信服务', 'US.XLE': '能源',
    'US.XLF': '金融', 'US.XLI': '工业', 'US.XLK': '信息技术',
    'US.XLP': '必需消费', 'US.XLRE': '房地产', 'US.XLU': '公用事业',
    'US.XLV': '医疗', 'US.XLY': '可选消费',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class CheckedRun:
    def __init__(self, path, storage):
        self.path = Path(path).resolve(strict=True)
        self.storage = storage
        self.checksums = read(self.path / 'checksums.json')
        self.meta = self.load('run.json')
        if self.meta.get('status') not in {'complete', 'partial'}:
            raise ValueError('run is not a completed batch: ' + str(self.path))
        self.data = self.load('inputs/data_manifest.json')
        self.report = self.load('results/trend/results.json')
        self.members = self.load('inputs/members.json')
        self.calendars = self.load('inputs/calendars.json')
        self.value_chain = self.load('inputs/value_chain.json')
        self.behavior = self.load('results/behavior.json')
        manifest = self.load('manifest.json')
        for item in [self.data, self.report, self.behavior, manifest]:
            if item.get('data_run_id') != self.meta['data_run_id']:
                raise ValueError('batch data identity mismatch')
        if manifest.get('run_id') != self.meta['run_id']:
            raise ValueError('batch run identity mismatch')

    def load(self, name):
        path = self.path / name
        expected = self.checksums.get(name)
        resolved = path.resolve()
        allowed = resolved.is_relative_to(self.path)
        if not allowed:
            ref = getattr(self, 'data', {}).get('input_refs', {}).get(name)
            allowed = bool(ref and self.storage.resolve_ref(self.path, ref).resolve() == resolved)
        if not expected or not allowed or sha(path) != expected:
            raise ValueError('input missing or checksum mismatch: ' + name)
        return read(path)

    def ref(self, name):
        return {'path': str(self.path / name), 'sha256': self.checksums[name]}


def window_returns(bars, days, cutoff):
    """Use n+1 official sessions ending at the market target; gaps stay unknown."""
    result = {}
    calendar = sorted(set(d for d in days if d <= cutoff))
    points = {}
    for bar in bars:
        day = bar[0]
        if day > cutoff:
            continue
        if day in points:
            raise ValueError('duplicate daily observation: ' + day)
        points[day] = bar[4]
    for n in HORIZONS:
        expected = calendar[-n - 1:]
        reason = None
        if not calendar or calendar[-1] != cutoff:
            reason = 'unknown_calendar_or_target'
        elif len(expected) != n + 1:
            reason = 'short_calendar'
        elif any(d not in points for d in expected):
            reason = 'missing_or_short_history'
        elif any(not isinstance(points[d], (float, int)) or not math.isfinite(points[d])
                 or points[d] <= 0 for d in expected):
            reason = 'invalid_close'
        result[str(n)] = {
            'value': None if reason else points[expected[-1]] / points[expected[0]] - 1,
            'status': reason or 'scored',
            'start': expected[0] if expected else None,
            'end': cutoff, 'required_closes': n + 1,
            'present_closes': sum(d in points for d in expected),
        }
    return result


def group_summary(items):
    result = {'securities': len(items), 'security_ids': [r['code'] for r in items], 'windows': {}}
    for n in HORIZONS:
        values = [r['price_returns'][str(n)]['value'] for r in items
                  if r['price_returns'][str(n)]['value'] is not None]
        result['windows'][str(n)] = {
            'scored': len(values), 'unknown': len(items) - len(values),
            'positive': sum(v > 0 for v in values),
            'positive_share': sum(v > 0 for v in values) / len(values) if values else None,
            'equal_weight_mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
        }
    return result


def unique(rows, key):
    result = {}
    for row in rows:
        identity = row[key]
        if identity in result:
            raise ValueError('duplicate identity: ' + identity)
        result[identity] = row
    return result


def build(run_path, repo, baseline_path=None):
    sys.path.insert(0, str(Path(repo).resolve() / 'analysis/stock_review_pair_v1'))
    storage = importlib.import_module('storage')
    run = CheckedRun(run_path, storage)
    baseline = CheckedRun(baseline_path, storage) if baseline_path else None
    records = run.members['records']
    records = list(records.values()) if isinstance(records, dict) else records
    members = unique(records, 'code')
    trends = unique(run.report['records'], 'code')
    behaviors = unique(run.behavior['records'], 'code')
    datasets = unique(run.data['datasets'], 'security_id')
    businesses = {r['code']: r.get('business') or {} for r in run.value_chain.get('rows', [])}
    old_trends = unique(baseline.report['records'], 'code') if baseline else {}
    compatible = bool(baseline and baseline.report['version'] == run.report['version'])
    capture = run.members.get('capture') or {}
    errors = capture.get('errors', [])
    sanitized_errors = [{k: e.get(k) for k in ('group', 'reason')} for e in errors]
    holdings_known = (capture.get('ok') is True and isinstance(capture.get('holdings'), list)
                      and not any(e.get('group') == 'holdings' for e in errors))
    output = []
    for code, member in members.items():
        entry = datasets.get(code)
        if entry is None:
            raise ValueError('member missing from data manifest: ' + code)
        bars = storage.price_bars(run.path, entry)
        normalized = hashlib.sha256(json.dumps([code, bars], sort_keys=True, ensure_ascii=False,
                                                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if normalized != entry['dataset_id']:
            raise ValueError('normalized price identity mismatch: ' + code)
        market = entry['market']
        target = run.data['market_cutoffs'][market]
        cal = run.calendars.get(market) or {}
        groups = member.get('groups', [])
        role = ('holding' if 'holdings' in groups else
                'watch_only' if any(g in groups for g in ('watchlist', 'special')) and holdings_known else
                'watch_membership_unconfirmed' if any(g in groups for g in ('watchlist', 'special')) else
                'background')
        trend = trends.get(code, {})
        metrics = {key: {'value': value, 'status': state}
                   for key, value, state in zip(run.report.get('keys', ['G', 'V', 'M', 'J', 'S']),
                                                trend.get('v', []), trend.get('status', []))}
        old = old_trends.get(code, {})
        deltas = {}
        if compatible:
            for key, before, after, os, ns in zip(run.report['keys'], old.get('v', []),
                                                trend.get('v', []), old.get('status', []), trend.get('status', [])):
                deltas[key] = after - before if ns == os == 'scored' and after is not None and before is not None else None
        behavior = behaviors.get(code)
        business = businesses.get(code, {})
        query = urlencode({'code': code})
        output.append({
            'code': code, 'name': member.get('name'), 'kind_captured': member.get('kind'),
            'role': role, 'groups': groups, 'chains': member.get('chains', []),
            'industry': business.get('industry'), 'business_label_status': business.get('review_status'),
            'quality': member.get('quality'), 'market': market, 'market_cutoff': target,
            'dataset_id': entry['dataset_id'], 'price_basis': entry['price_basis'],
            'source': entry.get('source'), 'received_at': entry.get('received_at'),
            'coverage': entry.get('coverage'),
            'price_returns': window_returns(bars, cal.get('days', []), target),
            'traits': metrics, 'trait_raw_value_delta': deltas if baseline else None,
            'behavior': ({k: behavior.get(k) for k in ('start', 'end', 'settings', 'shares', 'status')}
                         | {'folds': {k: (behavior.get('folds') or {}).get(k)
                                      for k in ('frequency', 'amplitudeMedian', 'amplitudeN')}}) if behavior else None,
            'evidence': {
                'dataset_manifest': run.ref('inputs/data_manifest.json'),
                'trend': {**run.ref('results/trend/results.json'), 'security_id': code},
                'behavior': {**run.ref('results/behavior.json'), 'security_id': code} if behavior else None,
                'trend_page': str(run.path / 'pages/ai_trend_quadrant_v2/index.html'),
                'trend_stock_navigation': {'method': 'page search/select', 'code': code},
                'behavior_detail': str(run.path / 'pages/stock_behavior_description_v3/detail.html') + '?' + query
                                   if behavior else None,
            },
            'current_membership_verified': False, 'current_quote': 'not_requested',
            'news': 'not_requested', 'action_assessment': 'not_computed',
        })
    by_code = {r['code']: r for r in output}
    benchmark = {}
    for n in HORIZONS:
        benchmark[str(n)] = next((c for c in ('US.SPY', 'US.VOO')
                                  if c in by_code and by_code[c]['price_returns'][str(n)]['value'] is not None), None)
    for row in output:
        row['relative_to_us_broad_price'] = {}
        for n in HORIZONS:
            code = benchmark[str(n)] if row['market'] == 'US' else None
            value = row['price_returns'][str(n)]['value']
            base = by_code[code]['price_returns'][str(n)]['value'] if code else None
            row['relative_to_us_broad_price'][str(n)] = {
                'benchmark': code, 'return_difference': value - base if value is not None and base is not None else None}
    chains, industries = defaultdict(list), defaultdict(list)
    for row in output:
        if row['kind_captured'] != 'STOCK':
            continue
        for chain in set(row['chains']):
            chains[(row['market'], chain)].append(row)
        if row['industry']:
            industries[(row['market'], row['industry'])].append(row)
    names = {g['id']: g.get('name', g['id']) for g in run.value_chain.get('groups', [])}
    role_counts = {role: sum(r['role'] == role for r in output)
                   for role in ('holding', 'watch_only', 'watch_membership_unconfirmed', 'background')}
    return {
        'schema_version': 'stock_review_insight_context_v1',
        'context_created_at': datetime.now(timezone.utc).isoformat(),
        'base_run_path': str(run.path),
        'base_run_id': run.meta['run_id'], 'data_run_id': run.meta['data_run_id'],
        'report_run_id': run.meta['report_run_id'], 'batch_status': run.meta['status'],
        'information_cutoff': run.meta['information_cutoff'], 'market_cutoffs': run.data['market_cutoffs'],
        'membership': {'observed_at': run.members.get('observed_at'), 'capture_errors': sanitized_errors,
                       'holdings_snapshot_query_succeeded': holdings_known, 'current_verified': False},
        'baseline': {'run_id': baseline.meta['run_id'], 'data_run_id': baseline.meta['data_run_id'],
                     'report_run_id': baseline.meta['report_run_id'], 'market_cutoffs': baseline.data['market_cutoffs'],
                     'same_evaluation': baseline.meta['report_run_id'] == run.meta['report_run_id'],
                     'trend_rule_compatible': compatible} if baseline else None,
        'auxiliary_source_states': run.data.get('auxiliary_sources', {}),
        'counts': {'securities': len(output), **role_counts,
                   'industry_labels_missing_stocks': sum(r['kind_captured'] == 'STOCK' and not r['industry'] for r in output)},
        'sector_etfs': [{'code': c, 'name': name, 'present': c in by_code,
                         'price_returns': by_code[c]['price_returns'] if c in by_code else None}
                        for c, name in SECTORS.items()],
        'value_chain_sample_statistics': [{'market': market, 'group_id': key, 'group_name': names.get(key, key),
                                           **group_summary(items)} for (market, key), items in sorted(chains.items())],
        'business_industry_sample_statistics': [{'market': market, 'industry': key, **group_summary(items)}
                                                for (market, key), items in sorted(industries.items())],
        'limitations': ['price_statistics_are_historical_not_trade_probabilities',
                       'membership_is_snapshot_not_current_verification', 'news_and_quotes_not_requested',
                       'sector_samples_overlap_and_are_not_full_market_breadth',
                       'asset_types_and_product_terms_require_verification'],
        'securities': output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    sources = [args.run.resolve()] + ([args.baseline.resolve()] if args.baseline else [])
    forbidden = [args.repo.resolve() / '.cache/stock_review_pair_v1'] + sources
    for source in sources:
        location = source / 'data_location.json'
        data = source / read(location)['root'] if location.exists() else source.parent.parent / 'data'
        forbidden.append(data.resolve())
    if any(target.is_relative_to(root) for root in forbidden):
        raise ValueError('output must be outside frozen pair runs and shared store')
    if target.exists():
        raise FileExistsError('output already exists: ' + str(target))
    result = build(args.run, args.repo, args.baseline)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(target), 'base_run_id': result['base_run_id'],
                      'counts': result['counts'], 'sector_etfs_present': sum(r['present'] for r in result['sector_etfs']),
                      'news': 'not_requested', 'buy_sell_signals': 'not_computed'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
