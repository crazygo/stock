"""Adapters preserve the approved algorithms and complete independent HTML templates."""
from __future__ import annotations

import html
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from datetime import timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, PKG, digest, fingerprint, now, read, write

TREND = ROOT / 'analysis/ai_trend_quadrant_v2'
BEHAVIOR = ROOT / 'analysis/stock_behavior_description_v3'
spec = importlib.util.spec_from_file_location('pair_five_traits_v2', TREND / 'model.py')
MODEL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODEL)


def encoded(data):
    return json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')


def at(bars, days, cutoff):
    frame = [b for b in bars if b[0] <= cutoff][-141:]
    values, states = [None] * 5, ['insufficient_history'] * 5
    if not frame:
        return values, states, None
    returns = np.diff(np.log([b[4] for b in frame]))
    raw = MODEL.measure(returns)[0]
    for k, n in enumerate(MODEL.WINDOWS):
        if len(frame) < n + 1:
            continue
        segment = frame[-n - 1:]
        expected = [d for d in days if segment[0][0] <= d <= cutoff]
        if not expected or segment[0][0] < days[0]:
            states[k] = 'unverified_calendar'
        elif [b[0] for b in segment] != expected or len(expected) != n + 1:
            states[k] = 'missing_sessions'
        elif not np.isfinite(raw[k]):
            states[k] = 'undefined_zero_variation'
        else:
            values[k], states[k] = float(raw[k]), 'scored'
    lengths = [n for k, n in enumerate(MODEL.WINDOWS) if states[k] == 'scored']
    return values, states, returns[-max(lengths):] if lengths else None


def trend_record(entry, days, end, cache):
    key = fingerprint([entry['dataset_id'], end, days, digest(TREND / 'model.py'), 'pair_adapter_v1'])
    path = cache / (key + '.json')
    if path.exists():
        return read(path), True
    values, states, returns = at(entry['bars'], days, end)
    uncertainty = MODEL.boot(returns, entry['security_id'], end) if returns is not None else [None] * 5
    history_start = (date.fromisoformat(end) - timedelta(days=59)).isoformat()
    history = []
    for day in days:
        if history_start <= day <= end:
            v, status, _ = at(entry['bars'], days, day)
            history.append({'day': day, 'v': v, 'status': status})
    result = {'v': values, 'status': states,
              'uncertainty': [uncertainty[k] if states[k] == 'scored' else None for k in range(5)],
              'history': history, 'mapped': values[0] is not None and values[1] is not None,
              'bars': len(entry['bars']), 'evaluation_cutoff': end,
              'zero_volume_warning': bool(entry['bars'] and
                                         sum(b[5] <= 0 for b in entry['bars'][-61:]) / min(61, len(entry['bars'])) > .1)}
    write(path, result)
    return result, False


def banner(page, manifest, other):
    cutoffs = '；'.join(f'{k} {v}' for k, v in manifest['market_cutoffs'].items())
    counts = manifest['summary']
    text = (f'数据批次 {manifest["data_run_id"]} · 完整交易日目标：{cutoffs} · '
            f'行情到达目标 {counts["current"]}/{counts["securities"]} · '
            f'财务与评级沿用注明截至日的缓存；小时证据保留其原截止；实时新闻未接入')
    bar = (f'<div id="pair-batch" data-run-id="{html.escape(manifest["run_id"])}" '
           'style="font:12px system-ui;border-bottom:1px solid #aaa;padding:6px 12px;background:#eee;color:#222">'
           f'{html.escape(text)} · <a href="{other}">另一页面</a></div>')
    # Added outside the approved chart/filter hierarchy, so its logic stays intact.
    import re
    return re.sub(r'(<body[^>]*>)', lambda m: m.group(1) + bar, page, count=1)


def build(destination, manifest, rows, value_chain, cal, store):
    entries = {d['security_id']: d for d in manifest['datasets']}
    end = max(manifest['market_cutoffs'].values())
    run_id = manifest['run_id']
    records = []
    reused = 0
    for i, row in enumerate(rows.values()):
        code, market = row['code'], row['code'][:2]
        entry = entries[code]
        result, cached = trend_record(entry, cal[market]['days'], manifest['market_cutoffs'][market],
                                      store / 'calculations/trend')
        reused += cached
        records.append({**row, **result, 'favorite': 'special' in row['groups'],
                        'ai_related': bool(row.get('chains')), 'quality_grade': (row.get('quality') or {}).get('grade'),
                        'coverage': {**entry['coverage'], 'path': entry.get('path'), 'sha256': entry.get('sha256'),
                                     'price_basis': entry['price_basis'], 'source': entry['source'],
                                     'received_at': entry.get('received_at'), 'last_day': entry['coverage']['actual_end']}})
        if (i + 1) % 25 == 0:
            print(f'计算五标签 {i + 1}/{len(rows)}', flush=True)
    results = {'version': MODEL.VERSION, 'as_of': end, 'history_start': (date.fromisoformat(end) - timedelta(days=59)).isoformat(),
               'generated_at': now(), 'watchlist_observed_at': manifest['universe']['observed_at'],
               'period_kind': 'calendar_days', 'periods': [14, 30, 60], 'keys': MODEL.KEYS,
               'names': MODEL.NAMES, 'windows': MODEL.WINDOWS, 'records': records, 'run_id': run_id,
               'data_run_id': manifest['data_run_id'], 'market_cutoffs': manifest['market_cutoffs'],
               'observation_mode': manifest['observation_mode'], 'history_observation_mode': 'historical_reconstruction',
               'summary': {'securities': len(records), 'mapped': sum(r['mapped'] for r in records),
                           'unmapped': sum(not r['mapped'] for r in records),
                           'trait_coverage': {MODEL.KEYS[k]: dict(Counter(r['status'][k] for r in records)) for k in range(5)}},
               'provenance': [{k: d.get(k) for k in ['security_id', 'path', 'sha256', 'source']} for d in entries.values()]}
    prices = {'version': 'daily_weekly_price_projection_v1', 'source_run_id': run_id, 'as_of': end,
              'mini_start': manifest['requirements']['start'], 'bar_fields': ['day', 'open', 'high', 'low', 'close'],
              'calendars': cal, 'records': [], 'data_run_id': manifest['data_run_id']}
    for row in rows.values():
        entry = entries[row['code']]
        prices['records'].append({'code': row['code'], 'market': row['code'][:2], 'currency': entry['currency'],
                                  'listing_date': row.get('listing_date'), 'bars': [b[:5] for b in entry['bars']],
                                  'status': 'available' if entry['bars'] else 'unavailable', 'invalid_days': [],
                                  **{k: entry.get(k) for k in ['source', 'path', 'sha256', 'price_basis', 'received_at']},
                                  'first_day': entry['coverage']['actual_start'], 'last_day': entry['coverage']['actual_end'],
                                  'missing_sessions': entry['coverage']['missing_sessions'],
                                  'calendar_unverified_bars': sum(not cal[row['code'][:2]]['start'] <= b[0] <= cal[row['code'][:2]]['end']
                                                                  for b in entry['bars'])})
    prices['summary'] = {'securities': len(rows), 'available': sum(bool(d['bars']) for d in entries.values()),
                         'daily_bars': sum(len(d['bars']) for d in entries.values()), 'invalid_bars': 0}
    for row in rows.values():
        row['price_ref'] = 'original'
    comparison = {'version': 'multi_stock_groups_v1', 'built_at': now(), 'as_of': end,
                  'group_observed_at': manifest['universe']['observed_at'], 'quality_as_of': read(destination / 'inputs/quality.json')['as_of'],
                  'membership_not_PIT': True, 'default_calendar_days': 60, 'records': list(rows.values()),
                  'extra_prices': [], 'value_chain_groups': value_chain['groups'],
                  'summary': {'members': len(rows), 'with_prices': prices['summary']['available'],
                              'group_counts': dict(Counter(g for r in rows.values() for g in r['groups']))}}
    template = (TREND / 'template.html').read_text()
    for key, data in [('DATA', results), ('PRICES', prices), ('COMPARISON', comparison)]:
        template = template.replace('__' + key + '__', encoded(data))
    for key, filename in [('TIMELINE', 'timeline.js'), ('PRICE_CHART', 'price_chart.js'),
                           ('DAILY_PATTERNS', 'daily_patterns.js'), ('COMPARISON_UI', 'comparison_ui.js')]:
        script = (TREND / filename).read_text()
        if key == 'COMPARISON_UI':
            # Keep a shared chart right edge, while scoring each market at its own
            # completed cutoff. HK closing today must not mark all US stocks stale.
            script = script.replace('DP.quadrant(price,cal,cmp.end)',
                'DP.quadrant(price,cal,cmp.end<D.market_cutoffs[price.market]?cmp.end:D.market_cutoffs[price.market])')
        template = template.replace('__' + key + '__', script)
    trend_out = destination / 'pages/ai_trend_quadrant_v2'
    trend_out.mkdir(parents=True)
    (trend_out / 'index.html').write_text(banner(template, manifest, '../stock_behavior_description_v3/index.html'))
    for key, data in [('results', results), ('prices', prices), ('comparison', comparison)]:
        write(destination / f'results/trend/{key}.json', data)

    fundamental = {r['code']: r['fundamental'] for r in read(destination / 'inputs/fundamentals.json')['records']}
    special = [r for r in rows.values() if 'special' in r['groups']]
    members = {'observed_at': manifest['universe']['observed_at'], 'group_name': '特别关注', 'members': special}
    bprices = {'as_of': end, 'requested_start': prices['mini_start'],
               'records': {code: d['bars'] for code, d in entries.items()}, 'us_sessions': cal['US']['days'],
               'us_early_close': cal['US']['early_close'],
               'hk_calendar': {'start': cal['HK']['start'], 'end': cal['HK']['end'], 'holidays': cal['HK']['holidays']},
               'sources': list(dict.fromkeys(cal['US']['source'] + cal['HK']['source'])),
               'calendars': cal, 'market_cutoffs': manifest['market_cutoffs']}
    payload = {'prices': bprices, 'intraday': read(destination / 'inputs/intraday.json'), 'members': members,
               'records': [{**r, 'coverage': {**entries[r['code']]['coverage'],
                            **{k: entries[r['code']].get(k) for k in ['source', 'price_basis', 'received_at', 'path', 'sha256']}},
                            'fundamental': fundamental.get(r['code'],
                              {'status': '未接入公司资料', 'facts': {}, 'leader': {'status': '待核验'},
                               'sources': [], 'missing': ['缺少公司资料'], 'risks': []})} for r in special],
               'manifest': {'data_manifest': {'path': 'inputs/data_manifest.json',
                                              'sha256': digest(destination / 'inputs/data_manifest.json')},
                            'prices.json': {'sha256': fingerprint(bprices)}},
               'version': 'historical_path_description_v3', 'model_sha256': digest(BEHAVIOR / 'core.js'),
               'generated_at': now(), 'data_run_id': manifest['data_run_id'], 'run_id': run_id,
               'view_version': 'historical_comparison_wall_v3', 'price_renderer_sha256': digest(BEHAVIOR / 'price_overlay.js')}
    behavior_out = destination / 'pages/stock_behavior_description_v3'
    behavior_out.mkdir(parents=True)
    for output, filename, full in [('index.html', 'comparison_template.html', False),
                                   ('detail.html', 'detail_template.html', True)]:
        source = (BEHAVIOR / filename).read_text()
        for key, script in [('CORE', 'core.js'), ('PRICE_OVERLAY', 'price_overlay.js'), ('SLOPE_OVERLAY', 'slope_overlay.js')]:
            source = source.replace('__' + key + '__', (BEHAVIOR / script).read_text())
        data = payload if full else {k: v for k, v in payload.items() if k != 'intraday'}
        source = source.replace('__DATA__', encoded(data)).replace('这是2026-10-07附近缓存的公司资料', '这是注明实际截至日的缓存公司资料')
        source = source.replace("US:{start:'2024-01-01',end:'2026-12-31',days:P.us_sessions,early_close:P.us_early_close}",
                                'US:P.calendars.US')
        source = source.replace('53证券有效日线截至10-07，HK.03006／HK.07747截至10-06。',
                                '各市场完整日线目标 ${Object.entries(P.market_cutoffs).map(([k,v])=>k+" "+v).join("；")}。')
        source = source.replace('默认截至已取得完整日线的 2026-10-07。',
                                '默认终点为本批次的历史观察右端，各证券实际行情截止单列。')
        source = source.replace('全55股共用条形刻度', '全${D.records.length}股共用条形刻度')
        (behavior_out / output).write_text(banner(source, manifest, '../ai_trend_quadrant_v2/index.html'))
    write(destination / 'results/behavior_input.json', {k: v for k, v in payload.items() if k != 'intraday'})
    import shutil
    node = shutil.which('node') or str(Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
    subprocess.run([node, str(PKG / 'evaluate_behavior.cjs'), str(destination)], check=True)
    return {'trend_records': len(records), 'behavior_records': len(special), 'calculation_cache_reused': reused}
