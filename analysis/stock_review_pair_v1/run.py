#!/usr/bin/env python3
"""One SOP run: prepare latest shared data, calculate both views, validate, publish."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from common import ROOT, PKG, DEFAULT_STORE, digest, fingerprint, lock, now, read, write
from data import Acquisition, calendars, cutoffs, membership
from views import build
from publish import publish
from storage import SharedData, hydrate, price_bars, verify_market


def source_files():
    files = [p for p in PKG.iterdir() if p.suffix in ['.py', '.cjs']]
    for folder, names in [
        ('ai_trend_quadrant_v2', ['model.py', 'template.html', 'timeline.js', 'price_chart.js', 'daily_patterns.js', 'comparison_ui.js']),
        ('stock_behavior_description_v3', ['core.js', 'comparison_template.html', 'detail_template.html', 'price_overlay.js', 'slope_overlay.js']),
        ('stock_traits_daily_v1', ['opend_worker.py']),
    ]:
        files.extend(ROOT / 'analysis' / folder / name for name in names)
    files.extend(ROOT / 'scripts' / name for name in ['model_history_calendar.py', 'r2_client.py'])
    return files


def code_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None  # An exported source bundle need not be inside a Git checkout.


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutoff', default='auto', help='auto or YYYY-MM-DD; completed sessions only')
    p.add_argument('--store', type=Path, default=DEFAULT_STORE)
    p.add_argument('--budget-seconds', type=float, default=900, help='total acquisition budget, not compute budget')
    p.add_argument('--request-timeout', type=float, default=15)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=11111)
    p.add_argument('--max-new-symbols', type=int, default=100)
    p.add_argument('--min-remaining-quota', type=int, default=400)
    p.add_argument('--offline', action='store_true', help='explicit cached input; never label membership freshly queried')
    p.add_argument('--no-opend', action='store_true')
    p.add_argument('--no-publish', action='store_true', help='build and validate a candidate without changing either URL')
    p.add_argument('--from-run', type=Path, help='validate input hashes and replay an immutable run without acquisition')
    p.add_argument('--recompute', action='store_true', help='recalculate all price statistics, bypassing calculation cache')
    p.add_argument('--skip-browser', action='store_true', help='candidate only; cannot publish without browser acceptance')
    args = p.parse_args()
    if args.skip_browser and not args.no_publish:
        p.error('--skip-browser requires --no-publish')
    if args.budget_seconds <= 0 or args.request_timeout <= 0:
        p.error('budgets and timeouts must be positive')
    return args


def verify_inputs(run):
    manifest = read(run / 'inputs/data_manifest.json')
    for name, sha in manifest['input_files'].items():
        if digest(run / name) != sha:
            raise ValueError('input checksum mismatch: ' + name)
    verify_market(run, manifest)
    return manifest


def validate(run, skip_browser=False):
    manifest = verify_inputs(run)
    behavior = read(run / 'results/behavior_input.json')
    prices = read(run / 'results/trend/prices.json')
    trend = read(run / 'results/trend/results.json')
    assert behavior['data_run_id'] == trend['data_run_id'] == manifest['data_run_id']
    for row in prices['records']:
        assert row['bars'] == [b[:5] for b in behavior['prices']['records'][row['code']]]
    for entry in manifest['datasets']:
        assert all(b[0] <= manifest['market_cutoffs'][entry['market']] for b in price_bars(run, entry))
    node = shutil.which('node') or str(Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
    # Compile every embedded script even when a browser runtime is unavailable.
    import re
    for path in (run / 'pages').rglob('*.html'):
        for index, script in enumerate(re.finditer(r'<script([^>]*)>(.*?)</script>', path.read_text(), re.S)):
            if 'application/json' in script.group(1):
                json.loads(script.group(2))
                continue
            if script.group(2).strip():
                temp = run / 'validation' / f'{path.parent.name}-{path.stem}-{index}.js'
                temp.parent.mkdir(parents=True, exist_ok=True)
                temp.write_text(script.group(2))
                subprocess.run([node, '--check', str(temp)], check=True, capture_output=True)
                temp.unlink()
    if not skip_browser:
        subprocess.run([node, str(PKG / 'verify_browser.cjs'), str(run)], check=True, timeout=180)
    write(run / 'validation/acceptance.json', {'status': 'candidate_only' if skip_browser else 'passed',
          'run_id': manifest['run_id'], 'input_hashes': True, 'shared_ohlc': True,
          'future_excluded': True, 'script_syntax': True, 'browser': 'not_run' if skip_browser else 'passed'})


def execute(args):
    args.store = args.store.resolve()
    args.store.mkdir(parents=True, exist_ok=True)
    with lock(args.store / 'run.lock'):
        started = time.monotonic()
        deadline = started + args.budget_seconds
        run_id = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d') + '__run-' + str(uuid.uuid4())
        run = args.store / 'runs' / run_id
        run.mkdir(parents=True)
        started_at = now()
        shared = SharedData(args.store)
        acquisition = None
        sources = {str(p.relative_to(ROOT)): digest(p) for p in source_files()}
        repro = run / 'repro/workspace'
        for name in sources:
            target = repro / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        params = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        kind = 'recompute' if args.recompute and args.from_run else 'render' if args.from_run else 'update'
        write(run / 'run.json', {'run_id': run_id, 'run_kind': kind,
                                 'started_at': started_at, 'information_cutoff': None,
                                 'status': 'preparing', 'parameters': params, 'source_sha256': sources})
        write(args.store / 'latest_attempt.json', {'run_id': run_id, 'started_at': now(), 'status': 'running'})
        try:
            print('1/5 固定成员、目标交易日和数据范围', flush=True)
            if args.from_run:
                source = args.from_run.resolve()
                manifest = verify_inputs(source)
                manifest = shared.replay(source, run, manifest)
                manifest = dict(manifest, run_id=run_id, replay_of=manifest['run_id'])
                capture = read(run / 'inputs/members.json')
                rows = {r['code']: r for r in capture['records']}
                value_chain = read(run / 'inputs/value_chain.json')
                cal = read(run / 'inputs/calendars.json')
            else:
                cal = calendars()
                ends = cutoffs(cal, args.cutoff)
                rows, capture, value_chain = membership(run, args, deadline)
                acquisition = Acquisition(args, run, cal, ends, deadline, shared)
                print(f'2/5 获取同口径日线：{len(rows)}证券；目标 {ends}', flush=True)
                datasets = []
                ordered = sorted(rows.values(), key=lambda r: ('special' not in r['groups'], 'holdings' not in r['groups'], r['code']))
                for i, row in enumerate(ordered):
                    datasets.append(acquisition.acquire(row))
                    if (i + 1) % 20 == 0:
                        print(f'行情 {i + 1}/{len(rows)}；已到目标 {sum(d["coverage"]["actual_end"] == ends[d["market"]] for d in datasets)}', flush=True)
                write(run / 'inputs/calendars.json', cal)
                input_files = {str(p.relative_to(run)): digest(p) for p in (run / 'inputs').glob('*.json')}
                current = sum(d['coverage']['actual_end'] == ends[d['market']] for d in datasets)
                summary = {'securities': len(rows), 'current': current,
                           'coverage': dict(Counter(d['coverage']['status'] for d in datasets))}
                facts = {'datasets': [{k: d.get(k) for k in ['security_id', 'dataset_id', 'sha256', 'coverage']} for d in datasets],
                         'members': list(rows.values()), 'market_cutoffs': ends,
                         'evidence_inputs': {k: v for k, v in input_files.items() if not k.endswith('members.json')}}
                manifest = {'schema_version': 'stock_data_manifest_v1', 'run_id': run_id,
                            'data_run_id': 'pair-data-' + fingerprint(facts)[:20],
                            'observation_mode': 'live' if args.cutoff == 'auto' and not args.offline else 'historical_reconstruction',
                            'started_at': started_at, 'finished_at': now(), 'requested_cutoff': args.cutoff,
                            'information_cutoff': now(), 'market_cutoffs': ends,
                            'universe': {'observed_at': capture['observed_at'], 'members_sha256': fingerprint(list(rows.values())),
                                         'source': capture.get('source'), 'capture_status': capture.get('live_check',
                                         {'status': 'queried' if capture.get('ok') else 'carried_forward'}), 'counts': len(rows)},
                            'requirements': {'start': acquisition.start, 'price_basis': 'QFQ_full_series_single_capture',
                                             'granularity': 'daily', 'minimum_scoring_closes': 141},
                            'datasets': datasets, 'input_files': input_files, 'summary': summary,
                            'auxiliary_sources': {'news': 'not_connected', 'financials': 'carried_forward',
                                                  'quality': 'carried_forward', 'intraday': 'carried_forward'},
                            'errors': acquisition.errors,
                            'status': 'complete' if all(d['coverage']['status'] == 'complete' for d in datasets) else 'partial'}
                shared.facts(run, manifest)
            write(run / 'inputs/data_manifest.json', manifest)
            if acquisition:
                acquisition.cleanup()
            print('3/5 分别计算两套描述并生成完整页面', flush=True)
            details = build(run, hydrate(run, manifest), rows, value_chain, cal, args.store, args.recompute)
            details['shared_storage'] = dict(shared.metrics)
            details['run_parquet_files'] = len(list(run.rglob('*.parquet')))
            evaluation_id = 'pair-report-' + fingerprint([manifest['data_run_id'],
                              digest(ROOT / 'analysis/ai_trend_quadrant_v2/model.py'),
                              digest(ROOT / 'analysis/stock_behavior_description_v3/core.js')])[:20]
            evaluation = args.store / 'evaluations' / evaluation_id / 'snapshot.json'
            if not evaluation.exists():
                write(evaluation, {'report_run_id': evaluation_id, 'data_run_id': manifest['data_run_id'],
                                    'first_calculated_at': now(), 'first_run_id': run_id,
                                    'trend': [{k: r.get(k) for k in ['code', 'v', 'status', 'uncertainty', 'evaluation_cutoff']}
                                              for r in read(run / 'results/trend/results.json')['records']],
                                    'behavior': read(run / 'results/behavior.json')})
                os.chmod(evaluation, 0o444)
                details['evaluation_reused'] = False
            else:
                details['evaluation_reused'] = True
            shutil.copyfile(evaluation, run / 'results/evaluation_snapshot.json')
            if args.from_run and (source / 'results/trend/results.json').exists():
                from review import compare_run
                compare_run(source, run, details, args.recompute)
            print('4/5 校验同批输入、实际图表和交互', flush=True)
            validate(run, args.skip_browser)
            if any(digest(ROOT / name) != sha for name, sha in sources.items()):
                raise RuntimeError('source changed during calculation; candidate kept, publication refused')
            write(run / 'run.json', {'run_id': run_id, 'data_run_id': manifest['data_run_id'], 'report_run_id': evaluation_id,
                                     'run_kind': kind, 'started_at': started_at,
                                     'finished_at': now(), 'information_cutoff': manifest['information_cutoff'],
                                     'parameters': params, 'source_sha256': sources,
                                     'status': manifest['status'], 'market_cutoffs': manifest['market_cutoffs'],
                                     'details': details, 'elapsed_seconds': round(time.monotonic() - started, 2),
                                     'prices': manifest['summary'], 'publication_requested': not args.no_publish,
                                     'fees': 'no model API calls; market-data charges unknown', 'code_sha': code_sha()})
            write(run / 'repro/environment.json', {'python': sys.version, 'platform': sys.platform,
                  'timezone': 'Asia/Shanghai', 'dependencies': {m: __import__(m).__version__ for m in ['pandas', 'numpy', 'pyarrow']}})
            replay = run / 'repro/replay.sh'
            replay.write_text('#!/bin/sh\nset -eu\n'
                'if [ "$#" -ne 1 ]; then echo "usage: replay.sh OUTPUT_STORE" >&2; exit 2; fi\n'
                'SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
                'RUN_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)\n'
                'exec python3 "$SCRIPT_DIR/workspace/analysis/stock_review_pair_v1/run.py" --from-run "$RUN_DIR" --store "$1" --no-publish\n')
            replay.chmod(0o755)
            (run / 'README.md').write_text(f'# 双页面批次 {run_id}\n\n'
                f'行情目标：{manifest["market_cutoffs"]}；覆盖 {manifest["summary"]}。\n\n'
                '离线打开 pages/stock_behavior_description_v3/index.html 或 pages/ai_trend_quadrant_v2/index.html。\n'
                '本运行引用共用 data/ 对象库；移动或分享请先用 export.py 导出完整离线包。\n'
                '断网重算：sh repro/replay.sh /absolute/path/to/new-store；使用封存源码和所引用输入，不接 OpenD/R2。\n'
                '依赖 Python（numpy、pandas、pyarrow）、Node.js 与 Playwright/Chromium。\n'
                '新闻未接入；财务、质量与小时证据沿用原截止缓存。历史时间谱为回算。\n')
            write(run / 'manifest.json', {'schema_version': 'stock_review_pair_bundle_v1', 'run_id': run_id,
                  'data_run_id': manifest['data_run_id'], 'report_run_id': evaluation_id,
                  'market_cutoffs': manifest['market_cutoffs'], 'status': manifest['status'],
                  'modules': manifest['auxiliary_sources'],
                  'storage_schema': manifest.get('storage_schema', 'legacy_inline'),
                  'shared_references': {d['security_id']: {k: d.get(k) for k in ['object_ref', 'raw_ref']}
                                        for d in manifest['datasets']},
                  'source_sha256': sources, 'files': {str(p.relative_to(run)): {'sha256': digest(p), 'bytes': p.stat().st_size}
                                                     for p in run.rglob('*') if p.is_file()}})
            write(run / 'checksums.json', {str(p.relative_to(run)): digest(p) for p in run.rglob('*') if p.is_file()})
            (run / 'checksums.sha256').write_text(''.join(f'{sha}  {name}\n' for name, sha in read(run / 'checksums.json').items())
                                                  + f'{digest(run / "checksums.json")}  checksums.json\n')
            for path in run.rglob('*'):
                if path.is_file():
                    path.chmod(0o555 if path.name == 'replay.sh' else 0o444)
            print('5/5 发布同一批次的两套页面' if not args.no_publish else '5/5 候选已保存；未切换页面', flush=True)
            if not args.no_publish:
                if args.skip_browser:
                    raise ValueError('--skip-browser requires --no-publish')
                publish(args.store, run)
            write(args.store / 'latest_attempt.json', {'run_id': run_id, 'finished_at': now(),
                  'status': 'published' if not args.no_publish else 'candidate', 'path': str(run)})
            print(json.dumps({'run': str(run), 'run_id': run_id, 'data_run_id': manifest['data_run_id'],
                  'report_run_id': evaluation_id, 'market_cutoffs': manifest['market_cutoffs'],
                  'coverage': manifest['summary'], 'details': details,
                  'published': not args.no_publish, 'elapsed_seconds': round(time.monotonic() - started, 2)},
                  ensure_ascii=False), flush=True)
            return run
        except BaseException as exc:
            write(run / 'failure.json', {'run_id': run_id, 'error_type': type(exc).__name__, 'error': str(exc),
                                         'finished_at': now(), 'previous_publication_preserved': True})
            write(args.store / 'latest_attempt.json', {'run_id': run_id, 'status': 'failed',
                                                      'error': type(exc).__name__, 'path': str(run)})
            raise
        finally:
            if acquisition:
                acquisition.cleanup()


if __name__ == '__main__':
    execute(parse())
