"""Bounded read-only Local -> R2 -> OpenD supplement for the two missing held ETFs.

Raw Parquet stays in ignored captures. No R2 upload, account query or trading action.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import sys
import time
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CODES = ['US.DISK', 'US.IHE']
PREFIX = '@@STDV1_RESULT@@ '


def basic_worker():
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    q = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
    try:
        ret, data = q.get_stock_basicinfo(ft.Market.US, stock_type=ft.SecurityType.ETF, code_list=CODES)
        result = {'ok': ret == ft.RET_OK, 'members': []}
        if ret == ft.RET_OK:
            result['members'] = [{k: str(r.get(k, '')) for k in ['code', 'name', 'stock_type', 'listing_date']}
                                 for r in data.to_dict('records')]
        print(PREFIX + json.dumps(result), flush=True)
    finally:
        q.close()


def call_worker(args, seconds=25):
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=seconds)
    line = next((s[len(PREFIX):] for s in p.stdout.splitlines() if s.startswith(PREFIX)), None)
    if not line:
        return {'ok': False, 'error_type': 'worker_no_result'}
    return json.loads(line)


def main():
    started = time.monotonic()
    run_id = 'comparison-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = ROOT / 'market_data/trend_quadrant_v2/captures' / run_id / 'daily'
    dest.mkdir(parents=True, exist_ok=False)
    prices = json.loads((OUT / 'prices.json').read_text())
    result = {'data_run_id': run_id, 'requested_cutoff': prices['as_of'], 'requested_start': prices['mini_start'],
              'started_at': datetime.now(timezone.utc).isoformat(), 'datasets': [], 'attempts': [],
              'r2_uploaded': False, 'max_new_quota_securities': 2, 'time_budget_seconds': 120}
    sys.path.insert(0, str(ROOT))
    from scripts.r2_client import R2Client
    client = R2Client(timeout=5, deadline=started + 18)
    pending = []
    previous_path = OUT / 'comparison_acquisition.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    reusable = {d['security_id']: d for d in previous.get('datasets', [])
                if d.get('last', '') >= prices['as_of'] and d.get('autype') == 'QFQ'}
    for code in CODES:
        local = ROOT / 'market_data/trend_quadrant_v1/daily' / (code + '.parquet')
        cached = reusable.get(code)
        if cached:
            candidate = ROOT / cached['path']
            if candidate.exists() and hashlib.sha256(candidate.read_bytes()).hexdigest() == cached.get('file_sha256'):
                local = candidate
        path = dest / (code + '.parquet')
        if local.exists():
            path.write_bytes(local.read_bytes()); source = 'local_daily_qfq'
            result['attempts'].append({'code': code, 'stage': 'local', 'status': 'hit'})
        else:
            result['attempts'].append({'code': code, 'stage': 'local', 'status': 'miss'})
            key = 'market_data/trend_quadrant_v1/daily/' + code + '.parquet'
            try:
                head = client.head_object(key)
                if head:
                    client.get_object(key, path); source = 'r2_daily_qfq'
                    result['attempts'].append({'code': code, 'stage': 'r2', 'status': 'hit', 'key': key})
                else:
                    result['attempts'].append({'code': code, 'stage': 'r2', 'status': 'miss', 'key': key})
                    pending.append(code); continue
            except Exception as exc:
                result['attempts'].append({'code': code, 'stage': 'r2', 'status': 'unavailable', 'error_type': type(exc).__name__})
                pending.append(code); continue
        result['datasets'].append({'security_id': code, 'kind': 'market_daily', 'source': source, 'path': str(path.relative_to(ROOT)),
                                   'price_basis': 'single_full_series_QFQ', 'autype': 'QFQ'})
    worker = ROOT / 'analysis/stock_traits_daily_v1/opend_worker.py'
    common = ['--host', '127.0.0.1', '--port', '11111']
    if pending:
        probe = call_worker([str(worker), 'probe', *common], 12)
        quota = probe.get('history_kl_quota', {})
        result['quota_before'] = quota
        if not probe.get('ok') or quota.get('remain', 0) < 402:
            result['status'] = 'partial'; result['blocked_reason'] = 'quote_gateway_or_quota_guard'
        else:
            for code in pending:
                if time.monotonic() - started > 85:
                    result['attempts'].append({'code': code, 'stage': 'opend', 'status': 'budget_exhausted'}); continue
                path = dest / (code + '.parquet'); temp = path.with_suffix('.partial.parquet')
                try:
                    fetched = call_worker([str(worker), 'kline_day', '--code', code, '--start', prices['mini_start'],
                                           '--end', prices['as_of'], '--out', str(temp), '--autype', 'QFQ', *common], 25)
                    result['attempts'].append({'code': code, 'stage': 'opend', 'status': 'hit' if fetched.get('ok') and fetched.get('rows') else 'failed',
                                               'rows': fetched.get('rows'), 'requests': fetched.get('requests'), 'error_type': fetched.get('error_type')})
                    if fetched.get('ok') and fetched.get('rows'):
                        pd.read_parquet(temp)  # A complete readable file is required before publishing.
                        temp.replace(path)
                        result['datasets'].append({'security_id': code, 'kind': 'market_daily', 'source': 'opend_day_qfq',
                                                   'path': str(path.relative_to(ROOT)), 'price_basis': 'single_full_series_QFQ', 'autype': 'QFQ'})
                except subprocess.TimeoutExpired:
                    result['attempts'].append({'code': code, 'stage': 'opend', 'status': 'timeout'})
                time.sleep(1.2)
    try:
        basics = call_worker([str(Path(__file__)), '--metadata-worker'], 12)
        meta = {r['code']: r for r in basics.get('members', [])}
    except subprocess.TimeoutExpired:
        meta = {}
    for item in result['datasets']:
        path = ROOT / item['path']; frame = pd.read_parquet(path)
        item.update(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(frame),
                    first=str(frame.time_key.min())[:10], last=str(frame.time_key.max())[:10], basic_info=meta.get(item['security_id']))
    result['status'] = 'complete' if len(result['datasets']) == 2 else 'partial'
    result['received_at'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    (OUT / 'comparison_acquisition.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'captured': [d['security_id'] for d in result['datasets']],
                      'attempts': result['attempts'], 'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--metadata-worker', action='store_true'); args = parser.parse_args()
    if args.metadata_worker: basic_worker()
    else: main()
