"""Read-only feasibility inventory for a separate transfer-learning candidate.

No prices or outcomes are fetched, and no frozen v69 input is changed.
R2 probes only known 5m keys. Errors retain type, never credential-bearing text.
"""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.r2_client import R2Client


def inspect():
    symbols = ['TQQQ', 'SQQQ', 'UPRO', 'SPXU']
    rows = []
    for symbol in symbols:
        for prefix in ['us_5m', 'model_training_history_v1/us_5m',
                       'us_60m_raw', 'us_day']:
            for year in [2024, 2025, 2026]:
                path = ROOT / 'market_data' / prefix / symbol / f'{year}.parquet'
                rows.append({'symbol': symbol, 'path': str(path.relative_to(ROOT)),
                             'exists': path.is_file(),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                             if path.is_file() else None,
                             'eligible_5m_candidate': prefix == 'us_5m'})
        path = ROOT / 'market_data/corporate_actions' / f'{symbol}.parquet'
        rows.append({'symbol': symbol, 'path': str(path.relative_to(ROOT)),
                     'exists': path.is_file(),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()
                     if path.is_file() else None,
                     'eligible_5m_candidate': False})
    remote = []
    deadline = time.monotonic() + 30
    client = R2Client(timeout=5, deadline=deadline)
    for symbol in symbols:
        for year in [2024, 2025, 2026]:
            key = f'us_5m/{symbol}/{year}.parquet'
            record = {'key': key}
            try:
                meta = client.head_object(key)
                record.update(status='found' if meta is not None else 'not_found',
                              metadata=meta)
            except Exception as exc:
                record.update(status='unknown', error_type=type(exc).__name__)
            remote.append(record)
    return {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'candidate': 'cross_pair_transfer_not_v69_policy',
            'known_v69_development_result': True,
            'intended_first_data_range': ['2024-10-14', '2026-04-30'],
            'local': rows, 'remote': remote, 'raw_prices_downloaded': False,
            'OpenD_called': False, 'R2_uploaded': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'Standard known local namespaces and exact primary R2 5m keys only. '
                     'A missing object is not proof that no other archive exists. '
                     'Metadata alone does not establish coverage, price basis, or corporate actions.'}


if __name__ == '__main__':
    output = Path(__file__).with_name('evidence') / 'transfer_input_inventory.json'
    if output.exists():
        raise SystemExit('Preserve existing capture; choose a new snapshot for another query')
    value = inspect()
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print({'local_present': sum(row['exists'] for row in value['local']),
           'remote_statuses': {status: sum(row['status'] == status for row in value['remote'])
                               for status in ['found', 'not_found', 'unknown']},
           'output': str(output)})
