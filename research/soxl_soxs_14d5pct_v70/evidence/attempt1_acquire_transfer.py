"""Versioned, bounded auxiliary input acquisition; no model or trade context."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
CACHE = ROOT / '.cache/soxl_soxs_14d5pct_v70'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    registration = read(SOURCE / 'evidence/data_registration.json')
    for name, digest in registration['source_sha256'].items():
        assert sha(ROOT / name) == digest
    quota = read(SOURCE / 'evidence/quota_preflight.json')
    assert quota['ok'] and quota['quota']['remain'] >= 5
    inventory = read(SOURCE / 'evidence/alternate_r2_inventory.json')
    assert all('error_type' not in row and not row['objects'] for row in inventory['rows'])
    assert not (CACHE / 'data/manifest.json').exists()
    args = [sys.executable, '-W', 'error',
            str(ROOT / 'research/soxl_soxs_14d5pct_v2/acquire.py'),
            '--output', str(CACHE / 'data'), '--symbols',
            'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SPY',
            '--start', '2024-10-14', '--end', '2026-04-30']
    with (SOURCE / 'evidence/acquisition.log').open('w') as log:
        subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       timeout=1800, check=True)
    receipts = []
    for symbol in ['TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SPY']:
        time.sleep(1.2)
        output = CACHE / 'data/captured/actions' / f'{symbol}.parquet'
        cp = subprocess.run([sys.executable, str(SOURCE / 'action_worker.py'),
                             '--symbol', symbol, '--output', str(output)],
                            cwd=ROOT, capture_output=True, text=True,
                            timeout=30, check=True)
        marker = '@@TRANSFER_ACTION@@ '
        record = json.loads(next(line[len(marker):] for line in cp.stdout.splitlines()
                                 if line.startswith(marker)))
        record.update(path=str(output.relative_to(ROOT)), sha256=sha(output))
        receipts.append(record)
    (SOURCE / 'evidence/action_receipts.json').write_text(
        json.dumps(receipts, indent=2) + '\n')
    print({'minute_acquisition_complete': read(CACHE / 'data/manifest.json')['complete'],
           'action_captures': len(receipts), 'data_audit_pending': True,
           'finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}, flush=True)


if __name__ == '__main__':
    main()
