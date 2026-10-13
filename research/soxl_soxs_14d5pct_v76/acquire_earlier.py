"""Local/R2/OpenD data extension using the existing versioned downloader."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess, sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
CACHE = ROOT/'.cache/soxl_soxs_14d5pct_v76'
sys.path.insert(0, str(ROOT/'research/soxl_soxs_14d5pct_v2'))
from acquire import normalize
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())


def main():
    for p, digest in read(SOURCE/'evidence/acquisition_registration.json')['source_sha256'].items():
        assert sha(ROOT/p) == digest
    assert read(CACHE/'time_audit.json')['passed']
    inventory = read(ROOT/'.cache/soxl_soxs_14d5pct_v75/older_history_inventory.json')
    assert all('objects' in r for r in inventory['r2'])
    protocol = read(SOURCE/'DATA_PROTOCOL.json')
    assert not (CACHE/'acquisition_status.json').exists(), 'Review existing run before resume'
    outcomes = []
    for symbol in protocol['symbols']:
        output = CACHE/'minutes'/symbol
        assert not output.exists(), 'Never overwrite prior acquisition'
        output.mkdir(parents=True)
        if symbol == 'QQQ':
            parts = []; captures = []
            for path in sorted((ROOT/'market_data/model_training_history_v1/parts/QQQ').glob('*/bars.parquet')):
                if '2023-04' <= path.parent.name <= '2024-10':
                    frame = pd.read_parquet(path)
                    assert frame.price_basis.eq('NONE').all()
                    target = output/'captured/local_parts'/path.parent.name/'bars.parquet'
                    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path,target)
                    parts.append(normalize(frame, symbol, 'local_QQQ_monthly_parts'))
                    captures.append(dict(path=str(path.relative_to(ROOT)), capture=str(target.relative_to(ROOT)), sha256=sha(path)))
            frame = pd.concat(parts).sort_values('end_at').drop_duplicates('end_at',keep='first')
            frame = frame[frame.session_date.between(protocol['extension_start'],protocol['overlap_end'])]
            target = output/'bars/QQQ.parquet';target.parent.mkdir(parents=True)
            frame.to_parquet(target,index=False,compression='zstd',compression_level=7)
            (output/'local_parts.json').write_text(json.dumps(captures,indent=2)+'\n')
        command = [sys.executable, '-W','error','-W','default::DeprecationWarning',
                   str(ROOT/'research/soxl_soxs_14d5pct_v2/acquire.py'), '--output',str(output),
                   '--symbols',symbol,'--start',protocol['extension_start'],'--end',protocol['overlap_end']]
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (output/'worker.log').open('w') as log:
            try:
                cp = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                outcome = dict(symbol=symbol,returncode=cp.returncode,started_at=started,
                    finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    complete=cp.returncode==0)
            except subprocess.TimeoutExpired:
                outcome = dict(symbol=symbol,timeout=True,complete=False,started_at=started)
        outcomes.append(outcome)
        (CACHE/'acquisition_status.json').write_text(json.dumps(outcomes,indent=2)+'\n')
        print(outcome,flush=True)
        if not outcome['complete']: raise SystemExit('Acquisition stopped for evidence review')


if __name__ == '__main__': main()
