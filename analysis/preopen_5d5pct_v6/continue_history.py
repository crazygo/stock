"""Bounded window orchestration, per-symbol atomic acquisition then preparation.

Leaves incomplete inputs and failures explicit. Never writes legacy artifacts.
"""
import argparse,json,subprocess,sys,time
from datetime import datetime,timezone
from common import OUT,OLD,write,now

def run(args,name):
    with (OUT/'logs'/name).open('a') as log:
        subprocess.run([sys.executable,'-u',str(OUT/args[0])]+args[1:],stdout=log,stderr=subprocess.STDOUT,check=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--finalize-only',action='store_true');a=ap.parse_args()
    expected={p.stem for p in (OLD/'raw').glob('*.parquet')};reserve=datetime.fromisoformat('2026-10-05T04:00:00+00:00')
    while not a.finalize_only:
        done={p.stem for p in (OUT/'R03/cache/parts').glob('*.parquet')};missing=expected-done
        if not missing:break
        ready=sorted(s for s in missing if (OUT/'cache/backfill'/f'{s}.json').exists())
        if ready:
            print(json.dumps(dict(at=now(),stage='prepare_completed_prefix',symbols=ready)),flush=True)
            run(['extended_adapter.py','--raw-dir',str(OUT/'cache/backfill'),'--output',str(OUT/'R03/cache/parts'),'--symbols']+ready+['--workers','2'],'R03_prefix_prepare.log')
        elif datetime.now(timezone.utc)>=reserve:
            print(json.dumps(dict(at=now(),stage='deadline_preserve_partial_history',symbols=sorted(missing))),flush=True)
            run(['extended_adapter.py','--raw-dir',str(OLD/'raw'),'--output',str(OUT/'R03/cache/parts'),'--symbols']+sorted(missing)+['--workers','2'],'R03_deadline_fallback.log')
        else:time.sleep(10)
    if not a.finalize_only:
        run(['history_prepare.py'],'R03_prepare.log')
        print(json.dumps(dict(at=now(),stage='R03_finite_training')),flush=True)
        run(['history_research.py','--fit','--workers','2'],'R03_matrix.log')
    metas=[json.loads(p.read_text()) for p in (OUT/'R03/cache/runs').glob('*/meta.json')]
    assert len(metas)==20 and all(m['status']=='completed_development' for m in metas),'R03 matrix has unfinished or failed jobs; inspect preserved logs'
    run(['restore_reporting_prices.py','--round','R03'],'R03_reporting_repair.log')
    run(['calibration_audit.py','--round','R03'],'R03_calibration.log')
    run(['favorite_report.py'],'R03_favorites.log')
    run(['verify.py','--round','R03'],'R03_verify.log')
    run(['export_models.py'],'export_models_R03.log')
    write(OUT/'R03/pipeline_status.json',dict(at=now(),status='completed',independent_pass=False))
    print(json.dumps(dict(at=now(),stage='R03_completed')),flush=True)
if __name__=='__main__':main()
