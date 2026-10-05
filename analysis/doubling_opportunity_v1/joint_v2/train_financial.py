#!/usr/bin/env python3
"""Fixed complete-cohort financial/market-heat development experiment; no trading."""
from __future__ import annotations
import argparse,importlib.util,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import joblib
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;sys.path.insert(0,str(HERE))
from prepare_joint import prepare,coverage_state,FEATURES,VERSION
# prepare_joint adds the parent directory for shared data imports. Resolve the
# event model by its file so the older parent model cannot be imported here.
_spec=importlib.util.spec_from_file_location('doubling_joint_event_model',HERE/'model.py')
_joint_model=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_joint_model)
forward_backtest=_joint_model.forward_backtest;fit_month=_joint_model.fit_month
sys.path.insert(0,str(PARENT));from data import CACHE


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wait-for-acquisition',action='store_true',help='Wait for both complete registered jobs; never evaluate a download-order subset.')
    ap.add_argument('--wait-timeout-minutes',type=int,default=180)
    args=ap.parse_args();deadline=time.monotonic()+args.wait_timeout_minutes*60
    statepath=CACHE/'joint_financial_training_status.json';waited=False
    while not all(s.get('terminal') for s in coverage_state().values()):
        status={'status':'waiting_for_registered_acquisition','updated_at':datetime.now(timezone.utc).isoformat(),'coverage':coverage_state()}
        statepath.write_text(json.dumps(status,indent=2))
        if not args.wait_for_acquisition:raise RuntimeError('Acquisition is not terminal; use --wait-for-acquisition or retry after complete coverage.')
        if time.monotonic()>=deadline:raise TimeoutError('Registered acquisition did not finish within the bounded wait; no partial-cohort result was evaluated.')
        print('waiting for fixed SEC cohort',[(k,v.get('completed'),v.get('registered_issuers')) for k,v in status['coverage'].items()],flush=True)
        waited=True
        time.sleep(45)
    if waited:
        # Read the final registered source from disk after a long acquisition wait.
        # Never fingerprint a newer source file while executing stale imported code.
        os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve())])
    statepath.write_text(json.dumps({'status':'preparing_frozen_joint_dataset','updated_at':datetime.now(timezone.utc).isoformat()}))
    d,manifest=prepare();cfg=json.loads((PARENT/'config.json').read_text());out=HERE/'financial_backtests'/manifest['key'][:16]
    pred,audits,stats=forward_backtest(d,cfg,out,FEATURES,VERSION,VERSION+'_forward_predictions.parquet','joint_signal_eligible')
    (out/'lineage.json').write_text(json.dumps({**manifest,'config':cfg},indent=2))
    current_month=d.date.max().strftime('%Y-%m');models,audit=fit_month(d,current_month,cfg,FEATURES,VERSION)
    mp=CACHE/'models'/f'{VERSION}_{current_month}_{manifest["key"][:16]}.joblib'
    joblib.dump((models,audit),mp)
    pointer={'path':str(out.resolve()),'dataset_key':manifest['key'],'model_path':str(mp),'model_month':current_month,
             'dataset_path':manifest['path'],'price_data_asof':d.date.max().date().isoformat(),'regular_hours_label_verified':False}
    (HERE/'latest_financial_backtest.json').write_text(json.dumps(pointer,indent=2))
    statepath.write_text(json.dumps({'status':'model_experiment_complete_reports_running','updated_at':datetime.now(timezone.utc).isoformat(),**pointer},indent=2))
    print(json.dumps([{k:r[k] for k in ['horizon','algorithm','signals','tp','fp','unknown','precision','historical_numeric_gate']} for r in stats],indent=2),flush=True)
    subprocess.run([sys.executable,str(PARENT/'scan.py'),'--top','30','--debug'],check=True)
    subprocess.run([sys.executable,str(HERE/'build_debug.py'),'--experiment','financial'],check=True)
    statepath.write_text(json.dumps({'status':'complete_development_experiment','updated_at':datetime.now(timezone.utc).isoformat(),**pointer},indent=2))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        (CACHE/'joint_financial_training_status.json').write_text(json.dumps({
            'status':'failed','updated_at':datetime.now(timezone.utc).isoformat(),
            'error_type':type(exc).__name__,'error':str(exc)},indent=2))
        raise
