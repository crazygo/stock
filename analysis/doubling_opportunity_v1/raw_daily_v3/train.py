#!/usr/bin/env python3
"""Frozen v3 experiment, optionally wait for its registered full acquisition."""
from __future__ import annotations
import argparse,fcntl,hashlib,importlib.util,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import joblib,pandas as pd
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(HERE));from build import build,FEATURES,VERSION
sys.path.insert(0,str(PARENT));from data import CACHE
_spec=importlib.util.spec_from_file_location('doubling_v3_event_model',PARENT/'joint_v2/model.py')
_model=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_model)

def write_status(status,**extra):
    p=CACHE/'raw_daily_v3_training_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'status':status,'updated_at':datetime.now(timezone.utc).isoformat(),**extra},indent=2));tmp.replace(p)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--wait-for-acquisition',action='store_true')
    ap.add_argument('--wait-timeout-minutes',type=int,default=240)
    ap.add_argument('--reuse-if-unchanged',action='store_true');args=ap.parse_args()
    deadline=time.monotonic()+args.wait_timeout_minutes*60;waited=False
    while True:
        state_path=CACHE/'futu_daily_v3_acquisition.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('error'):raise RuntimeError('Registered acquisition reported a failure; inspect its live process and error before resuming')
        if state.get('terminal') and state.get('full_registered_pool'):
            ready=HERE/'ACTION_AUDIT_READY.json'
            audit=json.loads(ready.read_text()) if ready.exists() else {}
            if audit.get('ready_for_first_training') and audit.get('source_snapshot_path')==state.get('path'):break
            if not args.wait_for_acquisition:raise RuntimeError('Full source finished; corporate-action source audit is not yet ready for this snapshot')
            if time.monotonic()>=deadline:raise TimeoutError('Source audit wait expired; no unaudited model was fitted')
            write_status('waiting_for_registered_action_audit',source_path=state.get('path'))
            print('v3 waiting full corporate-action audit and source recovery',flush=True)
            waited=True;time.sleep(45);continue
        if not args.wait_for_acquisition:raise RuntimeError('The full registered acquisition is not complete')
        if time.monotonic()>=deadline:raise TimeoutError('Bounded acquisition wait expired; no subset was evaluated')
        write_status('waiting_for_registered_full_source',completed=state.get('completed'),registered_count=state.get('registered_count'),source_path=state.get('path'))
        print('v3 waiting full source',state.get('completed'),'/',state.get('registered_count'),flush=True)
        waited=True;time.sleep(45)
    if waited:os.execv(sys.executable,[sys.executable,str(Path(__file__).resolve()),'--wait-for-acquisition','--wait-timeout-minutes',str(args.wait_timeout_minutes)]+(['--reuse-if-unchanged'] if args.reuse_if_unchanged else []))
    lock=(CACHE/'raw_daily_v3_training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    write_status('building_frozen_native_and_financial_dataset')
    dataset,manifest=build();cfg=json.loads((HERE/'config.json').read_text())
    # Actual model source and training protocol are both frozen in this evidence.
    training_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'TRAINING_PROTOCOL.md',HERE/'config.json']}
    training_key=hashlib.sha256(json.dumps([manifest['key'],training_hashes],sort_keys=True).encode()).hexdigest()
    out=HERE/'backtests'/training_key[:16]
    prediction_name=VERSION+'_'+training_key[:16]+'_forward_predictions.parquet'
    mp=Path(manifest['path'])/('current_models_'+training_key[:16]+'.joblib')
    pointer={'path':str(out.resolve()),'dataset_key':manifest['key'],'training_key':training_key,'dataset_path':manifest['dataset_path'],'panel_path':manifest['panel_path'],
        'asof':manifest['asof'],'model_path':str(mp),'model_month':pd.Timestamp(manifest['asof']).strftime('%Y-%m'),
        'prediction_path':str(CACHE/prediction_name),'source_path':manifest['path'],
        'regular_hours_label_verified':False,'deployment_qualified':False}
    if args.reuse_if_unchanged and (out/'lineage.json').exists() and mp.exists():
        old=json.loads((out/'lineage.json').read_text())
        if old.get('training_key')==training_key and old.get('artifact_hashes') and all(Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==s for p,s in old['artifact_hashes'].items()):
            (HERE/'latest_backtest.json').write_text(json.dumps(pointer,indent=2))
            subprocess.run([sys.executable,str(HERE/'run.py'),'--top','30'],check=True)
            subprocess.run([sys.executable,str(HERE/'verify.py')],check=True)
            write_status('complete_development_experiment_reused',**pointer);return
    write_status('fitting_monthly_forward_models',dataset_key=manifest['key'],training_key=training_key)
    with threadpool_limits(limits=4):
        pred,audits,stats=_model.forward_backtest(dataset,cfg,out,FEATURES,VERSION,prediction_name,'joint_signal_eligible')
        month=pd.Timestamp(manifest['asof']).strftime('%Y-%m')
        models,current_audit=_model.fit_month(dataset,month,cfg,FEATURES,VERSION)
    for row in stats:
        row['blocking_evidence']=[x for x in row['blocking_evidence'] if x not in ['corporate_action_vintages_not_reconciled','joint_quality_and_catalyst_not_backtested']]+[
            'complete_corporate_action_and_historical_identity_coverage_unverified','unsupported_actions_and_native_source_quality_retained','business_catalyst_history_not_backtested',
            'current_quote_entry_price_differs_from_daily_close_if_market_moved']
    (out/'backtest_summary.json').write_text(json.dumps(stats,indent=2))
    joblib.dump((models,current_audit),mp)
    artifacts=[mp,CACHE/prediction_name,out/'backtest_summary.json',out/'split_audit.json',out/'backtest_signals.csv']+list(out.glob('signals_*.csv'))
    artifact_hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    (out/'lineage.json').write_text(json.dumps({**manifest,'config':cfg,'training_key':training_key,'training_source_hashes':training_hashes,'artifact_hashes':artifact_hashes},ensure_ascii=False,indent=2))
    (HERE/'latest_backtest.json').write_text(json.dumps(pointer,indent=2))
    write_status('historical_experiment_complete_reports_running',**pointer)
    print(json.dumps([{k:r[k] for k in ['horizon','algorithm','signals','tp','fp','unknown','precision','historical_numeric_gate']} for r in stats],indent=2),flush=True)
    subprocess.run([sys.executable,str(HERE/'run.py'),'--top','30'],check=True)
    subprocess.run([sys.executable,str(HERE/'verify.py')],check=True)
    write_status('complete_development_experiment',**pointer)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_status('failed',error_type=type(exc).__name__,error=str(exc)[:500]);raise
