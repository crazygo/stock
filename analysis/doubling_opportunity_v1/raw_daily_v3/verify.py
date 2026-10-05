#!/usr/bin/env python3
"""Verify frozen real v3 outputs without treating engineering success as effectiveness."""
from __future__ import annotations
import hashlib,importlib.util,json,sys
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(HERE));from market import normalized,labels
from actions import audited_events
from prepare_actions import load_audit
sys.path.insert(0,str(PARENT));from data import CACHE

def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    pointer=json.loads((HERE/'latest_backtest.json').read_text());back=Path(pointer['path']);lineage=json.loads((back/'lineage.json').read_text())
    checked=0
    for path,digest in {**lineage['source_hashes'],**lineage['training_source_hashes'],**lineage['financial_input_hashes'],**lineage['derived_hashes'],**lineage['artifact_hashes']}.items():
        require(sha(path)==digest,'Frozen artifact/source mismatch: '+path);checked+=1
    for t,r in lineage['native_sources'].items():
        for kind in ['daily','rehab']:require(sha(r[kind+'_path'])==r[kind+'_sha256'],'Native source mismatch: '+t);checked+=1
    registration=lineage['registration'];members=registration['members'];state=lineage['acquisition_state']
    action_audit,action_ready=load_audit(validate_sources=True)
    require(action_ready==lineage['action_source_audit'],'Action registry differs from frozen dataset')
    require(set(action_audit['registry'])==set(members) and action_audit['full_registered_source_terminal'] and action_audit['recovery_terminal'],'Action audit omitted a registered member or unfinished recovery')
    require(state['terminal'] and state['full_registered_pool'] and state['completed']==len(members),'Incomplete full registered source')
    base=Path(pointer['source_path']);cover=pd.read_csv(base/'coverage.csv',keep_default_na=False)
    require(len(cover)==len(members) and set(cover.ticker)==set(members),'Coverage loses registered member')
    audit=json.loads((back/'split_audit.json').read_text());require([r['month'] for r in audit]==lineage['config']['evaluation_months'],'Missing frozen evaluation month')
    for r in audit:
        if r['status']=='development_joint_calibrated':
            require(pd.Timestamp(r['train_latest_label_maturity'])<pd.Timestamp(r['calibration_start']),'Training label leaks into calibration')
            require(pd.Timestamp(r['cal_latest_label_maturity'])<pd.Timestamp(r['evaluation_start']),'Calibration label leaks into evaluation')
    d=pd.read_parquet(pointer['dataset_path']);available=d.available_at.notna()
    require((d.loc[available,'available_at']<=d.loc[available,'date']).all(),'Future financial disclosure entered a feature')
    require(not (d.joint_signal_eligible & ((d.fin_report_age_days>183)|d.unsupported_action_feature_window)).any(),'Stale/unsupported state was signalled')
    pred=pd.read_parquet(pointer['prediction_path']);require(pred[['p30','p60','probability']].notna().all().all(),'Missing scored probability')
    require(((pred.p30>=0)&(pred.p30<=pred.p60)&(pred.p60<=1)).all(),'Horizon probability ordering failure')
    sig=[]
    for file in sorted(back.glob('signals_*.csv')):
        q=pd.read_csv(file,keep_default_na=False,na_values=['']);q['date']=pd.to_datetime(q.date);sig.append(q)
        require((q.probability>lineage['config']['threshold']).all(),'Low-confidence row entered emitted signals')
        for _,g in q.groupby('ticker'):
            require((g.sort_values('date').date.diff().dropna().dt.days>=lineage['config']['signal_cooldown_calendar_days']).all(),'Overlapping signals expanded supply')
        joined=q.merge(d[['ticker','date','joint_signal_eligible']],on=['ticker','date'],how='left',validate='many_to_one')
        require(joined.joint_signal_eligible.fillna(False).all(),'Signal fails contemporaneous joint eligibility')
    sig=pd.concat(sig,ignore_index=True) if sig else pd.DataFrame()
    index=pd.DatetimeIndex([r['session_date'] for r in lineage['calendar']['sessions']]);recomputed=0
    if len(sig):
        for t,g in sig.groupby('ticker'):
            source=lineage['native_sources'][t];raw=pd.read_parquet(source['daily_path']);events=pd.read_parquet(source['rehab_path'])
            events,_=audited_events(events,action_audit['registry'][t],pointer['asof'])
            _,adjusted,_,unknown,_,_,_=normalized(raw,index,events,pointer['asof'])
            for h,q in g.groupby('horizon'):
                lab=labels(adjusted,index,unknown,pointer['asof'],int(h),lineage['config']['reference_cost'])
                for r in q.itertuples():
                    actual=lab.loc[r.date,f'y{h}'];require((pd.isna(actual) and pd.isna(r.label)) or actual==r.label,'Emitted signal future label does not recompute');recomputed+=1
    run=Path(json.loads((HERE/'latest_run.json').read_text())['path']);report=json.loads((run/'report.json').read_text())
    require(sha(HERE/'run.py')==report['metadata']['report_code_sha256'],'Report generator changed after this report')
    require(sha(run/'quote_snapshot.json')==report['metadata']['quote_snapshot_sha256'],'Published quote source snapshot changed')
    stocks=pd.read_csv(run/'all_stocks.csv',keep_default_na=False,na_values=[''])
    require(len(stocks)==len(registration['universe']['stocks']) and stocks.ticker.is_unique,'Report lost or duplicated whole-pool stock')
    require(set(stocks.ticker)==set(registration['universe']['stocks']),'Report changes registered stock names')
    require({'source_status','source_original_status','source_original_rows','source_recovery_status','source_current_session_covered'}.issubset(stocks.columns),'Full stock report omits source/recovery outcomes')
    coverage=cover.set_index('ticker');published=stocks.set_index('ticker')
    require((published.source_status==coverage.loc[published.index,'source_status']).all(),'Report changed source availability outcomes')
    require(published.p30.notna().equals(published.p60.notna()),'Only one current horizon was published')
    forecast_rows=published[published.p60.notna()]
    require(((forecast_rows.p30>=0)&(forecast_rows.p30<=forecast_rows.p60)&(forecast_rows.p60<=1)).all(),'Published current probabilities violate ordering')
    models,current_audit=joblib.load(pointer['model_path'])
    latest=d[d.date==pd.Timestamp(pointer['asof'])].set_index('ticker');eligible=latest[latest.eligible]
    expected_scored=set(eligible.index) if models else set()
    require(set(forecast_rows.index)==expected_scored,'Current report omitted a scored stock or scored an ineligible stock')
    if models and len(eligible):
        spec=importlib.util.spec_from_file_location('doubling_v3_verification_probability_model',PARENT/'joint_v2/model.py')
        _model=importlib.util.module_from_spec(spec);spec.loader.exec_module(_model)
        model=models[current_audit['champion']]
        p30,p60,_=_model.nested_probabilities(model['calibrated'],eligible,model['features'])
        actual=published.loc[eligible.index]
        require(np.allclose(actual.p30,p30,rtol=1e-10,atol=1e-12) and np.allclose(actual.p60,p60,rtol=1e-10,atol=1e-12),'Published current probability does not reproduce from the frozen model')
    require(len(forecast_rows)==report['metadata']['price_scored_count'],'Current report scored-count mismatch')
    require(report['metadata']['data_asof']==pointer['asof'],'Report changed model anchor')
    require(report['metadata']['qualified_count']==0 and report['qualified_stocks']==[],'Unverified effect was called qualified')
    stats=json.loads((back/'backtest_summary.json').read_text())
    require(all(r['signals']==r['tp']+r['fp']+r['unknown'] for r in stats),'Failure/unknown removed from denominator')
    require(all(not r['deployment_qualified'] for r in stats),'Unverified evidence upgraded to deployment')
    evidence={'verified_at':datetime.now(timezone.utc).isoformat(),'version':lineage['version'],'engineering_checks_passed':True,
        'goal_achieved':False,'qualified_count':0,'source_and_artifact_sha_checks':checked,'registered_count':len(members),
        'equity_candidates':len(stocks),'source_available_count':len(lineage['native_sources']),'signal_labels_recomputed_including_strategy_duplicates':recomputed,
        'model_reference_asof':pointer['asof'],'probability_order_violations':0,'six_month_evaluation_months':lineage['config']['evaluation_months'],
        'current_forecasts_recomputed':len(forecast_rows),'all_stock_source_outcomes_verified':True,
        'backtest':[{k:r[k] for k in ['horizon','algorithm','signals','tp','fp','unknown','precision','historical_numeric_gate']} for r in stats],
        'pending':report['metadata']['pending'],'report_path':str(run),'training_key':pointer['training_key'],
        'report_files_sha256':{p.name:sha(p) for p in run.iterdir() if p.is_file()},
        'verification_code_sha256':sha(Path(__file__))}
    (HERE/'CURRENT_EVIDENCE.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(evidence,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
