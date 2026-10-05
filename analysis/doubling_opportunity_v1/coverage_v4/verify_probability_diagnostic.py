#!/usr/bin/env python3
"""Recompute the frozen high-estimate diagnostic labels directly from native sources."""
from __future__ import annotations
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from actions import audited_events
from market import normalized,labels
from prepare_actions import load_audit

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    pointer=json.loads((HERE/'latest_backtest.json').read_text());back=Path(pointer['path'])
    lineage=json.loads((back/'lineage.json').read_text());dp=json.loads((HERE/'latest_probability_diagnostic.json').read_text())
    require(dp['training_key']==pointer['training_key'],'Diagnostic uses another model')
    root=Path(dp['path']);dl=json.loads((root/'lineage.json').read_text());report=json.loads((root/'report.json').read_text())
    for file,digest in dl['artifacts'].items():require(sha(file)==digest,'Diagnostic artifact changed: '+file)
    require(sha(pointer['prediction_path'])==dl['registration']['prediction_sha256'],'Frozen diagnostic predictions changed')
    audit,ready=load_audit(validate_sources=True);require(ready==lineage['action_source_audit'],'Another action source audit')
    index=pd.DatetimeIndex([x['session_date'] for x in lineage['calendar']['sessions']]);sources={};count=0
    pred=pd.read_parquet(pointer['prediction_path'])
    for s in report['summary']:
        f=root/f"signals_{s['algorithm']}_{s['horizon']}.csv"
        q=pd.read_csv(f,keep_default_na=False,na_values=['']) if f.exists() else pd.DataFrame(columns=pred.columns)
        q['date']=pd.to_datetime(q.date);require(len(q)==s['signals'],'Diagnostic supply mismatch')
        known=q[q.label.notna()];require(int(known.label.sum())==s['tp'] and len(known)-s['tp']==s['fp'] and len(q)-len(known)==s['unknown'],'Diagnostic TP/FP/unknown changed')
        for t,g in q.groupby('ticker'):
            require((g.sort_values('date').date.diff().dropna().dt.days>=lineage['config']['signal_cooldown_calendar_days']).all(),'Diagnostic expanded overlapping windows')
            if t not in sources:
                r=lineage['native_sources'][t]
                for kind in ['daily','rehab']:require(sha(r[kind+'_path'])==r[kind+'_sha256'],'Native source changed: '+t)
                ev,_=audited_events(pd.read_parquet(r['rehab_path']),audit['registry'][t],pointer['asof'])
                _,adjusted,_,unknown,_,_,_=normalized(pd.read_parquet(r['daily_path']),index,ev,pointer['asof'])
                sources[t]=(adjusted,unknown)
            adjusted,unknown=sources[t];h=s['horizon'];lab=labels(adjusted,index,unknown,pointer['asof'],h,lineage['config']['reference_cost'])
            for r in g.itertuples():
                actual=lab.loc[r.date];expected=actual[f'y{h}']
                require((pd.isna(expected) and pd.isna(r.label)) or expected==r.label,'Native label mismatch: '+t)
                require(actual[f'label_status{h}']==r.label_status,'Unknown reason removed: '+t)
                require(r.probability>lineage['config']['threshold'],'Threshold changed')
                rows=pred[(pred.ticker==t)&(pred.date==r.date)&(pred.horizon==h)]
                rows=rows[rows.selected_algorithm] if s['algorithm']=='monthly_selected' else rows[rows.algorithm==s['algorithm']]
                require(len(rows)==1 and np.isclose(rows.iloc[0].probability,r.probability,rtol=1e-12,atol=1e-14),'Diagnostic probability changed: '+t)
                count+=1
    evidence={'verified_at':datetime.now(timezone.utc).isoformat(),'key':dp['key'],'training_key':pointer['training_key'],
        'native_label_checks_including_strategy_duplicates':count,'distinct_native_stocks':len(sources),
        'unknowns_retained_and_recomputed':True,'diagnostic_only':True,'goal_achieved':False,
        'verification_code_sha256':sha(Path(__file__))}
    verification=root/'VERIFICATION.json'
    if verification.exists():
        prior=json.loads(verification.read_text())
        require(prior['key']==dp['key'] and prior['verification_code_sha256']==sha(Path(__file__)),'Existing diagnostic verification differs')
        print(json.dumps(prior,indent=2));return
    verification.write_text(json.dumps(evidence,indent=2))
    dl['artifacts'][str(verification.resolve())]=sha(verification)
    (root/'lineage.json').write_text(json.dumps(dl,indent=2))
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':main()
