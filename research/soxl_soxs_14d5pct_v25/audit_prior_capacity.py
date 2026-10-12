"""Describe v24 proposed versus executable choices; no retrospective selector."""
from pathlib import Path
import sys,hashlib
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v24'))
from joint_loss import core
from verify_continuation import read


def audit():
    run=core.ROOT/'.cache/soxl_soxs_14d5pct_v24/runs/joint_records_20261012'
    manifest=read(core.ROOT/'research/soxl_soxs_14d5pct_v24/evidence/delivery_manifest.json')
    for n in ['joint_hgb_decisions.csv','panel.parquet','results.json','verification.json']:
        assert hashlib.sha256((run/n).read_bytes()).hexdigest()==manifest['cache_outputs_sha256'][n]
    panel=pd.read_parquet(run/'panel.parquet');ds=pd.read_csv(run/'joint_hgb_decisions.csv')
    cutoff=pd.Timestamp('2026-05-01',tz=core.ET);rows=[]
    for e in ds.itertuples():
        if e.requested_action not in core.ETFS:continue
        at=pd.Timestamp(e.decision_at);p=panel.loc[at];known=pd.notna(p.terminal_release) and p.terminal_release<cutoff
        net=float(p['rL' if e.requested_action=='SOXL' else 'rS']) if known else None
        rows.append({'at':at,'date':p.date,'symbol':e.requested_action,'allocation':e.requested_allocation,
            'account_decision':e.decision,'executed':e.decision.startswith('buy_'),'known':known,'net_return':net})
    by={}
    for status in sorted({r['account_decision'] for r in rows}):
        group=[r for r in rows if r['account_decision']==status];known=[r for r in group if r['known']]
        by[status]={'proposals':len(group),'known':len(known),'unknown':len(group)-len(known),'profitable':sum(r['net_return']>0 for r in known)}
    for n,take in [('executed',True),('skipped',False)]:
        g=[r for r in rows if r['executed']==take];known=[r for r in g if r['known']]
        by[n]={'proposals':len(g),'known':len(known),'unknown':len(g)-len(known),'profitable':sum(r['net_return']>0 for r in known)}
    out={'source_run':str(run),'prior_result_posthoc_diagnostic_only':True,'new_policy_evaluated':False,
        'counterfactual_returns_are_not_extra_capital_trades':True,'groups':by,'rows':rows}
    core.dump(Path(__file__).parent/'evidence/prior_capacity_audit.json',out);print(by)


if __name__=='__main__':audit()
