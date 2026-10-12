"""Post-result probability, proposal and executed-capital diagnostics."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from joint_loss import core
from run_joint import sha
from verify_continuation import read


def probability_summary(P,Y):
    if not len(Y):return {'n':0}
    P=np.array(P);Y=np.array(Y,int);truth=np.eye(4)[Y]
    binary=np.column_stack([(Y&2)>0,(Y&1)>0]);marginal=np.column_stack([P[:,2]+P[:,3],P[:,1]+P[:,3]])
    return {'n':len(Y),'joint_brier':float(np.square(P-truth).sum(axis=1).mean()),
        'joint_logloss_clipped1e12':float(-np.log(np.maximum(P[np.arange(len(Y)),Y],1e-12)).mean()),
        'profit_brier_pair_average':float(np.square(marginal-binary).mean()),
        'class_counts':dict(Counter(map(int,Y))),
        'side_probability':{s:float(marginal[:,i].mean()) for i,s in enumerate(core.ETFS)},
        'side_realized_profit_fraction':{s:float(binary[:,i].mean()) for i,s in enumerate(core.ETFS)}}


def selected_summary(rows):
    if not rows:return {'n':0,'unknown':0}
    known=[r for r in rows if r['known']];bins=[]
    for lo,hi in zip([0.,.2,.4,.6,.8],[.2,.4,.6,.8,1.]):
        members=[r for r in known if lo<=r['probability'] and (r['probability']<hi or hi==1. and r['probability']<=hi)]
        bins.append({'lo':lo,'hi':hi,'n':len(members),'mean_probability':float(np.mean([r['probability'] for r in members])) if members else None,
                     'realized_profit_fraction':float(np.mean([r['profit'] for r in members])) if members else None})
    return {'n':len(known),'unknown':len(rows)-len(known),'mean_probability':float(np.mean([r['probability'] for r in known])) if known else None,
        'realized_profit_fraction':float(np.mean([r['profit'] for r in known])) if known else None,
        'brier':float(np.mean([(r['probability']-r['profit'])**2 for r in known])) if known else None,'reliability_bins':bins}


def diagnose(run):
    run=Path(run);reg=read(run/'registration.json');settings=reg['preregistration']['protocol'][reg['phase']]
    panel=pd.read_parquet(run/'panel.parquet');panel=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])]
    cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive']);records={at:read(run/'training'/(str(r.date.date())+'.json')) for at,r in panel.iterrows()}
    out={'phase':reg['phase'],'post_result_descriptive_only':True,'results_sha256':sha(run/'results.json'),'diagnostic_source_sha256':sha(Path(__file__)),
        'probabilities':{},'capital':{},'definitions':{'joint_brier':'mean sum over4classes squared error','profit_brier_pair_average':'mean overdates then2ETF binary outcomes',
        'proposals':'eligible model capital actions,one perdate;not all executed','executed':'actual ledger entries',
        'maturity':'reportcommon fullwindow maturity and earlier jointterminal maturity separately;overlapnotadditive'}}
    for kind in ['hgb','logit']:
        for mode in ['raw','adjusted','empirical_prior']:
            for maturity in ['release','terminal_release']:
                ready=[at for at in panel.index if records[at]['learners'][kind]['ready']]
                ids=[at for at in ready if pd.notna(panel.loc[at,maturity]) and panel.loc[at,maturity]<cutoff]
                P=[records[at]['learners'][kind]['fit']['prior'] if mode=='empirical_prior' else records[at]['learners'][kind][mode] for at in ids]
                out['probabilities'][kind+'_'+mode+'_'+maturity]={**probability_summary(P,list(panel.loc[ids,'joint_class'])),
                    'ready':len(ready),'unknown_among_ready':len(ready)-len(ids)}
    result=read(run/'results.json')
    for n,s in result['routes'].items():
        ds=pd.read_csv(run/(n+'_decisions.csv'));bought=ds[ds.decision.str.startswith('buy_')]
        try:trades=pd.read_csv(run/(n+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        row={'mean_monthly_return':s['primary_average'],'monthly':s['monthly'],'actual_entries':len(bought),'completed':s['completed_trades'],
            'allocation_counts':dict(Counter(bought.requested_allocation)),'decision_statuses':dict(Counter(ds.decision)),
            'exit_reasons':dict(Counter(trades.reason)) if len(trades) else {}}
        if n not in ['source_early','cash','double_cost']:
            proposals=[];executed=[];actual={pd.Timestamp(t) for t in bought.decision_at}
            for at,r in panel.iterrows():
                saved=records[at]['capital'][n];e=saved['event'];d=saved['distribution']
                if e['action']=='CASH':continue
                p=d['probabilities'];s=e['action'];margin=p[2]+p[3] if s=='SOXL' else p[1]+p[3]
                known=pd.notna(r.terminal_release) and r.terminal_release<cutoff
                event={'at':at,'known':known,'probability':margin,'profit':bool(r['rL' if s=='SOXL' else 'rS']>0) if known else None}
                proposals.append(event)
                if at in actual:executed.append(event)
            row.update(proposal_probability=selected_summary(proposals),executed_probability=selected_summary(executed),
                model_capital_reasons=dict(Counter(records[at]['capital'][n]['event']['reason'] for at in panel.index)))
        out['capital'][n]=row
    core.dump(run/'diagnostics.json',out);print({n:{'entries':r['actual_entries'],'fractions':r['allocation_counts']} for n,r in out['capital'].items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
