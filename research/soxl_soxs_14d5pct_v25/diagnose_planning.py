"""Describe frozen decisions; never choose a replacement policy after results."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from run_planning import read,labels,key,dump,sha


def describe(rows):
    known=[r for r in rows if r['net_return'] is not None and np.isfinite(r['net_return'])]
    return {'proposals':len(rows),'known':len(known),'unknown':len(rows)-len(known),
        'profitable':sum(r['net_return']>0 for r in known),
        'mean_known_single_trade_return':float(np.mean([r['net_return'] for r in known])) if known else None,
        'not_a_capital_return':True}


def audit(run):
    run=Path(run);v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    manifest=read(run/'input_manifest.json');lab=labels(Path(manifest['label_source'])/'baseline_labels.json')
    stress=labels(Path(manifest['label_source'])/'stress_labels.json')
    frozen=read(run/'decisions_frozen.json')['routes'];result=read(run/'results.json');routes={};primary_rows=[]
    for name,events in frozen.items():
        ds=pd.read_csv(run/(name+'_decisions.csv'));ds.index=pd.to_datetime(ds.decision_at,utc=True)
        rows=[]
        for e in events:
            if e['action']=='CASH':continue
            at=pd.Timestamp(e['at']);label=(stress if name=='double_cost' else lab)[key(at,e['action'])]
            value=label.get('net_return');status=ds.loc[at,'decision']
            rows.append({'at':e['at'],'symbol':e['action'],'net_return':value if value is not None and np.isfinite(value) else None,
                'status':status,'executed':status.startswith('buy_'),'waiting_value':e.get('waiting_value'),
                'buy_advantage':e.get('buy_advantage'),'reason':label['reason'],'gap_below_stop':label.get('gap_below_stop',False)})
        groups={s:describe([r for r in rows if r['status']==s]) for s in sorted({r['status'] for r in rows})}
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        routes[name]={'all_proposals':describe(rows),'executed':describe([r for r in rows if r['executed']]),
            'by_account_status':groups,'capital_return':result['routes'][name]['continuous_return'],
            'account_status_counts':dict(Counter(ds.decision)),
            'actual_symbols':dict(Counter(trades.symbol)) if len(trades) else {},
            'complete_trade_reasons':dict(Counter(trades.reason)) if len(trades) else {},
            'executed_gap_losses':sum(r['executed'] and r['gap_below_stop'] and r['net_return'] is not None and r['net_return']<0 for r in rows)}
        if name=='planning_calibrated':primary_rows=rows
    differences={}
    for other in ['myopic_calibrated','planning_raw','myopic_raw','source_early']:
        changes=[]
        for a,b in zip(frozen['planning_calibrated'],frozen[other]):
            assert a['at']==b['at']
            if a['action']!=b['action']:changes.append({'at':a['at'],'primary':a['action'],'other':b['action']})
        differences[other]={'changed_dates':len(changes),'rows':changes}
    monthly=[]
    for month in sorted({e['date'][:7] for e in frozen['planning_calibrated']}):
        records=[read(run/'training'/(e['date'][:10]+'.json')) for e in frozen['planning_calibrated'] if e['date'][:7]==month]
        ready=[r for r in records if r['events']['planning_calibrated']['ready']]
        monthly.append({'month':month,'ready_dates':len(ready),
            'raw_action_values_mean':np.mean([r['raw']['planning'] for r in ready],axis=0) if ready else None,
            'correction_mean':np.mean([r['calibration']['planning']['correction'] for r in ready],axis=0) if ready else None,
            'corrected_advantage_mean':float(np.mean([r['events']['planning_calibrated']['buy_advantage'] for r in ready])) if ready else None})
    cost_differences=[]
    for a,b in zip(frozen['planning_calibrated'],frozen['double_cost']):
        assert a==b
        if a['action']=='CASH':continue
        k=key(pd.Timestamp(a['at']),a['action']);lo,hi=lab[k],stress[k]
        if lo['exit_at']!=hi['exit_at'] or lo['reason']!=hi['reason']:
            cost_differences.append({'at':a['at'],'symbol':a['action'],'baseline_exit':lo['exit_at'],'stress_exit':hi['exit_at'],
                'baseline_reason':lo['reason'],'stress_reason':hi['reason'],'baseline_net':lo.get('net_return'),
                'stress_net':hi.get('net_return')})
    out={'verified_result_sha256':sha(run/'results.json'),'routes':routes,'decision_differences':differences,
        'calibration_by_month':monthly,'primary_proposals':primary_rows,'double_cost_changed_terminal_proposals':cost_differences,
        'counterfactuals_not_additional_trades':True,'Q_values_are_discounted_log_values_not_probabilities':True,
        'diagnostic_not_selection_or_new_case_authorization':True}
    dump(run/'diagnostics.json',out)
    print({'routes':routes,'changed_dates':{k:x['changed_dates'] for k,x in differences.items()},
        'double_cost_changed_terminal_proposals':len(cost_differences)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
