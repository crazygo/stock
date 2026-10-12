"""Post-run allocation attribution only; cannot change frozen predictions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from allocation import core,ROUTES
from verify_continuation import read


def diagnose(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];settings=p[reg['phase']]
    source=Path(read(run/'input_manifest.json')['source_run']);bank=pd.read_parquet(run/'distribution_bank.parquet')
    prior={pd.Timestamp(e['at']):e for e in read(source/'predictions.json')['regret_regime']}
    frozen=read(run/'decisions_frozen.json')['routes'];cutoff=pd.Timestamp(settings['evaluation_end'],tz=core.ET)+pd.Timedelta(days=1)
    out={'post_run_diagnostic_only':True,'new_decisions_or_parameters_created':False,'routes':{},'overlapping_opportunities_are_not_capital_returns':True}
    for name in ROUTES:
        rows=[];unknown=[];allocations={};effective=[]
        for e in frozen[name]:
            at=pd.Timestamp(e['at']);fraction=e['allocation'];allocations[str(fraction)]=allocations.get(str(fraction),0)+1
            history=read(run/'training'/(e['date'][:10]+'.json'))['routes'][name]['fit'];info=history['distribution']
            if info is not None:effective.append(info['effective_dates'])
            if not e['ready']:continue
            b=bank.loc[(at,e['source_top'])]
            if pd.isna(b.release) or b.release>=cutoff:unknown.append(e['date'][:10]);continue
            w=np.array(info['weights']);r=np.array(info['returns']);source_event=prior[at]
            rows.append({'date':e['date'][:10],'allocation':fraction,'actual_return':float(b.net_return),
                'predicted_return':info['mean_return'],'predicted_log':float(w@np.log1p(r)),
                'source_predicted_log':source_event['adjusted'][e['source_top']],
                'actual_log':float(np.log1p(b.net_return)),
                'source_allowed':source_event['action']!='CASH','new_allowed':fraction>0})
        f=pd.DataFrame(rows);buy=f[f.new_allowed];over=f[f.new_allowed&~f.source_allowed];denied=f[~f.new_allowed&f.source_allowed]
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        out['routes'][name]={'evaluation_dates':len(frozen[name]),'ready_dates':sum(e['ready'] for e in frozen[name]),
            'mature_dates':len(f),'unknown_dates':unknown,'decision_allocation_counts':allocations,
            'effective_dates_range':[float(min(effective)),float(max(effective))] if effective else None,
            'predicted_mean_log':float(f.predicted_log.mean()) if len(f) else None,'source_predicted_mean_log':float(f.source_predicted_log.mean()) if len(f) else None,
            'actual_mean_log':float(f.actual_log.mean()) if len(f) else None,
            'distribution_log_mean_mse':float(np.mean((f.predicted_log-f.actual_log)**2)) if len(f) else None,
            'source_log_mean_mse':float(np.mean((f.source_predicted_log-f.actual_log)**2)) if len(f) else None,
            'positive_allocation_mature_dates':len(buy),'positive_allocation_actual_mean_return':float(buy.actual_return.mean()) if len(buy) else None,
            'new_buys_where_source_abstained':{'mature_dates':len(over),'actual_mean_return':float(over.actual_return.mean()) if len(over) else None},
            'abstentions_where_source_bought':{'mature_dates':len(denied),'actual_mean_return':float(denied.actual_return.mean()) if len(denied) else None},
            'allocation_slices':{str(a):{'dates':len(g),'predicted_mean_return':float(g.predicted_return.mean()),
                'actual_mean_return':float(g.actual_return.mean()),'source_allowed':int(g.source_allowed.sum())} for a,g in f.groupby('allocation')},
            'actual_trade_allocation_counts':{str(k):int(v) for k,v in trades.allocation.value_counts().sort_index().items()} if len(trades) else {},
            'actual_trade_mean_allocation':float(trades.allocation.mean()) if len(trades) else None}
    core.dump(run/'diagnosis.json',out);print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
