"""Compare immutable raw/calibrated scores to matured paired payoff after the run."""
import argparse
from supported import *

def diagnose(run):
    run=Path(run);verification=read(run/'verification.json');assert verification['passed'] and verification['results_sha256']==sha(run/'results.json')
    root=Path(read(run/'input_manifest.json')['verified_source']);panel=pd.read_parquet(root/'financial_panel.parquet')
    cutoff=pd.Timestamp(read(root/'input_manifest.json')['observation_cutoff_exclusive']);outputs={}
    for name,start,end in [('development','2026-01-01','2026-04-30'),('cases','2026-05-01','2026-09-30')]:
        part=panel[panel.date.between(start,end)];stats={};records=[]
        for at,row in part.iterrows():
            r=read(root/'training'/(tag(at)+'.json'));mature=bool(pd.notna(row.terminal_release) and row.terminal_release<cutoff and np.isfinite(row.C) and np.isfinite(row.D))
            ready=r['estimates']['full']['ready'];base={'at':at,'date':row.date,'mature':mature,'ready':ready,'source_proposed':r['events']['full_band']['action']!='CASH'}
            if mature and ready:
                truth=np.array([row.C+row.D,row.C-row.D])
                for kind,x in [('raw',r['raws']['full']),('calibrated',r['estimates']['full']['point'])]:
                    x=np.array(x);j=int(x[1]>x[0]);base[kind]={'score':x[j],'chosen_positive':float(truth[j]>0),'chosen_net_log':truth[j],
                        'squared_error':float(np.mean((x-truth)**2)),'bias':float(np.mean(x-truth)),
                        'regret':float(np.max(truth)-truth[j]),'direction_correct':float((x[1]>x[0])==(truth[1]>truth[0])),
                        'predicted_positive':float(x[j]>0)}
            records.append(base)
        for kind in ['raw','calibrated']:
            rows=[{'date':r['date'],**r[kind]} for r in records if kind in r]
            frame=pd.DataFrame(rows);stats[kind]=frame.groupby('date').mean().mean().to_dict() if len(frame) else None
        outputs[name]={'queries':len(records),'ready':sum(r['ready'] for r in records),'mature_paired_queries':sum(r['mature'] for r in records),
            'scored_dates':len(set(r['date'] for r in records if 'raw' in r)),'unknown_paired_queries':sum(not r['mature'] for r in records),
            'source_proposals':sum(r['source_proposed'] for r in records),'date_equal_statistics':stats}
    receipt={'post_result_only':True,'rules_changed':False,'not_trade_returns_or_independent_samples':True,
        'development_labels_here_use_extended_cutoff':True,'paired_maturity_cutoff':cutoff,'statistics':outputs,
        'source_verification_sha256':sha(root/'verification.json'),'case_results_sha256':sha(run/'results.json')}
    dump(run/'forecast_diagnostics.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
