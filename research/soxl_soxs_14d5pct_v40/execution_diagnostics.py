"""Separate first issued opportunities and actual capital-admissible entries."""
import argparse
from collections import defaultdict,Counter
from nonlinear_direction import *

def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    root=Path(read(run/'input_manifest.json')['source']);panel=pd.read_parquet(root/'financial_panel.parquet');events=read(run/'decisions_frozen.json')['routes'];out={}
    for name in ['tree_band','stump_band','linear_band']:
        proposals=[e for e in events[name] if e['action']!='CASH'];ds=pd.read_csv(run/(name+'_decisions.csv'));ds.index=pd.to_datetime(ds.decision_at,utc=True)
        groups=defaultdict(list)
        for e in proposals:groups[e['date']].append(e)
        first=[sorted(es,key=lambda e:e['at'])[0] for es in groups.values()];parts={'all':proposals,'first_of_date':first,
            'actual_buy':[e for e in proposals if ds.loc[pd.Timestamp(e['at']),'decision'].startswith('buy_')],
            'unavailable':[e for e in proposals if not ds.loc[pd.Timestamp(e['at']),'decision'].startswith('buy_')]}
        stats={}
        for kind,es in parts.items():
            rows=[];unknown=[]
            for e in es:
                t=pd.Timestamp(e['at']);row=panel.loc[t];ret=row.C+(row.D if e['action']=='SOXL' else -row.D)
                if not np.isfinite(ret):unknown.append(e['at']);continue
                rows.append({'date':e['date'],'at':e['at'],'net_log':ret,'positive':float(ret>0),'status':ds.loc[t,'decision']})
            bydate=defaultdict(list)
            for row in rows:bydate[row['date']].append(row)
            # Independent explicit mean-of-date-means, not pandas groupby implementation.
            mean=lambda field:float(np.mean([sum(x[field] for x in rr)/len(rr) for rr in bydate.values()])) if bydate else None
            stats[kind]={'queries':len(es),'dates':len(bydate),'unknown':unknown,'date_equal_net_log':mean('net_log'),'date_equal_positive':mean('positive'),
                'statuses':dict(Counter(x['status'] for x in rows)),'rows':rows}
        t=pd.read_csv(run/(name+'_trades.csv'));assert len(parts['actual_buy'])==len(t)
        for row in stats['actual_buy']['rows']:
            trade=t[t.decision_at==row['at']].iloc[0];np.testing.assert_allclose(np.log1p(trade.net_return),row['net_log'],atol=1e-12)
        earlier=read(run/'selection_diagnostics.json')['summaries'][name]['date_equal'];np.testing.assert_allclose(stats['all']['date_equal_net_log'],earlier['actual_log'],atol=1e-12)
        out[name]=stats
    receipt={'post_result_only':True,'rules_changed':False,'capital_status_comes_from_independently_reconciled_ledger':True,'sample_selection_not_a_new_policy':True,
        'route_groups':out,'results_sha256':sha(run/'results.json')};dump(run/'execution_diagnostics.json',receipt)
    print({n:{k:{f:v for f,v in s.items() if f!='rows'} for k,s in groups.items()} for n,groups in out.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
