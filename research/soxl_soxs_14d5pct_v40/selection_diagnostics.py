"""Locate the gap between whole-pool prediction errors and emitted proposals."""
import argparse
from nonlinear_direction import *

def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    root=Path(read(run/'input_manifest.json')['source']);panel=pd.read_parquet(root/'financial_panel.parquet');frozen=read(run/'decisions_frozen.json')['routes']
    cutoff=pd.Timestamp(read(root/'input_manifest.json')['observation_cutoff_exclusive']);summaries={};sets={};trade_sets={}
    for name in ['tree_band','stump_band','linear_band','tree_point']:
        rows=[];pending=[];sets[name]=set()
        for e in frozen[name]:
            if e['action']=='CASH':continue
            at=pd.Timestamp(e['at']);sets[name].add((e['at'],e['action']));row=panel.loc[at]
            if pd.isna(row.terminal_release) or row.terminal_release>=cutoff or not np.isfinite(row.C) or not np.isfinite(row.D):pending.append(e['at']);continue
            actual=row.C+(row.D if e['action']=='SOXL' else -row.D)
            rows.append({'at':at,'date':row.date,'symbol':e['action'],'actual_log':actual,'positive':float(actual>0),'lower_or_point':e['buy_advantage']})
        data=pd.DataFrame(rows);summary=data.drop(columns=['at','symbol']).groupby('date').mean(numeric_only=True).mean().to_dict() if rows else None
        t=pd.read_csv(run/(name+'_trades.csv'));trade_sets[name]={(x.decision_at,x.symbol):{'at':x.decision_at,'symbol':x.symbol,'reason':x.reason,'net_return':x.net_return} for x in t.itertuples()}
        summaries[name]={'emitted_proposals':len(sets[name]),'known_queries':len(rows),'known_dates':data.date.nunique() if rows else 0,'unknown_queries':pending,'date_equal':summary,
            'actual_entries':len(t),'actual_targets':int(t.reason.eq('target').sum()),'actual_stops':int(t.reason.eq('risk_stop').sum())}
    changes={}
    for name in ['tree_band','stump_band','tree_point']:
        current=trade_sets[name];old=trade_sets['linear_band']
        changes[name]={'common_proposals':len(sets[name]&sets['linear_band']),'added_proposals':len(sets[name]-sets['linear_band']),'removed_proposals':len(sets['linear_band']-sets[name]),
            'added_actual':[current[k] for k in sorted(current.keys()-old.keys())],'removed_actual':[old[k] for k in sorted(old.keys()-current.keys())]}
    receipt={'post_result_only':True,'rules_changed':False,'qualification_at_issuance_not_future_win':True,'hypothetical_overlapping_proposals_not_actual_trades':True,
        'summaries':summaries,'changes':changes,'results_sha256':sha(run/'results.json')}
    dump(run/'selection_diagnostics.json',receipt);print(summaries,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
