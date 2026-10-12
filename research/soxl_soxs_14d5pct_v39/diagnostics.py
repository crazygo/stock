"""Describe all actual trade changes and proposal errors; never alter frozen rules."""
import argparse
from collections import Counter
from supported import *

def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];start,end=settings['evaluation_start'],settings['evaluation_end']
    panel=pd.read_parquet(run/'selected_panel.parquet');panel=panel[panel.date.between(start,end)]
    predictions=read(run/'decisions_frozen.json')['routes'];primary=predictions['all_supported'];source=predictions['source_v35']
    counts={n:dict(Counter(e['action'] for e in es)) for n,es in predictions.items()}
    qualified=[(e,s) for e,s in zip(primary,source) if s['action'] in core.ETFS]
    reasons=Counter(e['reason'] for e,s in qualified);qrows=[]
    for e,s in qualified:
        at=pd.Timestamp(e['at']);rec=read(run/'training'/(tag(at)+'.json'));estimate=rec['estimates']['all_context'];row=panel.loc[at]
        qrows.append({'at':at,'date':row.date,'source_symbol':s['action'],'new_action':e['action'],'reason':e['reason'],
            'quality':estimate['quality'],'outside_columns':estimate.get('outside_columns',[]),'selected_net_log_return':row.selected_y,
            'terminal_release':row.terminal_release,'source_fallback':estimate['source_fallback']})
    def trades(name):
        try:t=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:return {}
        return {(x.decision_at,x.symbol):{'decision_at':x.decision_at,'symbol':x.symbol,'exit_at':x.exit_at,'reason':x.reason,'net_return':x.net_return,'capital':x.capital} for x in t.itertuples()}
    base=trades('source_v35');changes={}
    for name in predictions:
        t=trades(name);changes[name]={'added':[t[k] for k in sorted(t.keys()-base.keys())],'removed':[base[k] for k in sorted(base.keys()-t.keys())]}
    # All months are reported even when not in the requested mean; no future score drives a rule.
    monthly=[{'month':m['month'],'primary':m['return'],'source':next(x['return'] for x in r['routes']['source_v35']['monthly'] if x['month']==m['month'])} for m in r['primary']['monthly']]
    receipt={'post_result_only':True,'rules_changed':False,'phase':r['phase'],'candidate_counts':counts,
        'qualified_reason_counts':dict(reasons),'qualified_queries':qrows,'trade_changes':changes,'monthly_comparison':monthly,
        'results_sha256':sha(run/'results.json'),'pending_retained':r['primary']['pending_trades']}
    dump(run/'diagnostics.json',receipt);print({k:v for k,v in receipt.items() if k not in ['qualified_queries','trade_changes']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
