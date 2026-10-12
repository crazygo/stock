"""Count opportunity supply by clock separately from actual ledger execution."""
import argparse
from pathlib import Path
from collections import Counter
from clock_reliability import core,pd,np,read,labels,dump,sha,CLOCKS,tag
from verify_continuation import key


def summary(rows):
    known=[r for r in rows if r['return'] is not None and np.isfinite(r['return'])]
    return {'ticks':len(rows),'dates':len({r['at'][:10] for r in rows}),'known':len(known),'unknown':len(rows)-len(known),
        'profitable':sum(r['return']>0 for r in known),'mean_known_trade_return':float(np.mean([r['return'] for r in known])) if known else None,
        'ticks_are_not_independent_trades':True,'mean_is_not_capital_return':True}


def diagnose(run):
    run=Path(run);v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    frozen=read(run/'decisions_frozen.json')['routes'];base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json')
    out={'routes':{},'clock_supply':{},'results_sha256':sha(run/'results.json'),'post_result_diagnostic_not_policy_selection':True}
    choices={};entries={}
    for name,events in frozen.items():
        ds=pd.read_csv(run/(name+'_decisions.csv'));ds.index=pd.to_datetime(ds.decision_at,utc=True);rows=[]
        for e in events:
            if e['action']=='CASH':continue
            at=pd.Timestamp(e['at']);lab=(stress if name=='double_cost' else base)[key(at,e['action'])];net=lab.get('net_return');status=ds.loc[at,'decision']
            rows.append({'at':e['at'],'symbol':e['action'],'clock':at.tz_convert(core.ET).strftime('%H:%M:%S'),
                'return':net if net is not None and np.isfinite(net) else None,'status':status,'executed':status.startswith('buy_'),
                'gap_loss':bool(lab.get('gap_below_stop',False) and net is not None and np.isfinite(net) and net<0)})
        executed=[x for x in rows if x['executed']];entries[name]=executed;choices[name]=rows
        out['routes'][name]={'proposed':summary(rows),'executed':summary(executed),'skipped':summary([r for r in rows if not r['executed']]),
            'entry_clocks':dict(Counter(r['clock'] for r in executed)),'entry_symbols':dict(Counter(r['symbol'] for r in executed)),
            'account_statuses':dict(Counter(ds.decision)),'executed_gap_losses':sum(r['gap_loss'] for r in executed)}
    for c in CLOCKS:
        events=[e for e in frozen['score_sequential'] if pd.Timestamp(e['at']).tz_convert(core.ET).strftime('%H:%M:%S')==c]
        records=[read(run/'training'/(tag(e['at'])+'.json')) for e in events];ready=[r for r in records if r['band']['ready']]
        out['clock_supply'][c]={'ticks':len(events),'ready_ticks':len(ready),
            'candidates':summary([r for r in choices['score_sequential'] if r['clock']==c]),
            'executed':summary([r for r in entries['score_sequential'] if r['clock']==c]),
            'mean_margin':float(np.mean([r['estimate']['radius'] for r in ready])) if ready else None}
    full={(r['at'],r['symbol']) for r in entries['score_sequential']};pre={(r['at'],r['symbol']) for r in entries['premarket_only']}
    out['entry_set_difference']={'common':len(full&pre),'only_multiclock':len(full-pre),'only_premarket':len(pre-full),
        'not_additive_profit_opportunities':True}
    out['primary_candidates']=choices['score_sequential'];dump(run/'diagnostics.json',out)
    print({'clock_supply':out['clock_supply'],'entry_set_difference':out['entry_set_difference']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
