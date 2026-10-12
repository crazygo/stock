"""Explain changed cost-scenario paths from raw bars, without altering decisions."""
import argparse,math
from pathlib import Path
from clock_reliability import core,read,labels,dump,sha,pd,restrict_market
from verify_continuation import key,raw_exit


def audit(run):
    run=Path(run);v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    m=read(run/'input_manifest.json');market=core.load_market();assert market.manifest==m['full_market']
    market=restrict_market(market,pd.Timestamp(m['observation_cutoff_exclusive']))
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json')
    original=pd.read_csv(run/'score_sequential_trades.csv');other=pd.read_csv(run/'double_cost_trades.csv')
    a={(pd.Timestamp(r.decision_at),r.symbol) for r in original.itertuples()};b={(pd.Timestamp(r.decision_at),r.symbol) for r in other.itertuples()}
    changed=[];checks=0
    for at,symbol in sorted(a|b):
        lo,hi=base[key(at,symbol)],stress[key(at,symbol)]
        for label,multiple in [(lo,1),(hi,2)]:raw_exit(market,label,None,multiple);checks+=1
        if lo['exit_at']!=hi['exit_at'] or lo['reason']!=hi['reason']:
            r={'at':at,'symbol':symbol,'in_baseline_book':(at,symbol) in a,'in_stress_book':(at,symbol) in b,
               'baseline_exit':lo['exit_at'],'stress_exit':hi['exit_at'],'baseline_reason':lo['reason'],'stress_reason':hi['reason'],
               'baseline_net':lo.get('net_return'),'stress_net':hi.get('net_return')}
            if hi['reason']=='risk_stop':
                bar=market.bars[symbol].loc[hi['exit_at']];factor=bar.share_factor
                r.update(stress_stop_bar_low=float(bar.low),baseline_stop_raw=math.floor(lo['entry_fill_economic']*.94/factor*100+1e-10)/100,
                    stress_stop_raw=math.floor(hi['entry_fill_economic']*.94/factor*100+1e-10)/100)
            changed.append(r)
    out={'passed':True,'raw_exit_checks':checks,'actual_entry_union':len(a|b),'common_entries':len(a&b),
        'only_baseline':[[str(t),s] for t,s in sorted(a-b)],'only_stress':[[str(t),s] for t,s in sorted(b-a)],
        'changed_terminal_paths':changed,'results_sha256':sha(run/'results.json'),'frozen_decisions_unchanged':True,
        'post_result_explanation_not_future_filter':True}
    dump(run/'cost_path_audit.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
