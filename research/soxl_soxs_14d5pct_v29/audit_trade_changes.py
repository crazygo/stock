"""Attribute changed capital paths and audit actual raw-price exits."""
import argparse,math
from regime_learning import *
from verify_continuation import key,raw_exit,near


def audit(run):
    run=Path(run);v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    meta=read(run/'input_manifest.json');market=core.load_market();assert market.manifest==meta['full_market']
    market=restrict_market(market,pd.Timestamp(meta['observation_cutoff_exclusive']));base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json')
    books={};union=set()
    for route in ROUTES:
        try:book=pd.read_csv(run/(route+'_trades.csv'))
        except pd.errors.EmptyDataError:book=pd.DataFrame()
        books[route]={(pd.Timestamp(r.decision_at),r.symbol):r for r in book.itertuples()}
        union.update(books[route])
    checks=0
    for at,s in sorted(union):
        for labs,multiple in [(base,1),(stress,2)]:raw_exit(market,labs[key(at,s)],None,multiple);checks+=1
    current=books['regime_band'];old=books['source_v27'];new=[]
    for at,s in sorted(current.keys()-old.keys()):
        lab=base[key(at,s)];entry={'at':at,'symbol':s,'exit_at':lab['exit_at'],'reason':lab['reason'],'net_return':lab['net_return']}
        if lab['reason']=='risk_stop':
            bar=market.bars[s].loc[lab['exit_at']];stop=math.floor(lab['entry_fill_economic']*.94/bar.share_factor*100+1e-10)/100
            entry.update(raw_stop=stop,raw_exit_bar_open=float(bar.open),raw_exit_bar_low=float(bar.low),
                actual_economic_fill=lab['exit_fill_economic'],gap_below_stop=bool(bar.open<stop))
            near(lab['exit_fill_economic'],min(bar.open,stop)*bar.share_factor*.9995)
        new.append(entry)
    preds=read(run/'predictions.json');decisions=pd.read_csv(run/'regime_band_decisions.csv');decisions.index=pd.to_datetime(decisions.decision_at,utc=True)
    changes=[]
    for a,b in zip(preds['regime_band'],preds['source_v27']):
        if '2026-01-01'<=a['date'][:10]<='2026-04-30' and a['action']!=b['action']:
            changes.append({'at':a['at'],'new_action':a['action'],'old_action':b['action'],
                'actual_ledger_decision':decisions.loc[pd.Timestamp(a['at']),'decision']})
    out={'passed':True,'raw_exit_checks':checks,'unique_actual_entries':len(union),'changed_candidate_actions':changes,
        'new_executed_entries':new,'removed_executed_entries':[[str(t),s] for t,s in sorted(old.keys()-current.keys())],
        'post_result_diagnostic_not_entry_filter':True,'results_sha256':sha(run/'results.json')}
    dump(run/'trade_change_audit.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
