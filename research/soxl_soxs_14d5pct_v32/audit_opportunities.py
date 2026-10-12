"""Separate available signals, blocked signals and changed raw trade paths."""
from hierarchical_error import *
from verify_continuation import key,near,raw_exit


def audit(run):
    run=Path(run);result=read(run/'results.json');assert read(run/'verification.json')['passed']
    meta=read(run/'input_manifest.json');market=core.load_market();assert market.manifest==meta['full_market']
    market=restrict_market(market,pd.Timestamp(meta['observation_cutoff_exclusive']));base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json')
    books={};union=set()
    for n in ROUTES:
        try:t=pd.read_csv(run/(n+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        books[n]={(pd.Timestamp(r.decision_at),r.symbol):r for r in t.itertuples()};union.update(books[n])
    for at,s in sorted(union):
        for lab,multiple in [(base,1),(stress,2)]:raw_exit(market,lab[key(at,s)],None,multiple)
    ds=pd.read_csv(run/'partial_band_decisions.csv');ds.index=pd.to_datetime(ds.decision_at,utc=True)
    proposals=ds[ds.requested_action.isin(core.ETFS)];blocked=proposals[~proposals.decision.str.startswith('buy_')]
    monthly=[]
    for month,frame in ds.groupby(ds.index.tz_convert(core.ET).tz_localize(None).to_period('M')):
        active=frame[frame.requested_action.isin(core.ETFS)];actual=frame[frame.decision.str.startswith('buy_')]
        monthly.append({'month':str(month),'clock_ticks':len(frame),'proposed_buys':len(active),
            'proposed_dates':len(set(active.index.tz_convert(core.ET).date)),
            'actual_buys':len(actual),'blocked_proposals':active[~active.decision.str.startswith('buy_')].decision.value_counts().to_dict()})
    def describe(item):
        at,s=item;a=base[key(at,s)];b=stress[key(at,s)]
        return {'at':at,'symbol':s,'entry_at':a['entry_at'],'baseline_exit':a['exit_at'],'baseline_reason':a['reason'],'baseline_net':a['net_return'],
            'double_cost_exit':b['exit_at'],'double_cost_reason':b['reason'],'double_cost_net':b['net_return']}
    changed={}
    for control in ['independent_band','source_v27']:
        new=set(books['partial_band'])-set(books[control]);removed=set(books[control])-set(books['partial_band'])
        changed[control]={'new_actual_entries':[describe(x) for x in sorted(new)],'removed_actual_entries':[describe(x) for x in sorted(removed)]}
    out={'passed':True,'post_result_only':True,'unique_actual_entry_paths':len(union),'raw_exit_checks':len(union)*2,
        'primary_proposed_buys':len(proposals),'primary_actual_buys':len(books['partial_band']),
        'primary_blocked_proposals':blocked.decision.value_counts().to_dict(),'monthly_opportunities':monthly,
        'changed_actual_entries':changed,'cost_counterfactuals_are_not_new_capital_backtests':True,
        'not_hindsight_entry_filter':True,'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'opportunity_audit.json',out);print({k:v for k,v in out.items() if k!='changed_actual_entries'},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
