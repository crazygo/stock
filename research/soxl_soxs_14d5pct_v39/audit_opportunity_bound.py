"""Post-result exhaustive subset bound: can any filter of source proposals pass?"""
import argparse,itertools
from supported import *
from cycle_ledger import replay
from run_intraday import score
from study import save_replay

def audit(run):
    run=Path(run);v=read(run/'verification.json');r=read(run/'results.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    assert r['phase']=='cases';reg=read(run/'registration.json');p=reg['preregistration']['protocol'];settings=p['cases']
    start,end=settings['evaluation_start'],settings['evaluation_end'];root=Path(read(run/'input_manifest.json')['verified_source'])
    manifest=read(root/'input_manifest.json');market=core.load_market();assert market.manifest==manifest['full_market']
    market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']));features=pd.read_parquet(root/'features.parquet');features=features[features.date.between(start,end)]
    events=read(run/'decisions_frozen.json')['routes']['source_v35'];candidates=[e for e in events if e['action'] in core.ETFS]
    if len(candidates)>16:raise ValueError('Exhaustive subset audit budget exceeded; do not select only favorable proposals')
    output=run/'opportunity_bound';output.mkdir(exist_ok=False);base=list(labels(root/'baseline_labels.json').values());records=[]
    for bits in itertools.product([0,1],repeat=len(candidates)):
        enabled={pd.Timestamp(e['at']):e['action'] for e,b in zip(candidates,bits) if b}
        actions={pd.Timestamp(e['at']):enabled.get(pd.Timestamp(e['at']),'CASH') for e in events}
        book=replay(market,features,base,actions,start,end,cost_multiple=1,allocations={t:float(a!='CASH') for t,a in actions.items()},require_full_cash=False)
        name='subset_'+''.join(map(str,bits));save_replay(output,name,book);s=score(book,settings['regression_months'])
        records.append({'mask':bits,'name':name,'stats':s})
        if all(bits):
            for suffix in ['equity','trades','orders','decisions']:pd.testing.assert_frame_equal(pd.read_csv(output/(name+'_'+suffix+'.csv')),pd.read_csv(run/('source_v35_'+suffix+'.csv')))
    nonnegative=[x for x in records if x['stats']['all_months_nonnegative']]
    receipt={'post_result_hindsight_feasibility_bound_only':True,'is_executable_learned_strategy':False,'does_not_select_or_change_policy':True,
        'source_proposals':candidates,'all_subsets':records,'subsets_evaluated':len(records),'any_subset_passes':any(x['stats']['primary_pass'] for x in records),
        'maximum_requested_month_mean':max(x['stats']['primary_average'] for x in records),
        'maximum_requested_month_mean_with_no_negative_month':max(x['stats']['primary_average'] for x in nonnegative) if nonnegative else None,
        'source_all_enabled_four_csv_exact':True,'results_sha256':sha(run/'results.json')}
    dump(run/'opportunity_bound.json',receipt);print({k:v for k,v in receipt.items() if k not in ['source_proposals','all_subsets']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
