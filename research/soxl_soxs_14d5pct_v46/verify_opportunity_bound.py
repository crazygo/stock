"""Independently reconcile every exhaustive subset ledger and monthly bound."""
import argparse,itertools
from fixed_candidate import *
from audit_books import book_check
from verify_continuation import near

def verify(run):
    run=Path(run);data=read(run/'opportunity_bound.json');assert data['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];root=Path(read(run/'input_manifest.json')['source'])
    s=p['cases'];manifest=read(root/'input_manifest.json');market=core.load_market();assert market.manifest==manifest['full_market']
    market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']));features=pd.read_parquet(root/'features.parquet')
    features=features[features.date.between(s['evaluation_start'],s['evaluation_end'])];events=read(run/'decisions_frozen.json')['routes']['strong_band']
    candidates=[e for e in events if e['action'] in core.ETFS];assert data['source_proposals']==candidates
    masks={tuple(x['mask']) for x in data['all_subsets']};assert len(masks)==len(data['all_subsets'])==2**len(candidates)
    assert masks==set(itertools.product([0,1],repeat=len(candidates)))
    books=[];means=[];nonnegative=[];passed=[];lab=labels(root/'baseline_labels.json');out=run/'opportunity_bound'
    for row in data['all_subsets']:
        include={e['at'] for e,b in zip(candidates,row['mask']) if b};actions=[{**e,'action':e['action'] if e['at'] in include else 'CASH','allocation':float(e['at'] in include)} for e in events]
        books.append(book_check(out,row['name'],market,features,lab,actions,1,row['stats'],s['regression_months']))
        eq=pd.read_csv(out/(row['name']+'_equity.csv'),parse_dates=['date']);rates={};prior=1
        for month,g in eq.groupby(eq.date.dt.to_period('M')):rates[str(month)]=g.equity.iloc[-1]/prior-1;prior=g.equity.iloc[-1]
        average=float(np.mean([rates[m] for m in s['regression_months']]));means.append(average)
        if all(x>=-1e-12 for x in rates.values()):nonnegative.append(average);passed.append(average>=.1)
    near(data['maximum_requested_month_mean'],max(means));near(data['maximum_requested_month_mean_with_no_negative_month'],max(nonnegative))
    assert data['any_subset_passes']==any(passed)
    receipt={'passed':True,'independent_capital_books':books,'all_subsets_accounted':True,'maximum_requested_mean':max(means),
        'maximum_requested_mean_with_all_months_nonnegative':max(nonnegative),'no_filter_subset_can_meet_target_on_this_fixed_source':not any(passed),
        'hindsight_bound_not_an_executable_policy':True,'bound_sha256':sha(run/'opportunity_bound.json'),'results_sha256':sha(run/'results.json')}
    dump(run/'opportunity_bound_verification.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
