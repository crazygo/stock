"""Real-data weight-mass and feature-mask preflight before new capital outcomes."""
import argparse
from threadpoolctl import threadpool_limits
from history_learning import *
from reference_fit import from_panels
from verify_transfer import reference_parents,canonical


def preflight(inputs,output):
    inputs=Path(inputs);market=extended_market()
    panels={k:pd.read_parquet(inputs/(k+'_panel.parquet')) for k in PAIRS}
    shorts={k:pd.read_parquet(inputs/(k+'_short_panel.parquet')) for k in PAIRS}
    reg=matched_registry(panels,market);shortreg=matched_registry(shorts,market)
    checks=0;records=[]
    with threadpool_limits(limits=1):
        for date in ['2024-01-02','2025-06-03','2026-01-02','2026-04-30']:
            at=pd.Timestamp(date+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC')
            parents,feedback=mature_prefix(reg,at);sp,sf=mature_prefix(shortreg,at)
            rp,rf=reference_parents(panels,market,at);rsp,rsf=reference_parents(shorts,market,at)
            assert canonical(parents)==canonical(rp) and canonical(feedback)==canonical(rf)
            assert canonical(sp)==canonical(rsp) and canonical(sf)==canonical(rsf)
            for c in CLOCKS:
                models,members=clock_models(panels,shorts,parents,sp,c)
                reference=max(120,len(members['short_price']))
                for k,model in models.items():
                    ids=members[k];ref=from_panels(shorts if k=='short_price' else panels,ids,reference,k,FIELDS)
                    fields=['weights','mean_cd'] if k=='long_mean' else ['centers','scales','intercepts','coefficients','penalty','weights']
                    for f in fields:np.testing.assert_allclose(model[f],ref[f],atol=1e-10,rtol=1e-9)
                    assert model['distinct_dates']==len(ids)
                    checks+=1;records.append({'date':date,'clock':c,'variant':k,'dates':len(ids),
                        'reference_mass':reference,'date_ess':model['date_effective_sample_size']})
                ids=members['short_price']
                if len(ids)>=120:
                    n=len(ids);w=2.**(-np.arange(n)[::-1]/126.);w*=n/w.sum()
                    X=np.concatenate([p.loc[ids,FIELDS].to_numpy(float) for p in shorts.values()])
                    Y=np.concatenate([p.loc[ids,['C','D']].to_numpy(float) for p in shorts.values()])
                    names=np.repeat(list(shorts),n);weights=np.tile(w/3,3)
                    got=fit_state(X,Y,names,weights,n,n,'long_full');want=old.fit(X,Y,names,weights,n,mode='partial',common=100,deviation=300)
                    for f in ['centers','scales','intercepts','coefficients','penalty','weights']:np.testing.assert_array_equal(got[f],want[f])
    result={'passed':True,'real_data_fits_checked':checks,'records':records,'short_full_equal_mass_exact':True,
            'source_sha256':sha(Path(__file__)),'input_manifest_sha256':sha(inputs/'input_manifest.json'),
            'strategy_capital_outcomes_seen':False}
    dump(output,result);print({'passed':True,'real_data_fits_checked':checks},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();preflight(a.inputs,a.output)
