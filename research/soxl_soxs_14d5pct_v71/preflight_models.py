"""Fixed real-data numerical fixtures, before strategy/capital output."""
from transfer_learning import *
from verify_transfer import reference_models
from threadpoolctl import threadpool_limits


def preflight():
    inputs=ROOT/'.cache/soxl_soxs_14d5pct_v71/inputs/development'
    assert read(inputs/'input_audit.json')['passed']
    panels={k:pd.read_parquet(inputs/(k+'_panel.parquet')) for k in PAIRS}
    market=load_market();reg=matched_registry(panels,market);checks=0;fixtures=[]
    with threadpool_limits(limits=1):
        for d in ['2025-06-03','2025-11-28','2026-01-02','2026-04-30']:
            at=pd.Timestamp(d+' 09:25:30',tz=core.ET).tz_convert('UTC');parents,fb=mature_prefix(reg,at)
            for clock in CLOCKS:
                models,ids=fit_clock(panels,parents,clock);refs=reference_models(panels,ids) if models else {}
                for k,ref in refs.items():
                    for f in ['centers','scales','intercepts','coefficients','weights','penalty']:np.testing.assert_allclose(models[k][f],ref[f],rtol=1e-9,atol=1e-10)
                    checks+=1
                fixtures.append({'date':d,'clock':clock,'real_dates':len(ids),'models':len(models),'last_member':ids[-1] if ids else None})
    receipt={'passed':True,'models_normal_equations':checks,'independent_equals_solo':True,'fixtures':fixtures,
             'input_manifest_sha256':sha(inputs/'input_manifest.json'),'strategy_capital_outcomes_seen':False,
             'source_sha256':sha(Path(__file__))}
    dump(SOURCE/'evidence/model_preflight.json',receipt);print({'passed':True,'models':checks},flush=True)


if __name__=='__main__':preflight()
