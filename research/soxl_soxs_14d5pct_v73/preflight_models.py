"""Fixed normalized fixtures and exact reference-width nesting before capital evaluation."""
from normalized_learning import *
from audit_quadratic import reference,compare
from partial_pool import fit as old_fit,predict as old_predict
from threadpoolctl import threadpool_limits


def preflight():
    parent_sha=verify_parent();panels={k:pd.read_parquet(PARENT/(k+'_panel.parquet')) for k in PAIRS}
    market=load_market();reg=matched_registry(panels,market);fixtures=[]
    with threadpool_limits(limits=1):
        for day in ['2025-06-03','2025-11-28','2026-01-02','2026-04-30']:
            at=pd.Timestamp(day+' 09:25:30',tz=core.ET).tz_convert('UTC');parents,fb=mature_prefix(reg,at)
            for clock in CLOCKS:
                model,ids=fit_clock(panels,parents,clock);X,Y,names,w,_=training_arrays(panels,parents,clock)
                compare(model,reference(X,Y,names,w,len(ids),common=1050.,deviation=3150.))
                assert model["normalization"]==penalties(189)
                one=fit_family(X,Y,names,w,len(ids),degree=1)
                for mode in VARIANTS:
                    legacy=old_fit(X,Y,names,w,len(ids),mode=mode)
                    np.testing.assert_allclose(forecast(one,X[-3:],names[-3:],mode),old_predict(legacy,X[-3:],names[-3:]),atol=1e-10,rtol=1e-9)
                fixtures.append({'day':day,'clock':clock,'real_dates':len(ids),'columns':len(model['centers'][0]),'quadratic_modes_checked':3,'linear_augmented_modes_checked':3})
    result={'passed':True,'families':len(fixtures),'quadratic_models':len(fixtures)*3,'linear_nesting_models':len(fixtures)*3,
            'fixtures':fixtures,'parent_delivery_sha256':parent_sha,'strategy_outcomes_seen':False,'source_sha256':sha(Path(__file__))}
    dump(SOURCE/'evidence/model_preflight.json',result);print({'passed':True,'quadratic_models':len(fixtures)*3},flush=True)


if __name__=='__main__':preflight()
