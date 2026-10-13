import argparse
from tilt_learning import *
from verify_transfer import reference_parents,canonical
import reference_tilt as ref


def preflight():
    verify_parent();market=extended_market();panel=pd.read_parquet(PARENT/'semiconductor_panel.parquet')
    history={at:read(PARENT/'training'/(tag(at)+'.json')) for at in panel.index};checks=[]
    for day in ['2025-01-02','2026-01-02','2026-04-30']:
        at=pd.Timestamp(day+' 09:25:30',tz=core.ET).tz_convert('UTC');reg=registry(panel,market);parents,fb=mature_prefix(reg,at)
        alt,afb=reference_parents({'semiconductor':panel},market,at);assert canonical(parents)==canonical(alt) and canonical(fb)==canonical(afb)
        model=daily_models(panel,parents,fb,history,at);assert model['ready']
        for mode in MODES:
            P,y,w,meta=ref.data(panel,model['parents'],history,mode);got=model['models'][mode]
            for k in ['dates','real_dates','members']:assert canonical(got[k])==canonical(meta[k])
            np.testing.assert_allclose(got['weights'],w,atol=1e-12,rtol=1e-12)
            beta=ref.solve(P,y,w);np.testing.assert_allclose(got['beta'],beta,atol=1e-7,rtol=1e-6)
            loss,g=ref.objective(got['beta'],P,y,w);assert max(abs(g))<1e-7
            checks.append({'day':day,'mode':mode,'rows':len(y),'dates':got['real_dates'],'date_ess':meta['date_ess'],'gradient_inf':float(max(abs(g))),
                'coefficient_max_difference':float(max(abs(np.array(got['beta'])-beta)))})
    dump(SOURCE/'evidence/model_preflight.json',{'passed':True,'model_checks':len(checks),'records':checks,
        'parent_delivery_sha256':sha(ROOT/'research/soxl_soxs_14d5pct_v78/evidence/delivery_manifest.json'),
        'source_sha256':sha(Path(__file__)),'new_capital_outcomes_seen':False});print({'passed':True,'model_checks':len(checks)},flush=True)


if __name__=='__main__':preflight()
