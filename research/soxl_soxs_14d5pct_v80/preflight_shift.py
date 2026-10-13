from shift_learning import *
from verify_transfer import reference_parents,canonical
import reference_shift as ref
import reference_tilt as cal


def preflight():
    verify_parent();market=extended_market();panel=pd.read_parquet(PARENT/'semiconductor_panel.parquet');reg=registry(panel,market)
    full={at:read(BASE/'training'/(tag(at)+'.json')) for at in panel.index};checks=[]
    for day in ['2025-01-02','2026-01-02','2026-04-30']:
        at=pd.Timestamp(day+' 09:25:30',tz=core.ET).tz_convert('UTC');history={t:r for t,r in full.items() if t<day_cutoff(at)}
        old=read(PARENT/'daily'/(day+'.json'));out=daily_models(panel,reg,market,history,old,at)
        windows=ref.windows(panel,market.sessions,market.calendar,history,at,CLOCKS)
        parents,fb=reference_parents({'semiconductor':panel},market,at)
        chosen=[p for p in parents if all(t in history and history[t]['events']['gap_blend']['ready'] for t in p['members'])][-120:]
        assert canonical(chosen)==old['parents'];P,y,w,meta=cal.data(panel,chosen,history,'top');X,keys,_=ref.contexts(panel,chosen,history,FIELDS)
        for mode in VARIANTS:
            got=out['variants'][mode];assert got['ready'] and canonical(got['window'])==canonical(windows[mode])
            Z,targetkeys,v=ref.contexts(panel,windows[mode]['parents'],history,FIELDS);A,dy,dw,center,scale=ref.design(X,Z,w,v)
            domain=got['domain'];beta=ref.solve(A,dy,dw);np.testing.assert_allclose(domain['beta'],beta,atol=1e-7,rtol=1e-6)
            np.testing.assert_allclose(domain['center'],center,atol=1e-12);np.testing.assert_allclose(domain['scale'],scale,atol=1e-12)
            logratios=A[:len(X)]@domain['beta'];ratios=np.minimum(5,np.maximum(.2,np.exp(logratios)))
            adjusted,ess=ref.weights(w,ratios,keys);np.testing.assert_allclose(adjusted,got['weights'],atol=1e-12,rtol=1e-11)
            cbeta,optimizer=ref.solve_calibrator(P,y,adjusted)
            np.testing.assert_allclose(cbeta,got['model']['beta'],atol=1e-7,rtol=1e-6)
            norm=float(max(abs(ref.objective(domain['beta'],A,dy,dw)[1])));assert norm<1e-7
            checks.append({'day':day,'mode':mode,'rows':len(y),'target_rows':len(Z),'weighted_date_ess':ess,
                'domain_gradient_inf':norm,'coefficient_max_difference':float(max(abs(np.asarray(domain['beta'])-beta))),
                'independent_class_optimizer':optimizer})
    dump(SOURCE/'evidence/model_preflight.json',{'passed':True,'domain_and_calibrator_checks':len(checks),'records':checks,
        'parent_delivery_sha256':sha(ROOT/'research/soxl_soxs_14d5pct_v79/evidence/delivery_manifest.json'),
        'source_sha256':sha(Path(__file__)),'new_capital_outcomes_seen':False});print({'passed':True,'anchors':len(checks)},flush=True)


if __name__=='__main__':preflight()
