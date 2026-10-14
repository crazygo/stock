"""Independent population/scaling/optimizer anchors before new capital outcomes."""
from calibration91 import *
from verify_transfer import reference_parents,canonical
import reference91 as ref


def preflight():
    (SOURCE/'evidence').mkdir(parents=True,exist_ok=True)
    parent_hashes=verify_parent();panel=pd.read_parquet(PARENT/'semiconductor_panel.parquet');market=extended_market();reg=registry(panel,market);panel=prepare_panel(panel,market)
    history={at:read(PARENT/'training'/(tag(at)+'.json')) for at in panel.index};checks=[];unready_checks=[]
    for date in ['2025-01-02','2025-07-01','2026-01-02','2026-04-30']:
        at=pd.Timestamp(date+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC');parents,fb=mature_prefix(reg,at)
        independent,indfb=reference_parents({'semiconductor':panel},market,at)
        assert canonical(parents)==canonical(independent) and canonical(fb)==canonical(indfb)
        d=daily_models(panel,parents,fb,history,at);chosen=d['parents']
        for name,(pop,mode) in MODELS.items():
            P,y,w,X,sides,clocks,meta=ref.data(panel,chosen,history,pop,CLOCKS);m=d['models'][name]
            assert canonical(meta['members'])==canonical(m['members'])
            if not d['ready']:
                assert not d['population_ready'] and d['models']['eligible_fundamental']['real_dates']<30
                assert not m['ready'] and m['reason']=='insufficient_dates'
                unready_checks.append(dict(date=date,model=name,eligible_dates=d['models']['eligible_fundamental']['real_dates'],expected_action='CASH'))
                continue
            Z,center,scale=ref.matrix(X,sides,clocks,w,mode)
            np.testing.assert_allclose(m['center'],center,atol=1e-11,rtol=1e-10);np.testing.assert_allclose(m['scale'],scale,atol=1e-11,rtol=1e-10)
            expected=ref.solve(P,y,w,Z);np.testing.assert_allclose(m['beta'],expected,atol=2e-7,rtol=1e-5)
            value,g=ref.loss_gradient(np.asarray(m['beta']).ravel(),P,y,w,Z);norm=float(max(abs(g)));assert norm<1e-7
            np.testing.assert_allclose(query(m,P[-1],X[-1],sides[-1],clocks[-1]),ref.posterior(P[-1:],Z[-1:],expected)[0],atol=1e-8,rtol=1e-7)
            checks.append(dict(date=date,model=name,dates=meta['real_dates'],date_ess=meta['date_ess'],rows=len(y),gradient_inf=norm,coefficient_max_difference=float(np.max(abs(expected-m['beta'])))))
    dump(SOURCE/'evidence/model_preflight.json',dict(passed=True,model_checks=len(checks),records=checks,unready_checks=unready_checks,parent_delivery_sha256=parent_hashes,source_sha256=sha(Path(__file__)),new_capital_outcomes_seen=False))
    print({'passed':True,'model_checks':len(checks),'unready_checks':len(unready_checks)},flush=True)


if __name__=='__main__':preflight()
