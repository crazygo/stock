"""Independent paired sample, optimizer and projection checks before capital."""
from paired93 import *
from verify_transfer import reference_parents,canonical
import reference93 as ref
from reference92 import certificate,solve as project_reference


def preflight():
    (SOURCE/'evidence').mkdir(parents=True,exist_ok=True);parent_hashes=verify_parent()
    market=extended_market();panel=pd.read_parquet(PARENT/'semiconductor_panel.parquet');reg=registry(panel,market);panel=prepare_panel(panel,market)
    history={at:read(PARENT/'training'/(tag(at)+'.json')) for at in panel.index};checks=[];unready=[];projections=0;maxdiff=0.
    for date in ['2024-07-01','2025-01-02','2025-07-01','2025-12-01','2026-02-13','2026-04-30']:
        at=pd.Timestamp(date+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC');parents,fb=mature_prefix(reg,at)
        other,otherfb=reference_parents({'semiconductor':panel},market,at)
        assert canonical(parents)==canonical(other) and canonical(fb)==canonical(otherfb)
        prior=read(PREVIOUS/'daily'/(date+'.json'))['base'];d=daily_models(panel,parents,fb,history,at,prior['ready'])
        assert canonical(d['parents'])==prior['parents'];arrays=ref.data(panel,d['parents'],history,CLOCKS);P,y,w,X,sides,clocks,meta=arrays
        production=paired_samples(panel,d['parents'],history)
        for a,b in zip(production[:6],arrays[:6]):np.testing.assert_allclose(a,b,atol=1e-11,rtol=1e-10)
        assert canonical(meta['members'])==canonical(production[-1]['members'])
        for mode in PAIRED_MODELS:
            m=d['models'][mode]
            if not m['ready']:
                unready.append(dict(date=date,mode=mode,reason=m['reason'],eligible_dates=d['eligible_dates']));continue
            Z,mu,sd=ref.matrix(X,sides,clocks,w,mode);B=ref.solve(P,y,w,Z)
            np.testing.assert_allclose(m['beta'],B,atol=2e-7,rtol=1e-5)
            np.testing.assert_allclose(m['center'],mu,atol=1e-11,rtol=1e-10);np.testing.assert_allclose(m['scale'],sd,atol=1e-11,rtol=1e-10)
            _,g=ref.loss_gradient(np.asarray(m['beta']).ravel(),P,y,w,Z);assert max(abs(g))<1e-7
            checks.append(dict(date=date,mode=mode,rows=len(y),dates=meta['real_dates'],date_ess=meta['date_ess'],gradient_inf=float(max(abs(g))),coefficient_max_difference=float(np.max(abs(B-m['beta'])))))
        if d['ready']:
            for t,r in panel[panel.date==panel.loc[at,'date']].iterrows():
                src=history[t]['events']['recent_ratio'];base=read(BASE/'training'/(tag(t)+'.json'))
                estimates_now=estimates(r,d,src,base['distributions']['gap']['payoff_means'])
                for mode in PAIRED_MODELS:
                    for side,proof in enumerate(estimates_now[mode]['projections']):
                        observed=d['supports'][mode][f'{side}:{CLOCKS.index(r.clock)}']['corrections'];certificate(proof['raw'],observed,proof);projections+=1
                        # One premarket anchor per fit/side, not a strategy selection.
                        if r.clock==CLOCKS[0]:
                            independent=project_reference(proof['raw'],observed);diff=float(max(abs(independent-proof['correction'])));maxdiff=max(maxdiff,diff)
                            np.testing.assert_allclose(independent,proof['correction'],atol=2e-6,rtol=2e-5)
    out=dict(passed=True,optimizer_checks=len(checks),records=checks,unready_checks=unready,projection_certificates=projections,
        maximum_projection_anchor_difference=maxdiff,parent_delivery_sha256=parent_hashes,source_sha256=sha(Path(__file__)),new_capital_outcomes_seen=False)
    dump(SOURCE/'evidence/model_preflight.json',out);print(out,flush=True)


if __name__=='__main__':preflight()
