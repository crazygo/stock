from wait_learning import *
from verify_transfer import canonical
import reference_wait as ref


def preflight():
    verify_parent();panel=pd.read_parquet(PARENT/'semiconductor_panel.parquet');market=extended_market()
    records={at:read(PARENT/'training'/(tag(at)+'.json')) for at in panel.index};bank=target_registry(panel,market,records)
    independent=ref.bank(panel,market,records,CLOCKS);assert canonical(bank)==canonical(independent);checks=[]
    for date in ['2025-01-02','2026-01-02','2026-04-30']:
        at=pd.Timestamp(date+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC');d=daily_models(panel,bank,records,at);assert d['ready']
        parents,fb=ref.parents(independent,day_cutoff(at));assert canonical(parents)==canonical(d['parents']) and canonical(fb)==canonical(d['feedback'])
        X,Y,c,w,meta=ref.data(panel,parents,records,FIELDS,CLOCKS);assert canonical(meta['members'])==canonical(d['metadata']['members'])
        for mode in VARIANTS:
            expected,A=ref.solve(X,Y,c,w,mode);got=d['models'][mode]
            for key in ['center','scale','design_center','intercept','coefficients','penalty','weights']:np.testing.assert_allclose(got[key],expected[key],atol=1e-11,rtol=1e-10)
            coef=np.asarray(got['coefficients']);Z=A-got['design_center'];res=Y-got['intercept']-Z@coef
            g=Z.T@(w[:,None]*res)-np.asarray(got['penalty'])[:,None]*coef;norm=float(max(abs(g).flat)) if g.size else 0.
            assert norm<1e-9
            checks.append({'date':date,'mode':mode,'real_dates':len(parents),'rows':len(Y),'date_ess':meta['date_ess'],'normal_equation_residual':norm,
                'negative_wait_rows':int((Y[:,1]<0).sum()),'cash_wait_rows':sum(m['wait_action']=='CASH' for m in meta['members'])})
    dump(SOURCE/'evidence/model_preflight.json',{'passed':True,'model_checks':len(checks),'pair_registry_dates':len(bank),'records':checks,
        'parent_delivery_sha256':sha(ROOT/'research/soxl_soxs_14d5pct_v80/evidence/delivery_manifest.json'),
        'source_sha256':sha(Path(__file__)),'new_capital_outcomes_seen':False});print({'passed':True,'checks':len(checks),'pair_dates':len(bank)},flush=True)


if __name__=='__main__':preflight()
