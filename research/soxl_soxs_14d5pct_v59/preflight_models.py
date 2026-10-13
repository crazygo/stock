"""Three real predeclared dates, all new models and old controls, before capital."""
import argparse,json
from innovation_learning import *
from verify_innovation import check_base,expected_info,check_info
from audit_dense_calibration import daily_check
from run_planning import json_ready
from threadpoolctl import threadpool_limits


def preflight(inputs):
    inputs=Path(inputs);m=read(inputs/'input_manifest.json');assert read(inputs/'state_audit.json')['passed']
    source,source_sha,prior_sha=verified_source();panel=pd.read_parquet(inputs/'innovation_panel.parquet');source_panel=pd.read_parquet(inputs/'activity_panel.parquet');original=pd.read_parquet(inputs/'financial_panel.parquet')
    market=restrict_market(core.load_market(),pd.Timestamp(m['observation_cutoff_exclusive']));reg=registry(panel,market)
    history={k:{} for k in VARIANTS};chosen=['2025-12-15','2026-01-05','2026-04-01'];rows=[];previous=None
    canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel[panel.date<=chosen[-1]].iterrows()):
            selected=str(row.date.date()) in chosen
            if selected and row.date!=previous:
                daily={k:fit_day(panel,history[k],reg,at) for k in VARIANTS};references={};n=0
                for k in VARIANTS:references[k],count=daily_check(panel,market,history[k],at,canonical(daily[k]));n+=count
                rows.append({'date':str(row.date.date()),'calibration_equations':n,'base_equations':0,'old_control_refits_exact':0,'query_estimates':0})
            models,fits,raws=fit_bases(panel,at)
            for k in VARIANTS:
                if selected:
                    raw,n=check_base(panel,at,canonical(fits[k]),models[k],columns(row.clock,k));rows[-1]['base_equations']+=n
                    if raw is None:assert raws[k] is None
                    else:np.testing.assert_allclose(raw,raws[k],atol=1e-10)
                    check_info(issue(panel,at,raws[k],daily[k]),expected_info(panel,at,raw,daily[k],references[k]));rows[-1]['query_estimates']+=1
                history[k][at]=raws[k]
            if selected:
                saved=read(source/'training'/(tag(at)+'.json'))
                for table,cols,want in [(source_panel,columns(row.clock),saved['fits']['full']),(original,old55_columns(row.clock),saved['baseline']['fit'])]:
                    _,f,_=original_fit(table,at,cols);assert canonical(f)==want;rows[-1]['old_control_refits_exact']+=1
            previous=row.date
            if (i+1)%800==0:print('innovation preflight history through',at,flush=True)
    out={'passed':True,'dates':rows,'new_capital_outcomes_seen':False,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'state_audit_sha256':sha(inputs/'state_audit.json'),'source_v56_manifest_sha256':source_sha,'prior_v58_manifest_sha256':prior_sha}
    dump(SOURCE/'evidence/preflight_models.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);preflight(p.parse_args().inputs)
