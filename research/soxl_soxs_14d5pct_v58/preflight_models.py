"""Three real dates, independent joint equations and calibration before capital."""
import argparse,json
from shared_learning import *
from audit_base import check_day,predict_ref
from verify_shared import expected_info,check_info
from audit_dense_calibration import daily_check
from run_planning import json_ready
from threadpoolctl import threadpool_limits


def preflight(inputs):
    inputs=Path(inputs);m=read(inputs/'input_manifest.json');assert read(inputs/'input_audit.json')['passed']
    source,source_sha,prior_sha=verified_source();panel=pd.read_parquet(inputs/'activity_panel.parquet');original=pd.read_parquet(inputs/'financial_panel.parquet')
    market=restrict_market(core.load_market(),pd.Timestamp(m['observation_cutoff_exclusive']));reg=registry(panel,market)
    history={k:{} for k in VARIANTS};chosen=['2025-11-12','2026-01-05','2026-04-01'];rows=[];previous=None
    canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel[panel.date<=chosen[-1]].iterrows()):
            selected=str(row.date.date()) in chosen
            if row.date!=previous:
                base=fit_base_day(panel,reg,at)
                if selected:
                    base_n=check_day(panel,market,at,canonical(base))
                    daily={k:fit_day(panel,history[k],reg,at) for k in VARIANTS};references={};n=0
                    for k in VARIANTS:references[k],count=daily_check(panel,market,history[k],at,canonical(daily[k]));n+=count
                    rows.append({'date':str(row.date.date()),'calibration_equations':n,'base_equations':base_n,'old_control_refits_exact':0,'query_estimates':0,'mature_source_dates':base['prepared']['N']})
            for k in VARIANTS:
                detail,raw=predict_base(panel,at,base,k)
                if selected:
                    want,pair=predict_ref(panel,at,canonical(base),k)
                    np.testing.assert_allclose(raw,pair,atol=1e-10)
                    check_info(issue(panel,at,raw,daily[k]),expected_info(panel,at,pair,daily[k],references[k]));rows[-1]['query_estimates']+=1
                history[k][at]=raw
            if selected:
                saved=read(source/'training'/(tag(at)+'.json'))
                for table,cols,want in [(panel,columns(row.clock),saved['fits']['full']),(original,old55_columns(row.clock),saved['baseline']['fit'])]:
                    _,f,_=original_fit(table,at,cols);assert canonical(f)==want;rows[-1]['old_control_refits_exact']+=1
            previous=row.date
            if (i+1)%800==0:print('joint preflight history through',at,flush=True)
    out={'passed':True,'dates':rows,'new_capital_outcomes_seen':False,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'input_audit_sha256':sha(inputs/'input_audit.json'),'source_v56_manifest_sha256':source_sha,'prior_v57_manifest_sha256':prior_sha}
    dump(SOURCE/'evidence/preflight_models.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);preflight(p.parse_args().inputs)
