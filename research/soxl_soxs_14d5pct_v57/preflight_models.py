"""Three real dates with independent members/equations, before new capital."""
import argparse,json
from short_learning import *
from verify_short import check_base,reference_panels,expected_info,check_info,reference_gap
from audit_dense_calibration import daily_check
from run_planning import json_ready
from threadpoolctl import threadpool_limits


def preflight(inputs):
    inputs=Path(inputs);m=read(inputs/'input_manifest.json');assert read(inputs/'short_input_audit.json')['passed']
    source,source_sha=verified_source();panel=pd.read_parquet(inputs/'activity_panel.parquet');short=pd.read_parquet(inputs/'short_panel.parquet');original=pd.read_parquet(inputs/'financial_panel.parquet')
    panels=training_panels(panel,short);independent=reference_panels(panel,short)
    for k in VARIANTS:pd.testing.assert_frame_equal(panels[k],independent[k])
    market=restrict_market(core.load_market(),pd.Timestamp(m['observation_cutoff_exclusive']));reg=registry(panel,market)
    history={k:{} for k in VARIANTS};chosen=['2025-11-12','2026-01-05','2026-04-01'];rows=[];previous=None
    canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel[panel.date<=chosen[-1]].iterrows()):
            selected=str(row.date.date()) in chosen
            if selected and row.date!=previous:
                daily={k:fit_day(panel,history[k],reg,at) for k in VARIANTS};references={};n=0
                for k in VARIANTS:references[k],count=daily_check(panel,market,history[k],at,canonical(daily[k]));n+=count
                rows.append({'date':str(row.date.date()),'original14day_calibration_equations':n,'short_base_equations':0,'short_base_models':0,'old_control_refits_exact':0,'query_estimates':0,'fast_minus_lagged_members':[]})
            models,fits,raws=fit_bases(panels,at)
            if selected:
                gap=reference_gap(short[short.clock==row.clock],at)
                for k in VARIANTS:
                    raw,n=check_base(independent[k],at,canonical(fits[k]),models[k],columns(row.clock));assert fits[k]['short_unresolved_expired']==gap
                    if raw is None:assert raws[k] is None
                    else:np.testing.assert_allclose(raw,raws[k],atol=1e-10)
                    rows[-1]['short_base_equations']+=n;rows[-1]['short_base_models']+=1
                    info=issue(panel,at,raws[k],daily[k]);check_info(info,expected_info(panel,at,raw,daily[k],references[k]));rows[-1]['query_estimates']+=1
                saved=read(source/'training'/(tag(at)+'.json'))
                for k,table,cols,want in [('v56',panel,columns(row.clock),saved['fits']['full']),('v55',original,prior.prior.columns(row.clock,'full'),saved['baseline']['fit'])]:
                    _,f,_=original_fit(table,at,cols);assert canonical(f)==want;rows[-1]['old_control_refits_exact']+=1
                rows[-1]['fast_minus_lagged_members'].append(len(fits['fast']['fit']['members'])-len(fits['lagged']['fit']['members']))
            for k in VARIANTS:history[k][at]=raws[k]
            previous=row.date
            if (i+1)%800==0:print('short preflight history through',at,flush=True)
    out={'passed':True,'dates':rows,'new_capital_outcomes_seen':False,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'short_input_audit_sha256':sha(inputs/'short_input_audit.json'),'source_v56_manifest_sha256':source_sha}
    dump(SOURCE/'evidence/preflight_models.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);preflight(p.parse_args().inputs)
