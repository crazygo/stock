"""Three real dates, independent normalized equations and exact old input reuse."""
from dense_learning import *
from audit_dense_calibration import daily_check, predict as reference_predict, input_rows
from audit_models import model_equations
from verify_reliability import close
from run_planning import json_ready
import json


def preflight():
    run,source_sha=verified_dense_source();m=read(run/'input_manifest.json')
    market=core.load_market();assert market.manifest==m['full_market']
    leaders=load_panel(market.calendar,PANEL_ROOT);assert leaders[2]==m['panel']
    _,financial=load_sources();assert financial==m['financial_panel_source']
    for d in financial.values():
        for key in ['companyfacts','submissions']:assert sha(core.ROOT/d[key]['path'])==d[key]['sha256']
    for f,h in m['derived_files_sha256'].items():assert sha(run/f)==h
    market=restrict_market(market,pd.Timestamp(m['observation_cutoff_exclusive']))
    panel=pd.read_parquet(run/'financial_panel.parquet');reg=registry(panel,market);history={};rows=[];previous=None
    chosen=['2025-11-12','2026-01-05','2026-04-01']
    for at,row in panel[panel.date<=chosen[-1]].iterrows():
        record=read(run/'training'/(tag(at)+'.json'));raw=np.array(record['raw']) if record['raw'] is not None else None
        selected=str(row.date.date()) in chosen
        if selected:
            if row.date!=previous:
                day=fit_day(panel,history,reg,at);saved=json.loads(json.dumps(json_ready(day),default=str));models,n=daily_check(panel,market,history,at,saved)
                old=read(run/'daily'/(str(row.date.date())+'.json'))
                assert {k:v for k,v in saved.items() if k!='model'}=={k:v for k,v in old.items() if k!='model'}
                rows.append({'date':str(row.date.date()),'calibration_equations':n,'mature_dates':len(day['parents']),'base_refits_exact':0,'base_equations':0,'query_estimates':0})
            model,fit,recomputed=fit_base(panel,at);assert json.loads(json.dumps(json_ready(fit),default=str))==record['fit'];close(recomputed,raw)
            ids=pd.to_datetime(fit['fit']['members'],utc=True);want,n=model_equations(panel[panel.clock==row.clock],ids,at,model,fit['fit'],columns(row.clock,'full'))
            for key,value in want.items():close(fit['raw'][key],value)
            rows[-1]['base_equations']+=n;rows[-1]['base_refits_exact']+=1
            info=issue(panel,at,raw,day);x=input_rows(panel,{at:raw},[at],CLOCKS);point=raw+reference_predict(models['point_model'],x)[0]
            draws=np.array([raw+reference_predict(z,x)[0] for z in models['bootstrap_models']]);radius=max(0.,sorted(np.max(draws-point,axis=1))[179])
            close(info['point'],point);close(info['lower'],point-radius);rows[-1]['query_estimates']+=1
        previous=row.date;history[at]=raw
    out={'passed':True,'dates':rows,'source_inputs_and_raw_files_unchanged':True,'source_daily_members_and_weights_exact':True,
         'dense_source_manifest_sha256':source_sha,'input_manifest_sha256':sha(run/'input_manifest.json'),'input_audit_sha256':sha(run/'input_audit.json'),
         'new_capital_outcomes_seen':False}
    dump(SOURCE/'evidence/preflight_models.json',out);print(out,flush=True)

if __name__=='__main__':preflight()
