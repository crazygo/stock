"""Real prequential new-feature histories; three dates independently refit/audit."""
import argparse,json
from activity_learning import *
from audit_dense_calibration import daily_check,predict as ref_predict,input_rows
from audit_models import model_equations
from verify_reliability import close
from run_planning import json_ready
from threadpoolctl import threadpool_limits


def preflight(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json');assert read(inputs/'activity_audit.json')['passed'];source,source_sha=verified_source()
    panel=pd.read_parquet(inputs/'activity_panel.parquet');original=pd.read_parquet(inputs/'financial_panel.parquet');market=restrict_market(core.load_market(),pd.Timestamp(manifest['observation_cutoff_exclusive']));reg=registry(panel,market)
    history={k:{} for k in VARIANTS};chosen=['2025-11-12','2026-01-05','2026-04-01'];rows=[];previous=None
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel[panel.date<=chosen[-1]].iterrows()):
            selected=str(row.date.date()) in chosen
            if selected and row.date!=previous:
                daily={k:fit_day(panel,history[k],reg,at) for k in VARIANTS};references={};n=0
                for k in VARIANTS:
                    references[k],count=daily_check(panel,market,history[k],at,json.loads(json.dumps(json_ready(daily[k]),default=str)));n+=count
                rows.append({'date':str(row.date.date()),'calibration_equations':n,'base_equations':0,'base_models':0,'baseline_refits_exact':0,'query_estimates':0})
            models,fits,raws=fit_bases(panel,at)
            if selected:
                for k in VARIANTS:
                    ids=pd.to_datetime(fits[k]['fit']['members'],utc=True);want,n=model_equations(panel[panel.clock==row.clock],ids,at,models[k],fits[k]['fit'],columns(row.clock,k))
                    for key,value in want.items():close(fits[k]['raw'][key],value)
                    rows[-1]['base_equations']+=n;rows[-1]['base_models']+=1
                    info=issue(panel,at,raws[k],daily[k]);x=input_rows(panel,{at:raws[k]},[at],CLOCKS);point=raws[k]+ref_predict(references[k]['point_model'],x)[0];draws=np.array([raws[k]+ref_predict(z,x)[0] for z in references[k]['bootstrap_models']]);radius=max(0.,sorted(np.max(draws-point,axis=1))[179]);close(info['point'],point);close(info['lower'],point-radius);rows[-1]['query_estimates']+=1
                _,base_fit,base_raw=prior.fit_base(original,at);saved=read(source/'training'/(tag(at)+'.json'));assert json.loads(json.dumps(json_ready(base_fit),default=str))==saved['fit'];close(base_raw,saved['raw']);rows[-1]['baseline_refits_exact']+=1
            for k in VARIANTS:history[k][at]=raws[k]
            previous=row.date
            if (i+1)%800==0:print('preflight history through',at,flush=True)
    receipt={'passed':True,'dates':rows,'new_capital_outcomes_seen':False,'input_manifest_sha256':sha(inputs/'input_manifest.json'),'activity_audit_sha256':sha(inputs/'activity_audit.json'),'source_v55_manifest_sha256':source_sha}
    dump(SOURCE/'evidence/preflight_models.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);preflight(p.parse_args().inputs)
