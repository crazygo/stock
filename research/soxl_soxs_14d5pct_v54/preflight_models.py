"""Real as-issued new-clock history, three-day equation checks without capital."""
import argparse
from dense_learning import *
from audit_models import model_equations
from audit_dense_calibration import daily_check,predict as reference_predict,input_rows
from verify_reliability import close
from run_planning import json_ready
import json


def preflight(inputs):
    inputs=Path(inputs);manifest=read(inputs/'input_manifest.json');panel=pd.read_parquet(inputs/'financial_panel.parquet');features=pd.read_parquet(inputs/'features.parquet')
    market=restrict_market(core.load_market(),pd.Timestamp(manifest['observation_cutoff_exclusive']));source=Path(manifest['source_run']);reg=registry(panel,market);h={};rows=[];current=None;d=None;models=None
    chosen=['2025-11-12','2026-01-05','2026-04-01'];heads=0
    for at,row in features[features.date<=chosen[-1]].iterrows():
        day=str(row.date.date());selected=day in chosen
        if day!=current:
            current=day
            if selected:
                d=fit_day(panel,h,reg,at);assert d['ready'];saved=json.loads(json.dumps(json_ready(d),default=str));models,n=daily_check(panel,market,h,at,saved)
                rows.append({'date':day,'mature_dates':len(d['parents']),'calibration_equations':n,'new_base_models':0,'old_base_refits_exact':0,'base_equations':0,'query_estimates':0,'capital_outcomes_seen':False})
        if row.clock in OLD_CLOCKS:
            old=read(source/'training'/(tag(at)+'.json'));raw=old['raws']['full'];raw=np.array(raw) if raw is not None else None
            if selected:
                model,fit,recomputed=fit_base(panel,at);assert json.loads(json.dumps(json_ready(fit),default=str))==old['fits']['full'];close(recomputed,raw);rows[-1]['old_base_refits_exact']+=1
        else:
            model,fit,raw=fit_base(panel,at)
            if selected:
                ids=pd.to_datetime(fit['fit']['members'],utc=True);want,n=model_equations(panel[panel.clock==row.clock],ids,at,model,fit['fit'],columns(row.clock,'full'))
                for key,value in want.items():close(fit['raw'][key],value)
                rows[-1]['base_equations']+=n;rows[-1]['new_base_models']+=1
        if selected:
            info=issue(panel,at,raw,d);x=input_rows(panel,{at:raw},[at],CLOCKS);point=raw+reference_predict(models['point_model'],x)[0]
            draws=np.array([raw+reference_predict(b,x)[0] for b in models['bootstrap_models']]);radius=max(0.,sorted(np.max(draws-point,axis=1))[179])
            close(info['point'],point);close(info['bootstrap_predictions'],draws);close(info['lower'],point-radius);rows[-1]['query_estimates']+=1
        h[at]=raw
    out={'passed':True,'real_chronological_history':True,'dates':rows,'new_capital_outcomes_seen':False,'input_manifest_sha256':sha(inputs/'input_manifest.json')}
    dump(SOURCE/'evidence/preflight_models.json',out);print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);preflight(p.parse_args().inputs)
