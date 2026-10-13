"""Evaluate already-issued state predictions; never selects a trading route."""
import argparse
from path_learning import *


def report(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    panel=pd.read_parquet(run/'activity_panel.parquet');m=read(run/'input_manifest.json')
    market=core.load_market();assert market.manifest==m['full_market']
    cutoff=pd.Timestamp(m['observation_cutoff_exclusive']);p=read(SOURCE/'protocol.json')[r['phase']]
    positions={d:i for i,d in enumerate(market.sessions)};rows=[];unknown=[]
    for at,row in panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])].iterrows():
        saved=read(run/'training'/(tag(at)+'.json'))
        fit=saved['fits']['conditional']
        if not fit['ready']:
            unknown.append({'at':at,'reason':'forecast_not_ready'});continue
        end_date=market.sessions[positions[row.date]+2]
        end=pd.Timestamp(str(end_date.date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC')
        if end>=cutoff or end not in panel.index or not np.isfinite(panel.loc[end,PRIOR].to_numpy(float)).all():
            unknown.append({'at':at,'reason':'successor_unknown_at_cutoff','successor_at':end});continue
        prediction=np.array(fit['transition']['next_state_point']);persist=row[PRIOR].to_numpy(float);truth=panel.loc[end,PRIOR].to_numpy(float)
        scale=np.array(fit['kernels']['later']['scale'])
        rows.append({'at':at,'date':row.date,'successor_at':end,'prediction':prediction,'persistence':persist,'truth':truth,
            'transition_squared_error':(prediction-truth)**2,'persistence_squared_error':(persist-truth)**2,
            'normalized_transition_mse':float(np.mean(((prediction-truth)/scale)**2)),
            'normalized_persistence_mse':float(np.mean(((persist-truth)/scale)**2))})
    frame=pd.DataFrame(rows);by_date=frame.groupby('date')
    mse={}
    for model in ['transition','persistence']:
        per_date=np.stack([np.stack(g[model+'_squared_error']).mean(axis=0) for _,g in by_date])
        mse[model]={**dict(zip(PRIOR,per_date.mean(axis=0).tolist())),
            'standardized_date_equal_mse':float(by_date['normalized_'+model+'_mse'].mean().mean())}
    comparisons={}
    for label in ['conditional','fixed','unconditional']:
        z=r['routes'][label+'_band']
        comparisons[label]={'monthly_mean':z['primary_average'],'monthly':z['monthly'],'entries':z['entries'],'all_months_nonnegative':z['all_months_nonnegative']}
    out={'phase':r['phase'],'queries':len(rows),'dates':frame.date.nunique(),'unknown':unknown,'state_mse':mse,
         'capital_comparison':comparisons,'state_accuracy_is_not_return_acceptance':True,
         'results_sha256':sha(run/'results.json'),'records':rows}
    dump(run/'state_diagnostics.json',out);print({k:z for k,z in out.items() if k not in ['records','unknown']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run)
