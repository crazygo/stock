"""Post-run error decomposition versus causal own-pair intercept, no strategy fit."""
import argparse
import gzip
import json
from block_learning import *


def diagnose(run):
    run=Path(run);before={f:sha(run/f) for f in ['predictions.json','results.json']}
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');panel=panel[panel.date.between('2026-01-01','2026-04-30')]
    diagnostics=read(run/'diagnostics.json');rows=[];previous=None;kinds=VARIANTS+['uniform','old_quadratic','linear']
    for at,row in panel.iterrows():
        if pd.isna(row.terminal_release):continue
        if previous!=row.date:
            with gzip.open(run/'models'/(str(row.date.date())+'.json.gz'),'rt') as f:models=json.load(f)
            previous=row.date
        model=models[row.clock];j=model['names'].index('semiconductor');base=np.array(model['intercepts'][j]);truth=np.array([row.C,row.D])
        record=read(run/'training'/(tag(at)+'.json'));item={'at':at,'date':row.date,'clock':row.clock}
        for name,cd in [('causal_intercept',base)]:
            item[name+'_C_mse']=float((cd[0]-truth[0])**2);item[name+'_D_mse']=float((cd[1]-truth[1])**2)
        for kind in kinds:
            raw=record['raws'][kind] if kind in VARIANTS else record[kind+'_raw']
            cal=record['estimates'][kind] if kind in VARIANTS else record[kind+'_estimate']
            assert raw is not None and cal['ready']
            for stage,pair in [('raw',raw),('cal',cal['point'])]:
                cd=np.array([(pair[0]+pair[1])/2,(pair[0]-pair[1])/2]);prefix=kind+'_'+stage
                item[prefix+'_C_mse']=float((cd[0]-truth[0])**2);item[prefix+'_D_mse']=float((cd[1]-truth[1])**2)
                # Independent identity against the two-side squared error.
                np.testing.assert_allclose(sum((cd-truth)**2),np.mean((np.array(pair)-np.array([row.C+row.D,row.C-row.D]))**2),atol=1e-14)
                item[prefix+'_C_bias']=float(cd[0]-truth[0]);item[prefix+'_D_bias']=float(cd[1]-truth[1])
        rows.append(item)
    table=pd.DataFrame(rows);daily=table.groupby('date').mean(numeric_only=True);means=daily.mean();baseline={k:float(means['causal_intercept_'+k+'_mse']) for k in ['C','D']}
    results={}
    for kind in kinds:
        results[kind]={}
        for stage in ['raw','cal']:
            prefix=kind+'_'+stage;v={k:float(means[prefix+'_'+k]) for k in ['C_mse','D_mse','C_bias','D_bias']}
            np.testing.assert_allclose(v['C_mse']+v['D_mse'],diagnostics['errors'][kind][stage]['date_equal_mse'],atol=1e-12,rtol=1e-12)
            v['C_mse_improvement_vs_causal_intercept']=1-v['C_mse']/baseline['C'];v['D_mse_improvement_vs_causal_intercept']=1-v['D_mse']/baseline['D'];results[kind][stage]=v
    for f,value in before.items():assert sha(run/f)==value
    table.to_csv(run/'component_rows.csv',index=False)
    out={'passed':True,'stage':'post_run_descriptive_only','dates':len(daily),'ticks':len(table),'models':results,
         'causal_intercept_mse':baseline,'reference':'Own semiconductor intercept saved at day00, trained on same past fully mature prefix and clock. No current labels in this baseline; this is error comparison, not another capital strategy.',
         'two_side_identity_checks':len(table)*len(kinds)*2,'aggregate_identity_checks':len(kinds)*2,
         'source_sha256':sha(Path(__file__)),'source_results_sha256':before['results.json'],'changes_predictions_or_gate':False}
    dump(run/'component_diagnostic.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();diagnose(a.run)
