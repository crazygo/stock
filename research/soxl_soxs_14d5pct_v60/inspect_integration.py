"""Post-result numerical diagnostics only; never modify models, actions or gates."""
import argparse,gzip,pickle
from path_learning import *
from audit_dense_calibration_fast import predict as ref_predict


def inspect(run):
    run=Path(run);result=read(run/'results.json');assert read(run/'verification.json')['passed'];settings=read(SOURCE/'protocol.json')[result['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');panel=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])];rows=[];cache={};jacobian_checks=0
    for at,row in panel.iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));date=str(row.date.date())
        if date not in cache:cache[date]=read(run/'daily'/(date+'.json'))
        for k in VARIANTS:
            fit=rec['fits'][k];est=rec['estimates'][k];day=cache[date][k];c=CLOCKS.index(row.clock);raw=np.asarray(rec['raws'][k]) if rec['raws'][k] is not None else None
            item={'date':date,'month':date[:7],'kind':k,'raw_ready':fit['ready'],'calibration_ready':day['ready'],'expired':day['feedback']['expired_unresolved_prefix'],'point_action':rec['events']['conditional_point']['action'] if k=='conditional' else None,'band_action':rec['events'][k+'_band']['action']}
            if est['ready']:
                model=day['model']['point_model'];scale=np.array(model['input_scale'][c]);coef=np.array(model['coefficients'][c]);J=np.eye(2)+coef.T/scale[None,:]
                point=lambda a:a+ref_predict(model,np.array([[a[0],a[1],c]]))[0]
                np.testing.assert_allclose(point(raw),est['point'],atol=1e-11)
                for j in range(2):
                    delta=np.eye(2)[j]*1e-6;np.testing.assert_allclose((point(raw+delta)-point(raw-delta))/(2e-6),J[:,j],rtol=1e-7,atol=1e-8);jacobian_checks+=1
                with gzip.open(run/'models'/(k+'_'+tag(at)+'.pkl.gz'),'rb') as f:sim=pickle.load(f)['result']['log_returns']
                covariance=np.cov(sim,rowvar=False,ddof=1)/len(sim);raw_se=np.sqrt(np.diag(covariance));corrected_se=np.sqrt(np.maximum(0,np.diag(J@covariance@J.T)))
                # Second derivation: transform centered path residuals, then SE.
                transformed=(sim-sim.mean(axis=0))@J.T;np.testing.assert_allclose(corrected_se,transformed.std(axis=0,ddof=1)/np.sqrt(len(sim)),rtol=1e-11,atol=1e-12)
                item.update(raw_se=raw_se.tolist(),corrected_se=corrected_se.tolist(),radius=est['radius'],max_point=max(est['point']),max_lower=max(est['lower']))
            rows.append(item)
    output={}
    for k in VARIANTS:
        group=[r for r in rows if r['kind']==k];ready=[r for r in group if 'raw_se' in r];raw=np.array([r['raw_se'] for r in ready]);cor=np.array([r['corrected_se'] for r in ready]);ratio=cor/raw
        months={}
        for m in sorted(set(r['month'] for r in group)):
            g=[r for r in group if r['month']==m];f=[r for r in g if 'raw_se' in r];months[m]={'queries':len(g),'raw_ready':sum(r['raw_ready'] for r in g),'calibration_ready':sum(r['calibration_ready'] for r in g),'expired_prefix':sum(r['expired'] for r in g),'positive_point_queries':sum(r['max_point']>1e-12 for r in f),'positive_lower_queries':sum(r['max_lower']>1e-12 for r in f),'band_proposals':sum(r['band_action']!='CASH' for r in g),'radius_median':float(np.median([r['radius'] for r in f])),'max_point_median':float(np.median([r['max_point'] for r in f]))}
        output[k]={'queries':len(group),'dates':len(set(r['date'] for r in group)),'ready_queries':len(ready),'raw_integration_se_mean':float(raw.mean()),'calibrated_integration_se_mean':float(cor.mean()),'calibrated_integration_se_p95':float(np.quantile(cor,.95)),'calibrated_to_raw_se_median':float(np.median(ratio)),'months':months}
    receipt={'phase':result['phase'],'post_outcome_diagnostic':True,'variants':output,'jacobian_equalities':jacobian_checks,'covariance_and_transformed_residual_agree':True,'scope':'Numerical integration variance conditional on the already fitted calibration map only; not market prediction uncertainty, not the uncertainty of fitting the map, and not a counterfactual proof that more samples improve capital. No selection or gate changes.','results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__))};dump(run/'integration_inspection.json',receipt);print(receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);inspect(p.parse_args().run)
