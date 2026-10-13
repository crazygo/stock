"""Registered as post-result diagnosis; no prediction, action or threshold changes."""
import argparse
from path_learning import *
from audit_dense_calibration import predict


def inspect(run):
    run=Path(run);v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    results=read(run/'results.json');p=read(run/'registration.json')['preregistration']['protocol'][results['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');rows=[];queries=panel[panel.date.between(p['evaluation_start'],p['evaluation_end'])];prior=None;checks=0
    for at,row in queries.iterrows():
        if row.date!=prior:daily=read(run/'daily'/(str(row.date.date())+'.json'));prior=row.date
        r=read(run/'training'/(tag(at)+'.json'));a,b=r['estimates']['conditional'],r['estimates']['ordinary']
        if not a['ready'] or not b['ready']:continue
        raw_a=np.array(r['raws']['conditional']);raw_b=np.array(r['raws']['ordinary']);c=CLOCKS.index(row.clock)
        ma=daily['conditional']['model']['point_model'];mb=daily['ordinary']['model']['point_model']
        for raw,model,saved in [(raw_a,ma,a),(raw_b,mb,b)]:
            value=raw+predict(model,np.r_[raw,c])[0];np.testing.assert_allclose(value,saved['point'],atol=1e-11);checks+=1
        # Exact Jacobian of the old affine raw+residual map, row-vector convention.
        jacobian=np.eye(2)+np.asarray(mb['coefficients'][c])/np.array(mb['input_scale'][c])[:,None]
        direct=(raw_a-raw_b)@jacobian
        hypothetical=raw_a+predict(mb,np.r_[raw_a,c])[0]
        np.testing.assert_allclose(hypothetical-np.array(b['point']),direct,atol=1e-11)
        new_jacobian=np.eye(2)+np.asarray(ma['coefficients'][c])/np.array(ma['input_scale'][c])[:,None]
        propagated=new_jacobian.T@np.array(r['fits']['conditional']['integration']['mean_covariance'])@new_jacobian
        refit=np.array(a['point'])-hypothetical;total=np.array(a['point'])-np.array(b['point'])
        np.testing.assert_allclose(total,direct+refit,atol=1e-11)
        rows.append({'at':at,'date':row.date,'raw_difference':raw_a-raw_b,'direct_same_old_calibrator':direct,'calibrator_refit_difference':refit,'total_point_difference':total,
                     'radius_difference':a['radius']-b['radius'],'old_point_jacobian':jacobian,'old_largest_singular_gain':float(np.linalg.svd(jacobian,compute_uv=False)[0]),
                     'new_largest_singular_gain':float(np.linalg.svd(new_jacobian,compute_uv=False)[0]),'current_query_numerical_point_se':np.sqrt(np.maximum(np.diag(propagated),0)),
                     'action_changed':r['events']['conditional_band']['action']!=r['events']['ordinary_band']['action']})
    frame=pd.DataFrame(rows);out={'passed':True,'post_result_diagnostic':True,'rules_changed':False,'queries':len(rows),'real_dates':int(frame.date.nunique()),'prediction_equations':checks,'action_changes':int(frame.action_changed.sum()),'date_equal_mean_absolute':{}}
    for key in ['raw_difference','direct_same_old_calibrator','calibrator_refit_difference','total_point_difference']:
        vals=np.array(frame[key].to_list());dates=pd.DataFrame({'date':frame.date,'value':abs(vals).mean(axis=1)}).groupby('date').value.mean();out['date_equal_mean_absolute'][key]=float(dates.mean())
    out['date_equal_mean_absolute']['radius_difference']=float(frame.assign(value=abs(frame.radius_difference)).groupby('date').value.mean().mean())
    out['old_jacobian_gain_median_max']=[float(frame.old_largest_singular_gain.median()),float(frame.old_largest_singular_gain.max())]
    out['new_jacobian_gain_median_max']=[float(frame.new_largest_singular_gain.median()),float(frame.new_largest_singular_gain.max())]
    out['current_query_numerical_point_se_mean_max']=[float(np.mean(frame.current_query_numerical_point_se.to_list())),float(np.max(frame.current_query_numerical_point_se.to_list()))]
    out['fixed_calibrator_propagation_excludes_history_refit_and_market_uncertainty']=True
    out['rows']=rows;out['source_sha256']=sha(Path(__file__));out['results_sha256']=sha(run/'results.json');out['scope']='Exact algebraic decomposition between two existing estimators; no future-win selection, alternative capital replay or proposed new acceptance gate.'
    dump(run/'calibration_sensitivity_audit.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);inspect(p.parse_args().run)
