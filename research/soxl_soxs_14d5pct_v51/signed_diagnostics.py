"""Post-result confidence magnitude, boundary choices and action availability."""
import argparse
from collections import Counter
from signed_reliability import *


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']]
    rows=[];coef=[];warmup=0;status={k:Counter() for k in CORRECTIONS}
    for f in sorted((run/'daily').glob('*.json')):
        d=read(f)
        if not settings['evaluation_start']<=d['date'][:10]<=settings['evaluation_end']:continue
        if not d['ready']:warmup+=1;continue
        for k in CORRECTIONS:
            m=d['offsets'][k]['point_model'];bs=d['offsets'][k]['bootstrap_models'];ss=np.array([z['coefficients'][0] for z in bs])
            for z in [m,*bs]:status[k][z.get('solver_message','scalar_bisection_or_endpoint')]+=1
            coef.append({'date':d['date'],'variant':k,'slope':m['coefficients'][0],'parameters':m['coefficients'],
                'bootstrap_slope_mean':float(ss.mean()),'bootstrap_negative':int((ss<0).sum()),'bootstrap_at_minus_one':int((ss==-1).sum()),'bootstrap_at_zero':int((ss==0).sum()),'bootstrap_at_one':int((ss==1).sum()),
                'mature_parent_dates':len(d['parents']),'projected_gradient_max':m['projected_gradient_max']})
    for f in sorted((run/'training').glob('*.json')):
        d=read(f)
        if not settings['evaluation_start']<=d['date'][:10]<=settings['evaluation_end']:continue
        if not d['estimates']['baseline']['ready']:continue
        for k in CORRECTIONS+['inverse','neutral']:
            e=d['estimates'][k];raw=d['fits'][k]['raw'];base=d['fits']['baseline']['raw']
            rows.append({'date':d['date'],'variant':k,'q_shift':raw['q']-base['q'],'abs_q_shift':abs(raw['q']-base['q']),
                'direction_inversion':int((raw['q']-.5)*(base['q']-.5)<0),'direction_erased':int(raw['q']==.5 and base['q']!=.5),
                'source_abs_confidence':abs(base['q']-.5),'new_abs_confidence':abs(raw['q']-.5),'joint_radius':e['radius'],
                'base_radius':d['estimates']['baseline']['radius'],'max_point':max(e['point']),
                'point_positive':int(max(e['point'])>1e-12),'band_positive':int(max(e['lower'])>1e-12),
                'direct_radius':d['direct_estimate']['radius'] if k=='signed' else None})
    frame=pd.DataFrame(rows);summary={}
    for k in CORRECTIONS+['inverse','neutral']:
        s=frame[frame.variant==k];daily=s.groupby('date').mean(numeric_only=True)
        summary[k]={'queries':len(s),'dates':len(daily),'date_equal':daily.mean().to_dict(),
            'point_positive_queries':int(s.point_positive.sum()),'band_positive_queries':int(s.band_positive.sum())}
        if k!='signed':summary[k]['date_equal'].pop('direct_radius',None)
        if k in CORRECTIONS:
            slopes=np.array([x['slope'] for x in coef if x['variant']==k])
            summary[k]['slope']={'mean':float(slopes.mean()),'min':float(slopes.min()),'max':float(slopes.max()),'negative_dates':int((slopes<0).sum()),'dates_at_minus_one':int((slopes==-1).sum()),'dates_at_zero':int((slopes==0).sum()),'dates_at_one':int((slopes==1).sum())}
    assert summary['nonnegative']['date_equal']['direction_inversion']==0
    assert summary['signed']['date_equal']['new_abs_confidence']<=summary['signed']['date_equal']['source_abs_confidence']+1e-12
    result={'post_result_only':True,'rules_changed':False,'warmup_evaluation_dates':warmup,'summary':summary,'daily_slope_models':coef,
        'solver_status_counts':{k:dict(c) for k,c in status.items()},'q_is_magnitude_weighted_relative_direction_not_TP_probability':True,
        'original_base_estimation_uncertainty_not_included':True,'results_sha256':sha(run/'results.json')}
    dump(run/'signed_diagnostics.json',result);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
