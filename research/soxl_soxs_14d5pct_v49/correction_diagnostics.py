"""Descriptive source-to-correction shifts and uncertainty, after frozen results."""
import argparse
from direction_correction import *


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']]
    rows=[];coef=[];warmup=0
    for f in sorted((run/'daily').glob('*.json')):
        d=read(f)
        if not settings['evaluation_start']<=d['date'][:10]<=settings['evaluation_end']:continue
        if not d['ready']:warmup+=1;continue
        for k in CORRECTIONS:
            m=d['offsets'][k]['point_model']
            coef.append({'date':d['date'],'variant':k,'parameters':m['coefficients'],'newton_steps':m['iterations'],
                         'mature_parent_dates':len(d['parents']),'gradient_max':m['gradient_max']})
    for f in sorted((run/'training').glob('*.json')):
        d=read(f)
        if not settings['evaluation_start']<=d['date'][:10]<=settings['evaluation_end']:continue
        if not d['estimates']['baseline']['ready']:continue
        for k in CORRECTIONS:
            e=d['estimates'][k];raw=d['fits'][k]['raw'];base=d['fits']['baseline']['raw']
            rows.append({'date':d['date'],'variant':k,'q_shift':raw['q']-base['q'],'abs_q_shift':abs(raw['q']-base['q']),
                'raw_direction_flip':int((raw['q']>.5)!=(base['q']>.5)), 'joint_radius':e['radius'],
                'base_radius':d['estimates']['baseline']['radius'],'max_point':max(e['point']),
                'point_positive':int(max(e['point'])>1e-12),'band_positive':int(max(e['lower'])>1e-12),
                'direct_radius':d['direct_estimate']['radius'] if k=='conditional' else None})
    frame=pd.DataFrame(rows);summary={}
    for k in CORRECTIONS:
        s=frame[frame.variant==k];daily=s.groupby('date').mean(numeric_only=True)
        summary[k]={'queries':len(s),'dates':len(daily),'date_equal':daily.mean().to_dict(),
                    'point_positive_queries':int(s.point_positive.sum()),'band_positive_queries':int(s.band_positive.sum())}
        if k!='conditional':summary[k]['date_equal'].pop('direct_radius',None)
    result={'post_result_only':True,'rules_changed':False,'warmup_evaluation_dates':warmup,'summary':summary,'daily_offset_models':coef,
        'q_is_magnitude_weighted_relative_direction_not_TP_probability':True,
        'original_base_estimation_uncertainty_not_included':True,'results_sha256':sha(run/'results.json')}
    dump(run/'correction_diagnostics.json',result);print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
