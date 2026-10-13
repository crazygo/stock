"""Post-result state dependence, forecast errors and conditional support only."""
import argparse
from state_reliability import *


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']]
    panel=pd.read_parquet(run/'option_panel.parquet');source=Path(read(run/'input_manifest.json')['source'])
    cutoff=pd.Timestamp(read(source/'input_manifest.json')['observation_cutoff_exclusive'])
    rows=[];coefficients=[];daily_models={};unknown=[]
    for f in sorted((run/'daily').glob('*.json')):
        d=read(f)
        if not settings['evaluation_start']<=d['date'][:10]<=settings['evaluation_end']:continue
        daily_models[d['date']]=d
        if not d['ready']:continue
        for k in CORRECTIONS:
            m=d['offsets'][k]['point_model'];b=m['coefficients']
            coefficients.append({'date':d['date'],'variant':k,'coefficients':b,
                'anchor_difference':b[1]-b[0] if k=='state' else None,
                'parents':len(d['parents']),'max_projected_gradient':m['projected_gradient_max']})
    for f in sorted((run/'training').glob('*.json')):
        d=read(f)
        if not settings['evaluation_start']<=d['date'][:10]<=settings['evaluation_end']:continue
        if not d['estimates']['baseline']['ready']:continue
        at=pd.Timestamp(d['at']);row=panel.loc[at];u=float(row.state_heat);model=daily_models[d['date']]
        base=d['fits']['baseline']['raw'];mature=pd.notna(row.terminal_release) and row.terminal_release<cutoff and np.isfinite(row.C) and np.isfinite(row.D)
        if not mature:unknown.append(at)
        for k in VARIANTS:
            e=d['estimates'][k];raw=d['fits'][k]['raw'];effective=None
            if k=='state':
                b=model['offsets'][k]['point_model']['coefficients'];effective=(1-u)*b[0]+u*b[1]
                assert -1<=effective<=1 and abs(raw['q']-.5)<=abs(base['q']-.5)+1e-12
            rows.append({'at':at,'date':row.date,'variant':k,'heat':u,'effective_slope':effective,
                'bin':'low' if u<1/3 else 'middle' if u<2/3 else 'high',
                'direction_inversion':float((raw['q']-.5)*(base['q']-.5)<0),
                'source_abs_confidence':abs(base['q']-.5),'new_abs_confidence':abs(raw['q']-.5),
                'joint_radius':e['radius'],'point_positive':int(max(e['point'])>1e-12),'band_positive':int(max(e['lower'])>1e-12),
                'mature':mature,'magnitude':abs(float(row.D)) if mature else np.nan,
                'brier_numerator':abs(float(row.D))*(raw['q']-int(row.D>0))**2 if mature else np.nan})
    frame=pd.DataFrame(rows);summary={};bins={}
    for k in VARIANTS:
        s=frame[frame.variant==k];days=s.groupby('date').mean(numeric_only=True)
        summary[k]={'queries':len(s),'dates':len(days),'date_equal':days.mean().to_dict(),
            'point_positive_queries':int(s.point_positive.sum()),'band_positive_queries':int(s.band_positive.sum())}
        if k!='state':summary[k]['date_equal'].pop('effective_slope',None)
        if k in CORRECTIONS:
            b=np.array([c['coefficients'] for c in coefficients if c['variant']==k])
            summary[k]['parameters']={'mean':b.mean(axis=0),'min':b.min(axis=0),'max':b.max(axis=0),'at_lower':(b==-1).sum(axis=0),'at_upper':(b==1).sum(axis=0)}
        bins[k]={}
        for name in ['low','middle','high']:
            z=s[s.bin.eq(name)&s.mature];ds=z.groupby('date').mean(numeric_only=True)
            bins[k][name]={'queries':len(z),'dates':len(ds),'magnitude_weighted_brier':float(ds.brier_numerator.mean()/ds.magnitude.mean()) if len(ds) and ds.magnitude.sum()>0 else None}
    result={'post_result_only':True,'rules_changed':False,'summary':summary,'daily_models':coefficients,
        'state_bins':bins,'diagnostic_bins_only':'fixed thirds; never used to fit/select/act',
        'unknown_queries_retained':unknown,'not_hot_regime_probability':True,'results_sha256':sha(run/'results.json')}
    dump(run/'state_diagnostics.json',result);print({'summary':summary,'bins':bins},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
