"""Post-result: distinguish active nonnegative boundary from an interior optimum."""
import argparse
from confidence_shrinkage_v2 import *


def audit(run):
    run=Path(run);result=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[result['phase']];source=Path(read(run/'input_manifest.json')['source'])
    panel=pd.read_parquet(source/'financial_panel.parquet');records={pd.Timestamp(r['at']):r for f in sorted((source/'training').glob('*.json')) if (r:=read(f))};rows=[]
    for f in sorted((run/'daily').glob('*.json')):
        daily=read(f)
        if not settings['evaluation_start']<=daily['date'][:10]<=settings['evaluation_end'] or not daily['ready']:continue
        ids=list(pd.to_datetime(daily['members'],utc=True));n=len(daily['parents']);D=panel.loc[ids,'D'].to_numpy(float)
        q=np.array([records[t]['fits']['full']['raw']['q'] for t in ids]);x=np.log(np.clip(q,1e-6,1-1e-6))-np.log1p(-np.clip(q,1e-6,1-1e-6))
        w=np.array(daily['row_weights'])*abs(D);w*=n/w.sum();y=(D>0).astype(float)
        g0=float(sum(w*(.5-y)*x)-1.);h0=float(sum(w*.25*x*x)+1.)
        model=daily['offsets']['temperature']['point_model'];s=model['coefficients'][0]
        if s==0:assert g0>=-2e-8*n
        loss=lambda a:float(sum(w*(np.logaddexp(0,a*x)-y*a*x))+.5*(a-1)**2)
        assert loss(s)<=loss(0)+1e-8 and loss(s)<=loss(1)+1e-8
        rows.append({'date':f.stem,'slope':s,'gradient_at_zero':g0,'curvature_at_zero':h0,
            'objective_at_zero':loss(0),'objective_at_original':loss(1),'objective_at_fitted':loss(s),
            'negative_slope_descent_available':g0>1e-8*n,'mature_dates':n})
    zero=[r for r in rows if r['slope']==0];out={'passed':True,'post_result_only':True,'rules_changed':False,'evaluation_dates':len(rows),
        'at_zero_dates':len(zero),'zero_dates_with_negative_slope_descent':sum(r['negative_slope_descent_available'] for r in zero),
        'mean_gradient_at_zero':float(np.mean([r['gradient_at_zero'] for r in rows])),
        'interpretation':'Gradient refers only to previously matured feedback. It does not establish future inverse predictability or authorize reversing trades.',
        'rows':rows,'results_sha256':sha(run/'results.json')}
    dump(run/'boundary_audit.json',out);print({k:v for k,v in out.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
