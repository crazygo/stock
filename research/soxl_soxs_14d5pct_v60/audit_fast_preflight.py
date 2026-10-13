"""Real declared dates: compare all original and vectorized calibration equations."""
import argparse
from path_learning import *
from audit_dense_calibration import daily_check as slow
from audit_dense_calibration_fast import daily_check as fast
from threadpoolctl import threadpool_limits


def check(run):
    run=Path(run);market=restrict_market(core.load_market(),pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive']));panel=pd.read_parquet(run/'activity_panel.parquet');original=pd.read_parquet(run/'financial_panel.parquet');cache={};rows=[]
    with threadpool_limits(limits=1):
        for date in ['2026-05-01','2026-07-01','2026-09-01']:
            daily=read(run/'daily'/(date+'.json'));sources=read(run/'source_daily'/(date+'.json'));at=panel[panel.date==date].index[0]
            for k in VARIANTS+['v56','v55']:
                obj=daily[k] if k in VARIANTS else sources[k];history={}
                for t in pd.to_datetime(obj['members'],utc=True):
                    if t not in cache:cache[t]=read(run/'training'/(tag(t)+'.json'))
                    raw=cache[t]['raws'][k] if k in VARIANTS else cache[t]['baselines'][k]['raw'];history[t]=np.array(raw)
                table=original if k=='v55' else panel
                a,n=slow(table,market,history,at,obj);b,m=fast(table,market,history,at,obj);assert n==m
                for x,y in [(a['point_model'],b['point_model']),*zip(a['bootstrap_models'],b['bootstrap_models'])]:
                    for f in ['input_center','input_scale','intercepts','coefficients']:np.testing.assert_array_equal(x[f],y[f])
                rows.append({'date':date,'kind':k,'equal_equations':n})
    out={'passed':True,'rows':rows,'total_equations':sum(x['equal_equations'] for x in rows),'source_sha256':sha(Path(__file__)),'capital_results_read':False};dump(SOURCE/'evidence/fast_audit_preflight.json',out);print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);check(p.parse_args().run)
