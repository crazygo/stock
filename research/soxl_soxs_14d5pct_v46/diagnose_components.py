"""Post-result common/opposed payoff error decomposition; no oracle policy."""
import argparse
from fixed_candidate import *


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];s=p[r['phase']];panel=pd.read_parquet(run/'macro_panel.parquet');ev=panel[panel.date.between(s['evaluation_start'],s['evaluation_end'])];root=Path(read(run/'input_manifest.json')['source']);cutoff=pd.Timestamp(read(run/'input_manifest.json')['market']['observation_cutoff_exclusive']);records=[];unknown=[]
    for at,row in ev.iterrows():
        if pd.isna(row.terminal_release) or row.terminal_release>=cutoff or not np.isfinite(row.C) or not np.isfinite(row.D):unknown.append(at);continue
        rec=read(run/'training'/(tag(at)+'.json'));src=read(root/'training'/(tag(at)+'.json'));raw=src['fits']['full']['raw'];truth=np.array([row.C+row.D,row.C-row.D]);mag=abs(row.D)
        for kind,pred in [('raw',rec['raw']),('strong',rec['estimates']['strong']['point']),('plain',rec['estimates']['plain']['point'])]:
            pred=np.array(pred);c=pred.mean();d=(pred[0]-pred[1])/2;j=int(pred[1]>pred[0]);cm=(c-row.C)**2;dm=(d-row.D)**2
            np.testing.assert_allclose(cm+dm,np.mean((pred-truth)**2),atol=1e-14)
            records.append({'date':row.date,'at':at,'kind':kind,'C_truth':row.C,'C_pred':c,'D_truth':row.D,'D_pred':d,'C_mse':cm,'D_mse':dm,'pair_mse':cm+dm,
                'mean_pair_payoff':truth.mean(),'best_pair_payoff':truth.max(),'chosen_payoff':truth[j],'direction_regret':truth.max()-truth[j],
                'both_profitable':bool(truth.min()>0),'both_losing':bool(truth.max()<=0),'magnitude':mag,'raw_weighted_brier_numerator':mag*(raw['q']-int(row.D>0))**2})
    f=pd.DataFrame(records);out={}
    for kind in ['raw','strong','plain']:
        q=f[f.kind==kind];out[kind]={}
        for name,g in [('all',q)]+[(str(m),g) for m,g in q.groupby(q.date.dt.to_period('M'))]:
            stats=g.groupby('date').mean(numeric_only=True).mean().to_dict();stats['raw_magnitude_weighted_brier']=stats['raw_weighted_brier_numerator']/stats['magnitude'];out[kind][name]={'queries':len(g),'dates':g.date.nunique(),'date_equal':stats}
    f.to_csv(run/'component_queries.csv',index=False)
    dump(run/'component_diagnostics.json',{'post_result_only':True,'does_not_generate_actions':True,'phase':r['phase'],'unknown_queries':unknown,'statistics':out,'results_sha256':sha(run/'results.json'),'code_sha256':sha(Path(__file__)),
        'caveats':['Best-of-pair payoff and regret use future truth only as post-result description, not an available forecast or trade.','MSE components alone do not establish achievable incremental capital returns.','Repeated same-day outcomes are date-equal, still overlap across dates.']})
    print({k:{n:v for n,v in val.items() if n=='all'} for k,val in out.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
