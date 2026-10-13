"""Registered diagnostics; none select or alter the capital strategy."""
import argparse
from path_learning import *


def report(run):
    run=Path(run);result=read(run/'results.json');verification=read(run/'verification.json');assert verification['passed'] and verification['results_sha256']==sha(run/'results.json')
    p=read(SOURCE/'protocol.json');settings=p[result['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');query=panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])];cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive']);data={k:[] for k in VARIANTS};unknown={k:0 for k in VARIANTS};rows={k:[] for k in VARIANTS};stats={}
    for at,row in query.iterrows():
        rec=read(run/'training'/(tag(at)+'.json'))
        for k in VARIANTS:
            f=rec['fits'][k]
            if not f['ready']:continue
            rows[k].append({'at':at,'date':row.date,'source_blocks':len(f['members']),'effective_first':f['kernels']['first']['effective_dates'],'effective_later':f['kernels']['later']['effective_dates'],'se':f['monte_carlo_se'],'reason_frequencies':f['reason_frequencies']})
            if pd.isna(row.terminal_release) or row.terminal_release>=cutoff:unknown[k]+=1;continue
            raw=f['raw'];target=np.array([row.C+row.D,row.C-row.D]);point=np.array(rec['estimates'][k]['point']) if rec['estimates'][k]['ready'] else None
            data[k].append({'at':at,'date':row.date,'raw_mse':float(np.mean((np.array(rec['raws'][k])-target)**2)),'point_mse':float(np.mean((point-target)**2)) if point is not None else np.nan,'q_loss':float(abs(row.D)*(raw['q']-float(row.D>0))**2),'weight':float(abs(row.D))})
    for k in VARIANTS:
        f=pd.DataFrame(rows[k]);e=pd.DataFrame(data[k]);daily=e.groupby('date')[['raw_mse','point_mse','q_loss','weight']].mean();positive=daily.weight>0
        stats[k]={'ready_queries':len(f),'real_dates':int(f.date.nunique()),'mature_queries':len(e),'unknown_queries':unknown[k],'mature_dates':int(e.date.nunique()),'source_blocks_min_max':[int(f.source_blocks.min()),int(f.source_blocks.max())],'effective_first_min_median':[float(f.effective_first.min()),float(f.effective_first.median())],'effective_later_min_median':[float(f.effective_later.min()),float(f.effective_later.median())],'integration_se_mean_max':[float(np.mean(f.se.tolist())),float(np.max(f.se.tolist()))],'raw_log_return_mse_date_equal':float(daily.raw_mse.mean()),'calibrated_log_return_mse_date_equal':float(daily.point_mse.mean()),'direction_brier_absD_within_date_then_date_equal':float((daily.loc[positive,'q_loss']/daily.loc[positive,'weight']).mean()),'direction_neutral_brier':.25,'mean_simulated_reason_frequencies':{i:np.mean([x[i] for x in f.reason_frequencies],axis=0).tolist() for i in ['1','2','3']}}
    actual={};predictions=read(run/'decisions_frozen.json')['routes']
    for route in ROUTES:
        d=pd.read_csv(run/(route+'_decisions.csv'));proposals=[x for x in predictions[route] if x['action'] in core.ETFS]
        try:t=pd.read_csv(run/(route+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        actual[route]={'proposals':len(proposals),'proposal_dates':len(set(x['date'] for x in proposals)),'entries':len(t),'decisions':d.decision.value_counts().to_dict(),'trades':t[['decision_at','symbol','reason','net_return']].to_dict('records') if len(t) else []}
    out={'phase':result['phase'],'variants':stats,'capital_actions':actual,'scaled_minus_fresh_month_mean':result['primary']['primary_average']-result['routes']['fresh_point']['primary_average'],'primary_passed':result.get('development_gate',result.get('case_acceptance'))['passed'],'simulations_are_not_independent_market_dates':True,'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json')}
    dump(run/'diagnostics.json',out);print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);report(p.parse_args().run)
