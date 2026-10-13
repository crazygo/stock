"""Separate one-step state prediction, fourteen-day returns and actual capital."""
import argparse
from innovation_learning import *


def table(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];panel=pd.read_parquet(run/'innovation_panel.parquet');cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive']);states={pd.Timestamp(x['at']):x for x in read(run/'state_records.json')}
    errors=[];feedback=[];unknown=[];state_errors=[];state_unknown=[];changed={k:0 for k in ['difference','matched','v56']}
    for file in sorted((run/'training').glob('*.json')):
        rec=read(file)
        if not settings['evaluation_start']<=rec['date'][:10]<=settings['evaluation_end']:continue
        at=pd.Timestamp(rec['at']);truth=panel.loc[at];sr=states[at]
        if sr['query']['ready']:
            now=truth[STATES].to_numpy(float);lag=panel.loc[pd.Timestamp(sr['pair']['lag_at']),STATES].to_numpy(float);pred=np.array(sr['query']['prediction'])
            for j,f in enumerate(STATES):state_errors.append({'date':truth.date,'clock':truth.clock,'field':f,'forecast_squared_error':(now[j]-pred[j])**2,'persistence_squared_error':(now[j]-lag[j])**2})
        else:state_unknown.append(at)
        fb={'at':at,'date':truth.date,'clock':truth.clock,'state_training_pairs':sr['fit']['n'],'new_clock_training_rows':len(rec['fits']['innovation']['fit']['members']),'original_clock_training_rows':len(rec['baselines']['v56']['fit']['fit']['members']),'expired_base_gap':rec['fits']['innovation']['feedback']['expired_unresolved_prefix']};feedback.append(fb)
        known=pd.notna(truth.terminal_release) and truth.terminal_release<cutoff and np.isfinite(truth.C) and np.isfinite(truth.D)
        if not known:unknown.append(at)
        rawsets={**{k:rec['fits'][k]['raw'] for k in VARIANTS},**{k:rec['baselines'][k]['fit']['raw'] for k in ['v56','v55']}}
        estimates={**rec['estimates'],**{k:rec['baselines'][k]['estimate'] for k in ['v56','v55']}};common=all(e['ready'] for e in estimates.values())
        for k in changed:
            a=rawsets['innovation'];b=rawsets[k]
            if a and b:changed[k]+=int(max(abs(a[s]-b[s]) for s in core.ETFS)>1e-12)
        for k,raw in rawsets.items():
            if not raw:continue
            estimate=estimates[k];out={'kind':k,'date':truth.date,'raw_C_mse':np.nan,'raw_D_mse':np.nan,'raw_pair_mse':np.nan,'magnitude':np.nan,'brier_numerator':np.nan,'calibrated_pair_mse':np.nan,'common_calibrated_pair_mse':np.nan}
            if known:
                wanted=np.array([truth.C+truth.D,truth.C-truth.D]);pred=np.array([raw[s] for s in core.ETFS]);m=abs(truth.D)
                out.update(raw_C_mse=(raw['C']-truth.C)**2,raw_D_mse=(raw['D']-truth.D)**2,raw_pair_mse=float(np.mean((pred-wanted)**2)),magnitude=m,brier_numerator=m*(raw['q']-int(truth.D>0))**2)
                if estimate['ready']:out['calibrated_pair_mse']=float(np.mean((np.array(estimate['point'])-wanted)**2))
                if common:out['common_calibrated_pair_mse']=out['calibrated_pair_mse']
            errors.append(out)
    metrics={}
    for k,g in pd.DataFrame(errors).groupby('kind'):
        daily=g.groupby('date').mean(numeric_only=True);values=daily.mean().to_dict();values['magnitude_weighted_brier']=float(daily.brier_numerator.mean()/daily.magnitude.mean())
        metrics[k]={'queries':len(g),'dates':len(daily),'mature_raw_queries':int(g.raw_pair_mse.notna().sum()),'mature_calibrated_queries':int(g.calibrated_pair_mse.notna().sum()),'common_calibrated_queries':int(g.common_calibrated_pair_mse.notna().sum()),'common_calibrated_dates':int(g[g.common_calibrated_pair_mse.notna()].date.nunique()),'date_equal':values}
    sm={};state_table=pd.DataFrame(state_errors)
    for field,g in state_table.groupby('field'):
        bydate=g.groupby('date')[['forecast_squared_error','persistence_squared_error']].mean();sm[field]={'queries':len(g),'dates':len(bydate),'date_equal_mse':bydate.mean().to_dict(),'by_clock':g.groupby('clock')[['forecast_squared_error','persistence_squared_error']].mean().to_dict('index')}
    fb=pd.DataFrame(feedback);byday=fb.groupby('date').mean(numeric_only=True);summary={'queries':len(fb),'dates':len(byday),'changed_raw_queries_innovation_vs':changed,'date_equal_mean':byday.mean().to_dict(),'expired_base_gaps':int(fb.expired_base_gap.sum())}
    events=read(run/'decisions_frozen.json')['routes'];routes={}
    for name,es in events.items():
        proposals=[e for e in es if e['action']!='CASH'];trades=table(run/(name+'_trades.csv'));decisions=table(run/(name+'_decisions.csv'))
        routes[name]={'proposals':len(proposals),'proposal_dates':len(set(e['date'] for e in proposals)),'actual_trades':len(trades),'blocked':decisions.loc[decisions.requested_action.ne('CASH')&~decisions.decision.str.startswith('buy_'),'decision'].value_counts().to_dict()}
    primary=table(run/'innovation_band_trades.csv');entryset=lambda t:{(x.decision_at,x.symbol) for x in t.itertuples()};a=entryset(primary);increments={}
    for name in ['difference_band','matched_band','source_v56','source_v55']:
        control=table(run/(name+'_trades.csv'));b=entryset(control);details=lambda t,keys:[{'at':x.decision_at,'symbol':x.symbol,'reason':x.reason,'net_return':x.net_return} for x in t.itertuples() if (x.decision_at,x.symbol) in keys]
        increments[name]={'added_actual_buys':details(primary,a-b),'removed_actual_buys':details(control,b-a),'common_actual_buys':len(a&b),'primary_minus_control_monthly_mean':r['primary']['primary_average']-r['routes'][name]['primary_average'],'paired_capital_interval':r['paired_increments'][name]}
    out={'post_result_only':True,'rules_changed':False,'metrics':metrics,'state_prediction_metrics':sm,'unknown_state_queries_retained':state_unknown,'unknown_actual14_queries_retained':unknown,'feedback_summary':summary,'feedback_rows':feedback,'routes':routes,'increments':increments,'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',out);print({k:out[k] for k in ['metrics','state_prediction_metrics','feedback_summary','routes']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
