"""Read-only independent arithmetic for diagnostic denominators and sharing."""
import argparse
from collections import defaultdict
from innovation_learning import *
from verify_continuation import near


def csv(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def audit(run):
    run=Path(run);d=read(run/'diagnostics.json');r=read(run/'results.json');assert d['results_sha256']==sha(run/'results.json') and d['post_result_only'] and not d['rules_changed']
    v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    settings=read(run/'registration.json')['preregistration']['protocol'][r['phase']];panel=pd.read_parquet(run/'innovation_panel.parquet');cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive'])
    values=defaultdict(lambda:defaultdict(lambda:defaultdict(list)));counts=defaultdict(lambda:[0,0,0,0]);unknown=[];change={k:0 for k in ['difference','matched','v56']};fb=[];checks=0
    states={pd.Timestamp(x['at']):x for x in read(run/'state_records.json')};state_values=defaultdict(lambda:defaultdict(list));state_clock=defaultdict(list);unknown_state=[]
    for f in sorted((run/'training').glob('*.json')):
        rec=read(f);date=rec['date'][:10]
        if not settings['evaluation_start']<=date<=settings['evaluation_end']:continue
        at=pd.Timestamp(rec['at']);row=panel.loc[at];known=pd.notna(row.terminal_release) and row.terminal_release<cutoff and np.isfinite(row.C) and np.isfinite(row.D)
        if not known:unknown.append(str(at))
        raws={**{k:rec['fits'][k]['raw'] for k in VARIANTS},**{k:rec['baselines'][k]['fit']['raw'] for k in ['v56','v55']}}
        for k in change:change[k]+=int(bool(raws['innovation'] and raws[k]) and any(abs(raws['innovation'][s]-raws[k][s])>1e-12 for s in core.ETFS))
        sr=states[at]
        fb.append({'at':str(at),'date':str(row.date),'clock':rec['clock'],'state_training_pairs':sr['fit']['n'],'new_clock_training_rows':len(rec['fits']['innovation']['fit']['members']),'original_clock_training_rows':len(rec['baselines']['v56']['fit']['fit']['members']),'expired_base_gap':rec['fits']['innovation']['feedback']['expired_unresolved_prefix']})
        estimates={**rec['estimates'],**{k:rec['baselines'][k]['estimate'] for k in ['v56','v55']}};common=all(x['ready'] for x in estimates.values())
        if sr['query']['ready']:
            for j,field in enumerate(STATES):
                actual=float(row[field]);lag=float(panel.loc[pd.Timestamp(sr['pair']['lag_at']),field]);pred=sr['query']['prediction'][j]
                for key,value in [('forecast_squared_error',(actual-pred)**2),('persistence_squared_error',(actual-lag)**2)]:state_values[field,key][date].append(value);state_clock[field,rec['clock'],key].append(value)
        else:unknown_state.append(at)
        for k,raw in raws.items():
            if not raw:continue
            counts[k][0]+=1;out=values[k][date]
            if not known:continue
            counts[k][1]+=1;cerror=(raw['C']-float(row.C))**2;derror=(raw['D']-float(row.D))**2
            out['raw_C_mse'].append(cerror);out['raw_D_mse'].append(derror);out['raw_pair_mse'].append(cerror+derror)
            out['magnitude'].append(abs(float(row.D)));out['brier_numerator'].append(abs(float(row.D))*(raw['q']-(1. if row.D>0 else 0.))**2)
            estimate=rec['estimates'][k] if k in VARIANTS else rec['baselines'][k]['estimate']
            if estimate['ready']:
                a,b=estimate['point'];out['calibrated_pair_mse'].append(((a-float(row.C+row.D))**2+(b-float(row.C-row.D))**2)/2);counts[k][2]+=1
                if common:out['common_calibrated_pair_mse'].append(out['calibrated_pair_mse'][-1]);counts[k][3]+=1
    for k,days in values.items():
        saved=d['metrics'][k];assert [saved['queries'],saved['mature_raw_queries'],saved['mature_calibrated_queries'],saved['common_calibrated_queries']]==counts[k] and saved['dates']==len(days)
        estimates={}
        for field in ['raw_C_mse','raw_D_mse','raw_pair_mse','magnitude','brier_numerator','calibrated_pair_mse','common_calibrated_pair_mse']:
            perday=[sum(v[field])/len(v[field]) for v in days.values() if v[field]];estimates[field]=sum(perday)/len(perday);near(saved['date_equal'][field],estimates[field]);checks+=1
        near(saved['date_equal']['magnitude_weighted_brier'],estimates['brier_numerator']/estimates['magnitude']);checks+=1
    assert list(pd.to_datetime(d['unknown_actual14_queries_retained'],utc=True))==list(pd.to_datetime(unknown,utc=True))
    assert d['feedback_summary']['changed_raw_queries_innovation_vs']==change and d['feedback_rows']==fb
    weights=pd.DataFrame(fb).groupby('date').mean(numeric_only=True)
    for field,value in weights.mean().items():near(d['feedback_summary']['date_equal_mean'][field],value);checks+=1
    assert list(pd.to_datetime(d['unknown_state_queries_retained'],utc=True))==unknown_state
    for (field,key),days in state_values.items():
        saved=d['state_prediction_metrics'][field];assert saved['dates']==len(days) and saved['queries']==sum(len(x) for x in days.values())
        expected=sum(sum(x)/len(x) for x in days.values())/len(days);near(saved['date_equal_mse'][key],expected);checks+=1
    for (field,clock,key),items in state_clock.items():near(d['state_prediction_metrics'][field]['by_clock'][clock][key],sum(items)/len(items));checks+=1
    for name,es in read(run/'decisions_frozen.json')['routes'].items():
        positive=[e for e in es if e['action']!='CASH'];trades=csv(run/(name+'_trades.csv'));row=d['routes'][name]
        assert row['proposals']==len(positive) and row['proposal_dates']==len(set(e['date'] for e in positive)) and row['actual_trades']==len(trades);checks+=1
    primary=csv(run/'innovation_band_trades.csv');entry=lambda t:{(x.decision_at,x.symbol) for x in t.itertuples()};a=entry(primary)
    for name,row in d['increments'].items():
        control=csv(run/(name+'_trades.csv'));b=entry(control)
        assert {(x['at'],x['symbol']) for x in row['added_actual_buys']}==a-b
        assert {(x['at'],x['symbol']) for x in row['removed_actual_buys']}==b-a and row['common_actual_buys']==len(a&b)
        near(row['primary_minus_control_monthly_mean'],r['primary']['primary_average']-r['routes'][name]['primary_average']);assert row['paired_capital_interval']==r['paired_increments'][name];checks+=1
    receipt={'passed':True,'diagnostic_equations':checks,'unknown_queries_retained':len(unknown),'metrics_denominators':dict(counts),'new_rules':False,'source_sha256':sha(Path(__file__)),'diagnostics_sha256':sha(run/'diagnostics.json'),'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics_audit.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);audit(p.parse_args().run)
