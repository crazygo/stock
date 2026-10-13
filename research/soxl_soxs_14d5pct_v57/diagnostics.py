"""Separate proxy accuracy, fourteen-day accuracy, feedback and actual capital."""
import argparse
from short_learning import *


def table(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');short=pd.read_parquet(run/'short_panel.parquet');cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive'])
    errors=[];feedback=[];unknown14=[];unknown_short=[];changed=0
    for file in sorted((run/'training').glob('*.json')):
        rec=read(file)
        if not settings['evaluation_start']<=rec['date'][:10]<=settings['evaluation_end']:continue
        at=pd.Timestamp(rec['at']);truth=panel.loc[at];proxy=short.loc[at]
        known14=pd.notna(truth.terminal_release) and truth.terminal_release<cutoff and np.isfinite(truth.C) and np.isfinite(truth.D)
        knownshort=pd.notna(proxy.terminal_release) and proxy.terminal_release<cutoff and np.isfinite(proxy.C) and np.isfinite(proxy.D)
        if not known14:unknown14.append(at)
        if not knownshort:unknown_short.append(at)
        f,l=rec['fits']['fast'],rec['fits']['lagged']
        gap=len(f['fit']['members'])-len(l['fit']['members']);assert gap>=0
        if f['raw'] and l['raw']:changed+=int(max(abs(f['raw'][s]-l['raw'][s]) for s in core.ETFS)>1e-12)
        item={'at':at,'date':truth.date,'clock':truth.clock,'extra_recent_members':gap,'short_unresolved_expired':f['short_unresolved_expired']}
        for k in VARIANTS:
            members=rec['fits'][k]['fit']['members'];last=pd.Timestamp(members[-1]) if members else None
            item[k+'_last_member_age_calendar_days']=(at-last).total_seconds()/86400 if last is not None else None
        feedback.append(item)
        for k in VARIANTS+['v56','v55']:
            raw=rec['fits'][k]['raw'] if k in VARIANTS else rec['baselines'][k]['fit']['raw']
            estimate=rec['estimates'][k] if k in VARIANTS else rec['baselines'][k]['estimate']
            if not raw:continue
            out={'kind':k,'date':truth.date,'short_raw_pair_mse':np.nan,'short_direction_mse':np.nan,'short_magnitude':np.nan,'short_brier_numerator':np.nan,'calibrated_actual14_pair_mse':np.nan}
            if k in VARIANTS and knownshort:
                wanted=np.array([proxy.C+proxy.D,proxy.C-proxy.D]);pred=np.array([raw[s] for s in core.ETFS]);m=abs(proxy.D)
                out.update(short_raw_pair_mse=float(np.mean((pred-wanted)**2)),short_direction_mse=(raw['D']-proxy.D)**2,short_magnitude=m,short_brier_numerator=m*(raw['q']-int(proxy.D>0))**2)
            if known14 and estimate['ready']:
                wanted=np.array([truth.C+truth.D,truth.C-truth.D]);out['calibrated_actual14_pair_mse']=float(np.mean((np.array(estimate['point'])-wanted)**2))
            errors.append(out)
    metrics={}
    for k,g in pd.DataFrame(errors).groupby('kind'):
        daily=g.groupby('date').mean(numeric_only=True);values=daily.mean().to_dict()
        if k in VARIANTS:values['short_magnitude_weighted_brier']=float(daily.short_brier_numerator.mean()/daily.short_magnitude.mean())
        metrics[k]={'queries':len(g),'dates':len(daily),'date_equal':values,'mature_short_queries':int(g.short_raw_pair_mse.notna().sum()),'mature_actual14_queries':int(g.calibrated_actual14_pair_mse.notna().sum())}
    fb=pd.DataFrame(feedback);byday=fb.groupby('date').mean(numeric_only=True)
    feedback_summary={'queries':len(fb),'dates':len(byday),'raw_queries_changed_fast_vs_lagged':changed,'extra_recent_members_min':int(fb.extra_recent_members.min()),'extra_recent_members_max':int(fb.extra_recent_members.max()),'extra_recent_members_date_equal_mean':float(byday.extra_recent_members.mean()),'fast_last_age_date_equal_mean':float(byday.fast_last_member_age_calendar_days.mean()),'lagged_last_age_date_equal_mean':float(byday.lagged_last_member_age_calendar_days.mean()),'expired_short_gaps':int(fb.short_unresolved_expired.sum())}
    events=read(run/'decisions_frozen.json')['routes'];routes={}
    for name,es in events.items():
        proposals=[e for e in es if e['action']!='CASH'];trades=table(run/(name+'_trades.csv'));decisions=table(run/(name+'_decisions.csv'))
        routes[name]={'proposals':len(proposals),'proposal_dates':len(set(e['date'] for e in proposals)),'actual_trades':len(trades),'blocked':decisions.loc[decisions.requested_action.ne('CASH')&~decisions.decision.str.startswith('buy_'),'decision'].value_counts().to_dict()}
    primary=table(run/'fast_band_trades.csv');entryset=lambda t:{(x.decision_at,x.symbol) for x in t.itertuples()};a=entryset(primary);increments={}
    for name in ['lagged_band','source_v56','source_v55']:
        control=table(run/(name+'_trades.csv'));b=entryset(control);details=lambda t,keys:[{'at':x.decision_at,'symbol':x.symbol,'reason':x.reason,'net_return':x.net_return} for x in t.itertuples() if (x.decision_at,x.symbol) in keys]
        increments[name]={'added_actual_buys':details(primary,a-b),'removed_actual_buys':details(control,b-a),'common_actual_buys':len(a&b),'primary_minus_control_monthly_mean':r['primary']['primary_average']-r['routes'][name]['primary_average'],'paired_capital_interval':r['paired_increments'][name]}
    out={'post_result_only':True,'rules_changed':False,'metrics':metrics,'unknown_actual14_queries_retained':unknown14,'unknown_short_queries_retained':unknown_short,'feedback_summary':feedback_summary,'feedback_rows':feedback,'routes':routes,'increments':increments,'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',out);print({k:out[k] for k in ['metrics','feedback_summary','routes']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
