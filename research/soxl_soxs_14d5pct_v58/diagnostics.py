"""Date-equal original-target errors, parameter sharing and executable increments."""
import argparse
from shared_learning import *


def table(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    p=read(run/'registration.json')['preregistration']['protocol'];settings=p[r['phase']];panel=pd.read_parquet(run/'activity_panel.parquet');cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive'])
    errors=[];feedback=[];unknown=[];dispersion=[];changed={k:0 for k in ['independent','shared','v56']};previous=None
    for file in sorted((run/'training').glob('*.json')):
        rec=read(file)
        if not settings['evaluation_start']<=rec['date'][:10]<=settings['evaluation_end']:continue
        at=pd.Timestamp(rec['at']);truth=panel.loc[at]
        if rec['date']!=previous:
            base=read(run/'base_daily'/(rec['date'][:10]+'.json'));meta=base['prepared'];previous=rec['date']
            for k in VARIANTS:
                if not meta['ready']:continue
                mapping,_,names=parameter_layout(meta['columns'],k)
                for head in ['C','M','q']:
                    coef=np.array(base['models'][k][head]['coefficients']);squared=0.
                    for f in names:
                        cs=[c for c,fields in enumerate(meta['columns']) if f in fields];w=np.array([meta['eta'][c] for c in cs]);b=np.array([coef[mapping[c][meta['columns'][c].index(f)]] for c in cs]);mu=np.average(b,weights=w);squared+=float(w@((b-mu)**2))
                    dispersion.append({'date':truth.date,'kind':k,'head':head,'weighted_squared_slope_deviation':squared})
        c=CLOCKS.index(truth.clock);fb={'at':at,'date':truth.date,'clock':truth.clock,'joint_parent_dates':meta['N'],'joint_clock_rows':meta['clock_rows'][c],'old_v56_clock_rows':len(rec['baselines']['v56']['fit']['fit']['members']),'expired_unresolved_prefix':base['feedback']['expired_unresolved_prefix']};feedback.append(fb)
        known=pd.notna(truth.terminal_release) and truth.terminal_release<cutoff and np.isfinite(truth.C) and np.isfinite(truth.D)
        if not known:unknown.append(at)
        rawsets={**rec['raw_details'],**{k:rec['baselines'][k]['fit']['raw'] for k in ['v56','v55']}}
        for k in changed:
            a=rawsets['partial'];b=rawsets[k]
            if a and b:changed[k]+=int(max(abs(a[s]-b[s]) for s in core.ETFS)>1e-12)
        for k,raw in rawsets.items():
            if not raw:continue
            estimate=rec['estimates'][k] if k in VARIANTS else rec['baselines'][k]['estimate']
            out={'kind':k,'date':truth.date,'raw_C_mse':np.nan,'raw_D_mse':np.nan,'raw_pair_mse':np.nan,'magnitude':np.nan,'brier_numerator':np.nan,'calibrated_pair_mse':np.nan}
            if known:
                wanted=np.array([truth.C+truth.D,truth.C-truth.D]);pred=np.array([raw[s] for s in core.ETFS]);m=abs(truth.D)
                out.update(raw_C_mse=(raw['C']-truth.C)**2,raw_D_mse=(raw['D']-truth.D)**2,raw_pair_mse=float(np.mean((pred-wanted)**2)),magnitude=m,brier_numerator=m*(raw['q']-int(truth.D>0))**2)
                if estimate['ready']:out['calibrated_pair_mse']=float(np.mean((np.array(estimate['point'])-wanted)**2))
            errors.append(out)
    metrics={}
    for k,g in pd.DataFrame(errors).groupby('kind'):
        daily=g.groupby('date').mean(numeric_only=True);values=daily.mean().to_dict();values['magnitude_weighted_brier']=float(daily.brier_numerator.mean()/daily.magnitude.mean())
        metrics[k]={'queries':len(g),'dates':len(daily),'mature_raw_queries':int(g.raw_pair_mse.notna().sum()),'mature_calibrated_queries':int(g.calibrated_pair_mse.notna().sum()),'date_equal':values}
    fb=pd.DataFrame(feedback);byday=fb.groupby('date').mean(numeric_only=True)
    summary={'queries':len(fb),'dates':len(byday),'changed_raw_queries_partial_vs':changed,'date_equal_mean':byday.mean().to_dict(),'expired_gaps':int(fb.expired_unresolved_prefix.sum())}
    events=read(run/'decisions_frozen.json')['routes'];routes={}
    for name,es in events.items():
        proposals=[e for e in es if e['action']!='CASH'];trades=table(run/(name+'_trades.csv'));decisions=table(run/(name+'_decisions.csv'))
        routes[name]={'proposals':len(proposals),'proposal_dates':len(set(e['date'] for e in proposals)),'actual_trades':len(trades),'blocked':decisions.loc[decisions.requested_action.ne('CASH')&~decisions.decision.str.startswith('buy_'),'decision'].value_counts().to_dict()}
    primary=table(run/'partial_band_trades.csv');entryset=lambda t:{(x.decision_at,x.symbol) for x in t.itertuples()};a=entryset(primary);increments={}
    for name in ['independent_band','shared_band','source_v56','source_v55']:
        control=table(run/(name+'_trades.csv'));b=entryset(control);details=lambda t,keys:[{'at':x.decision_at,'symbol':x.symbol,'reason':x.reason,'net_return':x.net_return} for x in t.itertuples() if (x.decision_at,x.symbol) in keys]
        increments[name]={'added_actual_buys':details(primary,a-b),'removed_actual_buys':details(control,b-a),'common_actual_buys':len(a&b),'primary_minus_control_monthly_mean':r['primary']['primary_average']-r['routes'][name]['primary_average'],'paired_capital_interval':r['paired_increments'][name]}
    ds=pd.DataFrame(dispersion).groupby(['kind','head']).weighted_squared_slope_deviation.mean()
    out={'post_result_only':True,'rules_changed':False,'metrics':metrics,'unknown_actual14_queries_retained':unknown,'feedback_summary':summary,'feedback_rows':feedback,'mean_slope_dispersion':{k:{h:float(ds.loc[k,h]) for h in ['C','M','q']} for k in VARIANTS},'dispersion_rows':dispersion,'routes':routes,'increments':increments,'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',out);print({k:out[k] for k in ['metrics','feedback_summary','mean_slope_dispersion','routes']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
