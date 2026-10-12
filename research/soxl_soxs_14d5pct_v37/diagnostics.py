"""Post-result audits of selection, class-weighted quality and input support."""
import argparse
from collections import Counter
from selected_risk import core,np,pd,Path,read,dump,sha,tag,CLOCKS,SCORES,CONTEXT,columns


def totals(frame,column):
    if frame.empty:return {'queries':0,'dates':0}
    groups=frame.groupby('date');weighted=[]
    for _,g in groups:
        r=g.actual.to_numpy();q=g[column].to_numpy();m=np.abs(r);y=(r>0).astype(float)
        weighted.append([np.sum(m*(q-y)**2)/len(g),np.sum(m*(-y*np.log(np.clip(q,1e-15,1))-(1-y)*np.log(np.clip(1-q,1e-15,1))))/len(g),m.mean()])
    vals=np.array(weighted).mean(axis=0)
    return {'queries':len(frame),'dates':frame.date.nunique(),'actual_win_fraction_date_equal':float(groups.actual.apply(lambda s:(s>0).mean()).mean()),
        'actual_mean_netlog_date_equal':float(groups.actual.mean().mean()),'magnitude_weighted_brier':float(vals[0]/vals[2]),
        'magnitude_weighted_logloss':float(vals[1]/vals[2]),'mean_quality_date_equal':float(groups[column].mean().mean())}


def diagnose(run):
    run=Path(run);manifest=read(run/'input_manifest.json');root=Path(manifest['verified_source']);panel=pd.read_parquet(run/'selected_panel.parquet')
    source={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))};rows=[];daily_rows=[]
    for f in sorted((run/'daily').glob('*.json')):
        d=read(f)
        if not ('2026-01-01'<=d['date'][:10]<='2026-04-30'):continue
        result={'date':d['date'][:10],'ready':d['ready'],'meta_dates':len(d['parents'])}
        if d['ready']:
            ids=list(pd.to_datetime(d['members'],utc=True));w=np.array(d['row_weights']);r=panel.loc[ids,'selected_y'].to_numpy(float)
            eligible=np.array([source[t]['events']['full_band']['action']!='CASH' for t in ids])
            subsets={}
            for name,m in [('all',np.ones(len(ids),bool)),('source_eligible',eligible),('source_rejected',~eligible)]:
                ww=w[m];rr=r[m];subsets[name]={'rows':int(m.sum()),'dates':panel.loc[np.array(ids)[m],'date'].nunique(),
                    'weighted_mean_netlog':float(np.average(rr,weights=ww)) if len(ww) else None,
                    'positive_gain_share':float(np.sum(ww*np.maximum(rr,0))/np.sum(ww*np.abs(rr))) if len(ww) and np.sum(ww*np.abs(rr)) else None}
            result['training_subsets']=subsets
        daily_rows.append(result)
    for at,row in panel[panel.date.between('2026-01-01','2026-04-30')].iterrows():
        saved=read(run/'training'/(tag(at)+'.json'));old=source[at];info=saved['estimates']['context'];d=read(run/'daily'/(str(row.date.date())+'.json'))
        active=bool(info['ready'] and not info['source_fallback']);eligible=old['events']['full_band']['action']!='CASH'
        entry={'at':at,'date':row.date,'clock':row.clock,'active':active,'eligible':eligible,'source_action':old['events']['full_band']['action'],
            'new_action':saved['events']['context_gate']['action'],'actual':row.selected_y,'known':bool(np.isfinite(row.selected_y)),
            **{k:saved['estimates'][k]['quality'] for k in ['context','scores','frequency']}}
        if active:
            hist=panel.loc[list(pd.to_datetime(d['members'],utc=True))];outside=[c for c in SCORES+CONTEXT if row[c]<hist[c].min()-1e-12 or row[c]>hist[c].max()+1e-12]
            entry['outside_history_columns']=outside
            entry['context_terms']={c:float((row[c]-d['models']['context']['center'][j])/d['models']['context']['scale'][j]*d['models']['context']['coef'][j]) for j,c in enumerate(SCORES+CONTEXT)}
            entry['context_intercept']=d['models']['context']['intercept']
        rows.append(entry)
    frame=pd.DataFrame(rows);active=frame[frame.active&frame.known];error={}
    for name,g in [('all_active_top',active),('source_eligible',active[active.eligible]),('source_rejected',active[~active.eligible])]:
        error[name]={k:totals(g,k) for k in ['context','scores','frequency']}
    # All as-issued top rows remain in diagnostics, including source-rejected and unknown.
    current=pd.read_csv(run/'context_gate_trades.csv');old=pd.read_csv(run/'source_v35_trades.csv')
    keys=lambda f:set(zip(f.decision_at,f.symbol));added=keys(current)-keys(old);removed=keys(old)-keys(current)
    subset=lambda f,k:f[[x in k for x in zip(f.decision_at,f.symbol)]][['decision_at','symbol','reason','net_return','capital']].to_dict('records')
    removed_rows=subset(old,removed)
    for r in removed_rows:
        at=pd.Timestamp(r['decision_at']);r['query']=next(x for x in rows if x['at']==at)
    pred=read(run/'predictions.json');candidate={k:dict(Counter(e['action'] for e in es if '2026-01-01'<=e['date'][:10]<='2026-04-30')) for k,es in pred.items()}
    ready=[d for d in daily_rows if d['ready']]
    out={'post_result_only':True,'rules_changed_after_results':False,'first_meta_ready_date':ready[0]['date'],
        'active_dates':len(ready),'active_queries':int(frame.active.sum()),'active_known_queries':len(active),'active_unknown_retained':int((frame.active&~frame.known).sum()),
        'errors':error,'candidate_counts':candidate,'actual_added':subset(current,added),'actual_removed':removed_rows,
        'source_eligible_rejected':frame[frame.eligible&(frame.new_action=='CASH')].to_dict('records'),'daily_training_subsets':daily_rows,
        'source_eligible_quality_rows':frame[frame.active&frame.eligible].to_dict('records'),
        'quality_is_not_win_probability':True,'results_sha256':sha(run/'results.json')}
    dump(run/'diagnostics.json',out)
    print({k:v for k,v in out.items() if k not in ['daily_training_subsets','source_eligible_quality_rows','source_eligible_rejected','actual_removed']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
