"""Post-run date-equal paired prediction errors and separate actual executed entries."""
import argparse
from wait_learning import *
NEW=['paired_state','paired_score','paired_mean','absolute_pair','absolute_now']


def metrics(rows):
    known=rows[rows.ready & rows.mature];fields=['now_prediction','wait_prediction','advantage','now_actual','wait_actual','actual_advantage','now_squared_error','wait_squared_error','advantage_squared_error','source_squared_error','source_advantage_squared_error']
    daily=known.groupby('date')[fields].mean()
    return {'dates':len(daily),'known_rows':len(known),'rows':len(rows),'ready_rows':int(rows.ready.sum()),'unknown_rows':int((~rows.mature).sum()),
        'date_equal_means':daily.mean().to_dict() if len(daily) else {}}


def diagnostics(run):
    run=Path(run);settings=read(run/'registration.json')['preregistration']['protocol']['development'];panel=pd.read_parquet(run/'semiconductor_panel.parquet')
    bank={p['date'][:10]:p for p in read(run/'paired_registry.json')};records={};source={};rows=[]
    end=pd.Timestamp(settings['evaluation_end'],tz=core.ET).tz_convert('UTC')+pd.Timedelta(days=1)
    for at,row in panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])].iterrows():
        records[at]=read(run/'training'/(tag(at)+'.json'));source[at]=read(PARENT/'training'/(tag(at)+'.json'));src=source[at]['events']['recent_ratio']
        pair=bank.get(str(row.date.date()));mature=bool(pair and pair['release'] is not None and pd.Timestamp(pair['release'])<end)
        for name in ['paired_state','paired_score','paired_mean']:
            e=records[at]['events'][name];item={'at':at,'date':row.date,'clock':row.clock,'route':name,'ready':e['ready'],'mature':mature,
                'source_eligible':src['action'] in core.ETFS,'action':e['action'],'source_action':src['action']}
            if e['ready']:
                item.update(now_prediction=e['values'][0],wait_prediction=e['values'][1],advantage=e['advantage'],source_value=e['source_value'])
                if mature:
                    future=read(PARENT/'training'/(tag(pd.Timestamp(pair['next_at']))+'.json'))['events']['recent_ratio']['action'];j=core.ETFS.index(src['top_symbol'])
                    now=float(row['L_log' if j==0 else 'S_log']);wait=0. if future=='CASH' else float(panel.loc[pd.Timestamp(pair['next_at']),'L_log' if future=='SOXL' else 'S_log']);delta=now-wait
                    item.update(now_actual=now,wait_actual=wait,actual_advantage=delta,wait_action=future,wait_at=pair['next_at'],
                        now_squared_error=(e['values'][0]-now)**2,wait_squared_error=(e['values'][1]-wait)**2,advantage_squared_error=(e['advantage']-delta)**2,
                        source_squared_error=(e['source_value']-now)**2,source_advantage_squared_error=(e['source_value']-delta)**2)
            rows.append(item)
    table=pd.DataFrame(rows);table.to_csv(run/'diagnostic_rows.csv',index=False)
    scores={name:{'all_top':metrics(g),'source_eligible':metrics(g[g.source_eligible]),'accepted':metrics(g[g.action!='CASH'])} for name,g in table.groupby('route')}
    entries=[];counts={};summaries={}
    for name in ROUTES:
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        counts[name]={'entries':len(trades),'by_symbol':trades.symbol.value_counts().to_dict() if len(trades) else {},
            'by_clock':pd.to_datetime(trades.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S').value_counts().to_dict() if len(trades) else {}}
        if name not in NEW:continue
        items=[]
        for t in trades.itertuples():
            at=pd.Timestamp(t.decision_at);e=records[at]['events'][name];known=np.isfinite(t.net_return)
            item={'route':name,'at':at,'symbol':t.symbol,'reason':t.reason,'net_return':t.net_return,'known':known,
                'now_prediction':e['values'][0],'wait_prediction':e['values'][1],'advantage':e['advantage'],'source_value':e['source_value'],
                'actual_log_return':float(np.log1p(t.net_return)) if known else None}
            entries.append(item);items.append(item)
        known=[x for x in items if x['known']];summaries[name]={'entries':len(items),'completed':len(known),'pending':len(items)-len(known),
            'mean_now_prediction':float(np.mean([x['now_prediction'] for x in known])) if known else None,
            'mean_actual_log_return':float(np.mean([x['actual_log_return'] for x in known])) if known else None}
    pd.DataFrame(entries,columns=['route','at','symbol','reason','net_return','known','now_prediction','wait_prediction','advantage','source_value','actual_log_return']).to_csv(run/'actual_entry_forecasts.csv',index=False)
    differences={}
    main=[records[t]['events']['paired_state'] for t in records]
    for name in ['paired_score','paired_mean','absolute_pair','absolute_now','source_matched','source_recent']:
        other=[records[t]['events'][name] for t in records]
        differences[name]={'different_actions':sum(a['action']!=b['action'] for a,b in zip(main,other)),
            'main_buys':sum(e['action']!='CASH' for e in main),'other_buys':sum(e['action']!='CASH' for e in other)}
    out={'stage':'post_run_descriptive_only','scores':scores,'actual_entry_counts':counts,'actual_entry_summary':summaries,'action_comparisons':differences,
        'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),'no_policy_changes':True,'goal_complete':False}
    dump(run/'diagnostics.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);diagnostics(p.parse_args().run)
