"""Date-equal scores for both sides, source tops and eligible candidates; separate actual fills."""
import argparse
from shift_learning import *


def metrics(g):
    known=g[g.ready & g.mature];fields=['brier','log_loss','predicted_log_return','actual_log_return']+[f'p{k}' for k in range(4)]+[f'y{k}' for k in range(4)]
    daily=known.groupby('date')[fields].mean()
    return {'dates':len(daily),'known_rows':len(known),'rows':len(g),'ready_rows':int(g.ready.sum()),'unknown_rows':int((~g.mature).sum()),
        'date_equal_means':{k:float(daily[k].mean()) for k in fields} if len(daily) else {}}


def diagnostics(run):
    run=Path(run);p=read(run/'registration.json')['preregistration']['protocol'];panel=pd.read_parquet(run/'semiconductor_panel.parquet')
    settings=p['development'];table=[];models=[];recordcache={};previous=None
    for at,row in panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));old=read(BASE/'training'/(tag(at)+'.json'));recordcache[at]=rec
        src=old['events']['gap_blend'];info=old['estimates']['gap_blend'];top=int(info['values'][1]>info['values'][0]) if info['ready'] else None
        if row.date!=previous:
            d=read(run/'daily'/(str(row.date.date())+'.json'))
            for mode,m in d['variants'].items():
                dm=m.get('domain');fit=m.get('model',{})
                models.append({'date':row.date,'mode':mode,'ready':m['ready'],
                    'date_ess':m.get('weighted_date_ess'),'original_mass':m.get('original_weight_sum'),
                    'target_dates':len(m['window']['dates']),'overlap_dates':m.get('overlap_dates'),
                    'ratio_min':min(dm['ratios']) if dm else None,'ratio_max':max(dm['ratios']) if dm else None,
                    'clipped_low':dm['clipped_low'] if dm else None,'clipped_high':dm['clipped_high'] if dm else None,
                    **{f'beta{k}':fit['beta'][k] if fit.get('ready') else None for k in range(4)}})
            previous=row.date
        for name in ['recent_ratio','lagged_ratio','uniform_top']:
            e=rec['events'][name]
            for j,side in enumerate(['L','S']):
                item={'at':at,'date':row.date,'clock':row.clock,'route':name,'side':side,'ready':e['ready'],'mature':pd.notna(row.terminal_release),
                    'source_top':j==top,'source_eligible_top':j==top and src['action'] in core.ETFS,'action':e['action']}
                if e['ready']:
                    q=np.array(e['probabilities'][j]);item.update(predicted_log_return=e['values'][j],**{f'p{k}':q[k] for k in range(4)})
                    if item['mature']:
                        y=int(row[side+'_gap']);item.update(label=y,actual_log_return=row[side+'_log'],brier=float(sum((q-np.eye(4)[y])**2)),
                            log_loss=float(-np.log(max(q[y],1e-8))),**{f'y{k}':int(y==k) for k in range(4)})
                table.append(item)
    t=pd.DataFrame(table);t.to_csv(run/'diagnostic_rows.csv',index=False);pd.DataFrame(models).to_csv(run/'calibrator_history.csv',index=False)
    scores={name:{'both':metrics(g),'top':metrics(g[g.source_top]),'eligible':metrics(g[g.source_eligible_top])} for name,g in t.groupby('route')}
    actual=[];counts={}
    for name in ROUTES:
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        counts[name]={'entries':len(trades),'by_symbol':trades.symbol.value_counts().to_dict() if len(trades) else {},
            'by_clock':pd.to_datetime(trades.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S').value_counts().to_dict() if len(trades) else {}}
        if name not in ['recent_ratio','lagged_ratio','uniform_top']:continue
        for trade in trades.itertuples():
            at=pd.Timestamp(trade.decision_at);j=core.ETFS.index(trade.symbol);e=recordcache[at]['events'][name];known=np.isfinite(trade.net_return)
            cls=(0 if trade.reason=='target' else (2 if trade.gap_below_stop else 1) if trade.reason=='risk_stop' else 3) if known else None
            old=read(BASE/'training'/(tag(at)+'.json'));source=old['estimates']['gap_blend'];q=np.array(e['probabilities'][j])
            actual.append({'route':name,'at':at,'symbol':trade.symbol,'known':known,'class':cls,'reason':trade.reason,'net_return':trade.net_return,
                'predicted_log_return':e['values'][j],'source_predicted_log_return':source['values'][j],
                'actual_log_return':float(np.log1p(trade.net_return)) if known else None,
                **{f'p{k}':q[k] for k in range(4)},**{f'source_p{k}':source['probabilities'][j][k] for k in range(4)}})
    entries=pd.DataFrame(actual);entries.to_csv(run/'actual_entry_forecasts.csv',index=False)
    summaries={}
    if len(entries):
        for name,g in entries.groupby('route'):
            known=g[g.known];summaries[name]={'entries':len(g),'completed':len(known),'pending':int((~g.known).sum()),
                'class_counts':known['class'].value_counts().to_dict(),'mean_predicted_log_return':float(known.predicted_log_return.mean()) if len(known) else None,
                'mean_actual_log_return':float(known.actual_log_return.mean()) if len(known) else None}
    differences={}
    preds=read(run/'predictions.json');main=[e for e in preds['recent_ratio'] if settings['evaluation_start']<=e['date'][:10]<=settings['evaluation_end']]
    for name in ['uniform_top','lagged_ratio','source_top','source_blend']:
        other=[e for e in preds[name] if settings['evaluation_start']<=e['date'][:10]<=settings['evaluation_end']]
        differences[name]={'different_actions':sum(a['action']!=b['action'] for a,b in zip(main,other)),
            'main_buys':sum(e['action']!='CASH' for e in main),'other_buys':sum(e['action']!='CASH' for e in other)}
    out={'stage':'post_run_descriptive_only','scores':scores,'actual_entry_counts':counts,'actual_entry_summary':summaries,'action_comparisons':differences,
        'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),'no_policy_changes':True,'goal_complete':False}
    dump(run/'diagnostics.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);diagnostics(p.parse_args().run)
