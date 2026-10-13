"""Proper probability scores, log-payoff errors and capital-subset diagnostics."""
import argparse
from class_learning import *


def diagnostics(run):
    run=Path(run);result=read(run/'results.json');p=read(run/'registration.json')['preregistration']['protocol'];settings=p['development']
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');rows=[];mix=[]
    for at,row in panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));mature=pd.notna(row.terminal_release)
        for mode in MODES:
            dist=rec['distributions'][mode]
            for variant in ['blend','raw','prior','equal']:
                name=mode+'_'+variant;info=rec['estimates'][name]
                for j,side in enumerate(['L','S']):
                    item={'at':at,'date':row.date,'clock':row.clock,'mode':mode,'route':name,'side':side,'mature':mature,'ready':info['ready'],
                        'action':rec['events'][name]['action']}
                    if info['ready']:
                        prob=np.array(info['probabilities'][j]);item.update(predicted_log_return=info['values'][j],alpha=info['alpha'],
                            target_probability=prob[0],gap_probability=prob[2] if mode=='gap' else None)
                        if mature:
                            y=int(row[side+'_'+mode]);truth=float(row[side+'_log']);common=int(row[side+'_merged'])
                            commonp=np.array([prob[0],prob[1]+prob[2],prob[3]]) if mode=='gap' else prob
                            item.update(label=y,actual_log_return=truth,brier=float(sum((prob-np.eye(len(prob))[y])**2)),
                                log_loss=float(-np.log(max(prob[y],1e-8))),common_three_brier=float(sum((commonp-np.eye(3)[common])**2)),
                                payoff_squared_error=float((info['values'][j]-truth)**2),target_actual=int(y==0))
                    rows.append(item)
        if row.clock==CLOCKS[0]:
            day=read(run/'daily'/(str(row.date.date())+'.json'))
            for mode in MODES:
                d=day['mixture'][mode];mix.append({'date':row.date,'mode':mode,'ready':d['ready'],'dates':len(d['parents']),
                    'alpha':d['alpha'],'logodds':d.get('logodds')})
    table=pd.DataFrame(rows);table.to_csv(run/'diagnostic_rows.csv',index=False);weights=pd.DataFrame(mix);weights.to_csv(run/'mixture_weights.csv',index=False)
    fields=['brier','log_loss','common_three_brier','payoff_squared_error'];scores={}
    for name,g in table.groupby('route'):
        known=g[g.mature&g.ready];daily=known.groupby('date')[fields].mean()
        scores[name]={'dates':len(daily),'etf_clock_rows':len(known),'ready_rows':int(g.ready.sum()),'unknown_rows':int((~g.mature).sum()),
            **{f:float(daily[f].mean()) for f in fields}}
    entries={};selected=[]
    for name in ROUTES:
        try:t=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        entries[name]={'entries':len(t),'by_symbol':t.symbol.value_counts().to_dict() if len(t) else {},
            'by_clock':pd.to_datetime(t.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S').value_counts().sort_index().to_dict() if len(t) else {}}
        if name!=p['primary']:continue
        for trade in t.itertuples():
            at=pd.Timestamp(trade.decision_at);j=core.ETFS.index(trade.symbol);rec=read(run/'training'/(tag(at)+'.json'));info=rec['estimates']['gap_blend']
            selected.append({'decision_at':at,'symbol':trade.symbol,'predicted_log_return':info['values'][j],
                'actual_log_return':float(np.log1p(trade.net_return)) if np.isfinite(trade.net_return) else None,
                'target_probability':info['probabilities'][j][0],'gap_probability':info['probabilities'][j][2],
                'alpha':info['alpha'],'net_return':trade.net_return,'reason':trade.reason,'gap_below_stop':trade.gap_below_stop})
    chosen=pd.DataFrame(selected);chosen.to_csv(run/'actual_entry_forecasts.csv',index=False)
    output={'stage':'post_run_descriptive_only','scores':scores,'entries':entries,
        'mixture_summary':{mode:{'dates':len(g),'alpha_min':float(g.alpha.min()),'alpha_max':float(g.alpha.max()),'alpha_mean':float(g.alpha.mean())} for mode,g in weights.groupby('mode')},
        'actual_primary_entries':{'entries':len(chosen),'completed':int(chosen.actual_log_return.notna().sum()) if len(chosen) else 0,
            'means':chosen[['predicted_log_return','actual_log_return','target_probability','gap_probability','alpha']].mean().to_dict() if len(chosen) else {}},
        'four_vs_three_brier_not_directly_comparable':True,'common_three_brier_compares_same_aggregated_target_stop_expiry_classes':True,
        'unknown_labels_retained':True,'date_equal_metrics_not_independent_rows':True,'results_sha256':sha(run/'results.json'),
        'source_sha256':sha(Path(__file__)),'changes_primary_or_gate':False,'goal_complete':False}
    dump(run/'diagnostics.json',output);print(output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);diagnostics(p.parse_args().run)
