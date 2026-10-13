"""Post-run probability and return diagnostics on common and selected populations."""
import argparse
from state_calibration import *
NEW=list(MODELS)+['identity_matched']
FIELDS_SCORE=['value','actual','squared_error','brier','log_loss']+[f'q{k}' for k in range(4)]+[f'y{k}' for k in range(4)]


def metrics(rows):
    known=rows[rows.ready & rows.mature];daily=known.groupby('date')[FIELDS_SCORE].mean()
    return dict(rows=len(rows),ready_rows=int(rows.ready.sum()),known_rows=len(known),unknown_rows=int((~rows.mature).sum()),
                known_dates=len(daily),outside_rows=int(rows.outside.sum()),date_equal_means=daily.mean().to_dict() if len(daily) else {})


def diagnostics(run):
    run=Path(run);assert read(run/'verification.json')['passed'];p=read(SOURCE/'protocol.json');s=p['development']
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');test=panel[panel.date.between(s['evaluation_start'],s['evaluation_end'])]
    end=pd.Timestamp(s['evaluation_end'],tz=core.ET)+pd.Timedelta(days=1);records={};rows=[]
    for at,r in test.iterrows():
        saved=read(run/'training'/(tag(at)+'.json'));records[at]=saved
        source=read(PARENT/'training'/(tag(at)+'.json'))['events']['recent_ratio']
        mature=bool(pd.notna(r.terminal_release) and r.terminal_release<end)
        for name in NEW:
            e=saved['events'][name]
            item=dict(at=at,date=r.date,clock=r.clock,route=name,ready=e['ready'],mature=mature,action=e['action'],source_action=source['action'],
                      source_eligible=source['action'] in core.ETFS,outside=bool(e.get('outside_state_fields')))
            if e['ready']:
                item.update(value=e['value'],symbol=e['top_symbol'])
                item.update({f'q{k}':e['probabilities'][k] for k in range(4)})
                if mature:
                    side='L' if e['top_symbol']=='SOXL' else 'S';y=int(r[side+'_gap']);actual=float(r[side+'_log']);assert 0<=y<4 and np.isfinite(actual)
                    q=np.array(e['probabilities']);item.update(truth=y,actual=actual,squared_error=(e['value']-actual)**2,
                        brier=float(sum((q-np.eye(4)[y])**2)),log_loss=float(-np.log(max(q[y],1e-8))))
                    item.update({f'y{k}':int(y==k) for k in range(4)})
            rows.append(item)
    table=pd.DataFrame(rows);table.to_csv(run/'diagnostic_rows.csv',index=False)
    scores={name:{'all_top':metrics(g),'source_eligible':metrics(g[g.source_eligible]),'accepted':metrics(g[g.action!='CASH'])} for name,g in table.groupby('route')}
    entries=[];counts={};summaries={}
    for name in ROUTES:
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        counts[name]=dict(entries=len(trades),by_symbol=trades.symbol.value_counts().to_dict() if len(trades) else {},
                         by_clock=pd.to_datetime(trades.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S').value_counts().to_dict() if len(trades) else {})
        if name not in NEW:continue
        items=[]
        for t in trades.itertuples():
            at=pd.Timestamp(t.decision_at);e=records[at]['events'][name];known=np.isfinite(t.net_return)
            truth=0 if t.reason=='target' else (2 if t.gap_below_stop else 1) if t.reason=='risk_stop' else 3 if known else None
            item=dict(route=name,at=at,symbol=t.symbol,reason=t.reason,truth=truth,net_return=t.net_return,known=known,value=e['value'],
                      actual_log_return=float(np.log1p(t.net_return)) if known else None,outside=bool(e['outside_state_fields']))
            item.update({f'q{k}':float(e['probabilities'][k]) for k in range(4)});entries.append(item);items.append(item)
        known=[x for x in items if x['known']]
        summaries[name]=dict(entries=len(items),completed=len(known),pending=len(items)-len(known),outside_entries=sum(x['outside'] for x in items),
            mean_value=float(np.mean([x['value'] for x in known])) if known else None,mean_actual=float(np.mean([x['actual_log_return'] for x in known])) if known else None,
            class_probabilities=[float(np.mean([x[f'q{k}'] for x in known])) for k in range(4)] if known else None,
            class_counts=[sum(x['truth']==k for x in known) for k in range(4)])
    pd.DataFrame(entries,columns=['route','at','symbol','reason','truth','net_return','known','value','actual_log_return','outside','q0','q1','q2','q3']).to_csv(run/'actual_entry_forecasts.csv',index=False)
    comparisons={};primary=[records[t]['events']['eligible_state'] for t in records]
    for name in ['eligible_clock','eligible_bias','top_state','identity_matched','source_recent','source_wait']:
        other=[records[t]['events'][name] for t in records]
        comparisons[name]=dict(different_actions=sum(a['action']!=b['action'] for a,b in zip(primary,other)),main_buys=sum(a['action']!='CASH' for a in primary),other_buys=sum(b['action']!='CASH' for b in other))
    daily=[]
    for date in sorted(test.date.unique()):
        d=read(run/'daily'/(str(pd.Timestamp(date).date())+'.json'))
        daily.append(dict(date=date,ready=d['ready'],common_dates=len(d['parents']),eligible_dates=d['models']['eligible_state']['real_dates'],eligible_date_ess=d['models']['eligible_state']['date_ess']))
    out=dict(stage='post_run_descriptive_only',scores=scores,actual_entry_counts=counts,actual_entry_summary=summaries,action_comparisons=comparisons,daily_readiness=daily,
             results_sha256=sha(run/'results.json'),source_sha256=sha(Path(__file__)),no_policy_changes=True,goal_complete=False)
    dump(run/'diagnostics.json',out);print({'scores':scores,'actual_entry_summary':summaries,'action_comparisons':comparisons},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);diagnostics(p.parse_args().run)
