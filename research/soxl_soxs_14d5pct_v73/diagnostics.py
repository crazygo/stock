"""Date-equal prediction and selected-entry diagnostics; no policy changes."""
import argparse
from normalized_learning import *


def diagnostics(run):
    run=Path(run);panel=pd.read_parquet(run/'semiconductor_panel.parquet');rows=[];kinds=VARIANTS+['old_quadratic','linear']
    for at,row in panel[panel.date.between('2026-01-01','2026-04-30')].iterrows():
        record=read(run/'training'/(tag(at)+'.json'));truth=np.array([row.C+row.D,row.C-row.D]);mature=pd.notna(row.terminal_release)
        item={'at':at,'date':row.date,'clock':row.clock,'mature':mature}
        for k in kinds:
            raw=record[k+'_raw'] if k in ['linear','old_quadratic'] else record['raws'][k]
            info=record[k+'_estimate'] if k in ['linear','old_quadratic'] else record['estimates'][k]
            item[k+'_raw_available']=raw is not None;item[k+'_cal_available']=info['ready']
            for stage,pred in [('raw',raw),('cal',info['point'])]:
                if pred is not None and mature:
                    j=int(pred[1]>pred[0]);item[k+'_'+stage+'_mse']=float(np.mean((np.asarray(pred)-truth)**2))
                    item[k+'_'+stage+'_top_regret']=float(max(truth)-truth[j])
            prefix=k if k in ['linear','old_quadratic'] else 'normalized_'+k
            for suffix in ['band','point']:item[k+'_'+suffix+'_action']=record['events'][prefix+'_'+suffix]['action']
        rows.append(item)
    table=pd.DataFrame(rows);table.to_csv(run/'diagnostic_rows.csv',index=False);errors={};differences={}
    for k in kinds:
        errors[k]={}
        for stage in ['raw','cal']:
            part=table.dropna(subset=[k+'_'+stage+'_mse']);daily=part.groupby('date')[[k+'_'+stage+'_mse',k+'_'+stage+'_top_regret']].mean()
            errors[k][stage]={'date_equal_mse':float(daily.iloc[:,0].mean()),'date_equal_top_regret':float(daily.iloc[:,1].mean()),
                              'dates':len(daily),'ticks':len(part),'available_ticks':int(table[k+'_'+stage+'_available'].sum())}
    for k in ['old_quadratic','linear','independent','pooled']:
        differences[k]={}
        for stage in ['raw','cal']:
            cols=['partial_'+stage+'_mse',k+'_'+stage+'_mse'];daily=table.dropna(subset=cols).groupby('date')[cols].mean();d=daily.iloc[:,0]-daily.iloc[:,1]
            differences[k][stage]={'matched_dates':len(d),'mean_mse_difference':float(d.mean()),'dates_partial_better':int(d.lt(0).sum())}
        for mode in ['band','point']:differences[k][mode+'_different_proposal_ticks']=int(table['partial_'+mode+'_action'].ne(table[k+'_'+mode+'_action']).sum())
    entries={}
    for name in ROUTES:
        try:t=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        if len(t):
            clocks=pd.to_datetime(t.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S')
            entries[name]={'actual_entries':len(t),'by_clock':clocks.value_counts().sort_index().to_dict(),'by_symbol':t.symbol.value_counts().to_dict()}
        else:entries[name]={'actual_entries':0,'by_clock':{},'by_symbol':{}}
    try:trades=pd.read_csv(run/'normalized_partial_band_trades.csv')
    except pd.errors.EmptyDataError:trades=pd.DataFrame()
    selected=[]
    for t in trades.itertuples():
        rec=read(run/'training'/(tag(pd.Timestamp(t.decision_at))+'.json'));j=core.ETFS.index(t.symbol);info=rec['estimates']['partial']
        selected.append({'decision_at':t.decision_at,'symbol':t.symbol,'raw_selected':rec['raws']['partial'][j],
                         'point_selected':info['point'][j],'lower_selected':info['lower'][j],
                         'actual_log':float(np.log1p(t.net_return)) if np.isfinite(t.net_return) else None,'net_return':t.net_return,'reason':t.reason})
    selection=pd.DataFrame(selected);selection.to_csv(run/'actual_entry_forecasts.csv',index=False)
    summary={'entries':len(selection),'completed':int(selection.actual_log.notna().sum()) if len(selection) else 0,
             'mean':selection[['raw_selected','point_selected','lower_selected','actual_log']].mean().to_dict() if len(selection) else {},
             'raw_selected_positive':int(selection.raw_selected.gt(0).sum()) if len(selection) else 0}
    out={'stage':'post_run_descriptive_only','errors':errors,'primary_minus_controls':differences,'entries':entries,
         'primary_actual_entry_forecasts':summary,'raw_top_regret_definition':'best of the two matured hypothetical log returns minus model-ranked side log return, before cash filter; date equal, not additional capital returns',
         'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),'changes_primary_or_gate':False,'goal_complete':False}
    dump(run/'diagnostics.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();diagnostics(a.run)
