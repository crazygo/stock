"""Descriptive date-equal errors and actual entries; never selection criteria."""
import argparse
from transfer_learning import *


def diagnostics(run):
    run=Path(run);panel=pd.read_parquet(run/'semiconductor_panel.parquet')
    result=read(run/'results.json');rows=[]
    for at,row in panel[panel.date.between('2026-01-01','2026-04-30')].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'))
        item={'at':at,'date':row.date,'clock':row.clock,'mature':pd.notna(row.terminal_release)}
        truth=np.array([row.C+row.D,row.C-row.D])
        for kind in VARIANTS:
            raw=rec['raws'][kind];info=rec['estimates'][kind]
            item[kind+'_raw_available']=raw is not None;item[kind+'_cal_available']=info['ready']
            if item['mature'] and raw is not None:
                item[kind+'_raw_mse']=float(np.mean((np.array(raw)-truth)**2))
            if item['mature'] and info['ready']:
                item[kind+'_cal_mse']=float(np.mean((np.array(info['point'])-truth)**2))
            item[kind+'_band_action']=rec['events'][kind+'_band']['action']
            item[kind+'_point_action']=rec['events'][kind+'_point']['action']
        rows.append(item)
    table=pd.DataFrame(rows);table.to_csv(run/'diagnostic_rows.csv',index=False)
    errors={};increment={}
    for kind in VARIANTS:
        errors[kind]={}
        for stage in ['raw','cal']:
            col=kind+'_'+stage+'_mse';part=table[table[col].notna()]
            errors[kind][stage]={'date_equal_mse':float(part.groupby('date')[col].mean().mean()),
                               'dates':part.date.nunique(),'ticks':len(part)}
        errors[kind]['raw_available']=int(table[kind+'_raw_available'].sum())
        errors[kind]['cal_available']=int(table[kind+'_cal_available'].sum())
    for stage in ['raw','cal']:
        cols=[k+'_'+stage+'_mse' for k in ['partial','independent']];part=table.dropna(subset=cols)
        d=part.groupby('date')[cols].mean();diff=d.iloc[:,0]-d.iloc[:,1]
        increment[stage]={'matched_dates':len(d),'partial_minus_independent_mse':float(diff.mean()),
                          'dates_partial_lower_error':int((diff<0).sum())}
    for mode in ['band','point']:
        increment[mode+'_different_proposal_ticks']=int(table['partial_'+mode+'_action'].ne(table['independent_'+mode+'_action']).sum())
    entries={}
    for name in ROUTES:
        try:t=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        if len(t):
            clocks=pd.to_datetime(t.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S')
            entries[name]={'actual_entries':len(t),'by_clock':clocks.value_counts().sort_index().to_dict(),'by_symbol':t.symbol.value_counts().to_dict()}
        else:entries[name]={'actual_entries':0,'by_clock':{},'by_symbol':{}}
    actual=pd.read_csv(run/'partial_band_trades.csv');selected=[]
    for trade in actual.itertuples():
        record=read(run/'training'/(tag(pd.Timestamp(trade.decision_at))+'.json'))
        raw=record['raws']['partial'];info=record['estimates']['partial'];j=core.ETFS.index(trade.symbol)
        selected.append({'decision_at':trade.decision_at,'symbol':trade.symbol,
                         'raw_long':raw[0],'raw_short':raw[1],'raw_selected':raw[j],
                         'point_selected':info['point'][j],'lower_selected':info['lower'][j],
                         'radius':info['radius'],'actual_log':float(np.log1p(trade.net_return)),
                         'net_return':trade.net_return,'exit_reason':trade.reason})
    selection=pd.DataFrame(selected);selection.to_csv(run/'actual_entry_forecasts.csv',index=False)
    selected_summary={'entries':len(selection),
                      'mean':selection[['raw_selected','point_selected','lower_selected','actual_log']].mean().to_dict(),
                      'raw_selected_positive':int(selection.raw_selected.gt(0).sum()),
                      'not_an_independent_calibration_test':True}
    out={'stage':'post_run_descriptive_only','scope':'Jan-Apr development, complete14day labels only for prediction errors; all ticks retained in capital.',
         'errors':errors,'sharing_increment':increment,'entries':entries,'primary_actual_entry_forecasts':selected_summary,
         'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),
         'changes_primary_or_gate':False,'goal_complete':False}
    dump(run/'diagnostics.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();diagnostics(a.run)
