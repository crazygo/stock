"""Date-equal prediction and actual-entry diagnostics; no policy/gate changes."""
import argparse
import gzip,json
from history_learning import *


def diagnostics(run):
    run=Path(run);p=read(run/'registration.json')['preregistration']['protocol'];settings=p['development']
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');rows=[];kinds=VARIANTS+['short_full','short_mean']
    for at,row in panel[panel.date.between(settings['evaluation_start'],settings['evaluation_end'])].iterrows():
        record=read(run/'training'/(tag(at)+'.json'));truth=np.array([row.C+row.D,row.C-row.D]);mature=pd.notna(row.terminal_release)
        item={'at':at,'date':row.date,'clock':row.clock,'mature':mature}
        for k in kinds:
            raw=record[k+'_raw'] if k.startswith('short_') and k!='short_price' else record['raws'][k]
            info=record[k+'_estimate'] if k.startswith('short_') and k!='short_price' else record['estimates'][k]
            item[k+'_raw_available']=raw is not None;item[k+'_cal_available']=info['ready']
            for stage,pred in [('raw',raw),('cal',info['point'])]:
                if pred is not None and mature:
                    j=int(pred[1]>pred[0]);item[k+'_'+stage+'_mse']=float(np.mean((np.asarray(pred)-truth)**2))
                    item[k+'_'+stage+'_top_regret']=float(max(truth)-truth[j])
            for suffix in ['band','point']:item[k+'_'+suffix+'_action']=record['events'][k+'_'+suffix]['action']
        rows.append(item)
    table=pd.DataFrame(rows);table.to_csv(run/'diagnostic_rows.csv',index=False);errors={};differences={}
    for k in kinds:
        errors[k]={}
        for stage in ['raw','cal']:
            part=table.dropna(subset=[k+'_'+stage+'_mse']);daily=part.groupby('date')[[k+'_'+stage+'_mse',k+'_'+stage+'_top_regret']].mean()
            errors[k][stage]={'date_equal_mse':float(daily.iloc[:,0].mean()),'date_equal_top_regret':float(daily.iloc[:,1].mean()),
                'dates':len(daily),'ticks':len(part),'available_ticks':int(table[k+'_'+stage+'_available'].sum())}
    for k in kinds:
        if k=='long_price':continue
        differences[k]={}
        for stage in ['raw','cal']:
            cols=['long_price_'+stage+'_mse',k+'_'+stage+'_mse'];daily=table.dropna(subset=cols).groupby('date')[cols].mean();d=daily.iloc[:,0]-daily.iloc[:,1]
            differences[k][stage]={'matched_dates':len(d),'mean_mse_difference':float(d.mean()),'dates_primary_better':int(d.lt(0).sum())}
        for mode in ['band','point']:differences[k][mode+'_different_proposal_ticks']=int(table['long_price_'+mode+'_action'].ne(table[k+'_'+mode+'_action']).sum())
    entries={};selected=[]
    for name in ROUTES:
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        if len(trades):
            clocks=pd.to_datetime(trades.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S')
            entries[name]={'actual_entries':len(trades),'by_clock':clocks.value_counts().sort_index().to_dict(),'by_symbol':trades.symbol.value_counts().to_dict()}
        else:entries[name]={'actual_entries':0,'by_clock':{},'by_symbol':{}}
        if name!=p['primary']:continue
        for t in trades.itertuples():
            rec=read(run/'training'/(tag(pd.Timestamp(t.decision_at))+'.json'));j=core.ETFS.index(t.symbol);info=rec['estimates']['long_price']
            selected.append({'decision_at':t.decision_at,'symbol':t.symbol,'raw_selected':rec['raws']['long_price'][j],
                'point_selected':info['point'][j],'lower_selected':info['lower'][j],
                'actual_log':float(np.log1p(t.net_return)) if np.isfinite(t.net_return) else None,'net_return':t.net_return,'reason':t.reason})
    selection=pd.DataFrame(selected);selection.to_csv(run/'actual_entry_forecasts.csv',index=False)
    masses=[]
    for date in sorted(set(table.date)):
        with gzip.open(run/'models'/(str(date.date())+'.json.gz'),'rt') as f:models=json.load(f)
        for clock,variants in models.items():
            for k,m in variants.items():masses.append({'date':date,'clock':clock,'variant':k,'real_dates':m['distinct_dates'],
                'reference_mass':m.get('reference_mass'),'penalty_scale':m.get('penalty_scale'),'date_ess':m['date_effective_sample_size']})
    mass=pd.DataFrame(masses);mass.to_csv(run/'sample_mass.csv',index=False)
    out={'stage':'post_run_descriptive_only','errors':errors,'primary_minus_controls':differences,'entries':entries,
        'sample_mass_range':{k:g[['real_dates','reference_mass','penalty_scale','date_ess']].agg(['min','max']).to_dict() for k,g in mass.groupby('variant')},
        'primary_actual_entry_forecasts':{'entries':len(selection),'completed':int(selection.actual_log.notna().sum()) if len(selection) else 0,
            'mean':selection[['raw_selected','point_selected','lower_selected','actual_log']].mean().to_dict() if len(selection) else {}},
        'raw_top_regret_definition':'best matured hypothetical side log return minus model-ranked side log return; date equal, not additional capital profit',
        'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),'changes_primary_or_gate':False,'goal_complete':False}
    dump(run/'diagnostics.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);diagnostics(p.parse_args().run)
