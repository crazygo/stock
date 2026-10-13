"""Independently recompute descriptive prediction errors and actual-entry summaries."""
import argparse
import math
import gzip,json
from history_learning import *


def audit(run):
    run=Path(run);report=read(run/'diagnostics.json');panel=pd.read_parquet(run/'semiconductor_panel.parquet')
    table=pd.read_csv(run/'diagnostic_rows.csv');kinds=VARIANTS+['short_full','short_mean'];data={};checks=0
    for item in table.itertuples():
        at=pd.Timestamp(item.at);row=panel.loc[at];record=read(run/'training'/(tag(at)+'.json'))
        truth=[row.C+row.D,row.C-row.D];mature=pd.notna(row.terminal_release)
        assert bool(item.mature)==mature
        for k in kinds:
            raw=record['raws'][k] if k in VARIANTS else record[k+'_raw']
            info=record['estimates'][k] if k in VARIANTS else record[k+'_estimate']
            for stage,values,available in [('raw',raw,raw is not None),('cal',info['point'],info['ready'])]:
                assert getattr(item,k+'_'+stage+'_available')==available
                data.setdefault((k,stage),[])
                if values is not None and mature:
                    mse=sum((values[j]-truth[j])**2 for j in range(2))/2
                    selected=1 if values[1]>values[0] else 0;regret=max(truth)-truth[selected]
                    np.testing.assert_allclose([getattr(item,k+'_'+stage+'_mse'),getattr(item,k+'_'+stage+'_top_regret')],[mse,regret],atol=1e-14,rtol=1e-11)
                    data[k,stage].append((str(row.date.date()),mse,regret));checks+=2
            for mode in ['band','point']:assert getattr(item,k+'_'+mode+'_action')==record['events'][k+'_'+mode]['action']
    for (k,stage),values in data.items():
        days={}
        for date,mse,regret in values:days.setdefault(date,[]).append((mse,regret))
        daily={date:[sum(row[j] for row in rs)/len(rs) for j in [0,1]] for date,rs in days.items()}
        summary=report['errors'][k][stage]
        assert summary['dates']==len(days) and summary['ticks']==len(values)
        assert summary['available_ticks']==int(table[k+'_'+stage+'_available'].sum())
        np.testing.assert_allclose([summary['date_equal_mse'],summary['date_equal_top_regret']],
            [sum(v[j] for v in daily.values())/len(daily) for j in [0,1]],atol=1e-14,rtol=1e-11)
    for k,summary in report['primary_minus_controls'].items():
        for stage in ['raw','cal']:
            a={};b={}
            for date,mse,_ in data['long_price',stage]:a.setdefault(date,[]).append(mse)
            for date,mse,_ in data[k,stage]:b.setdefault(date,[]).append(mse)
            dates=sorted(set(a)&set(b));diff=[sum(a[d])/len(a[d])-sum(b[d])/len(b[d]) for d in dates]
            got=summary[stage];assert got['matched_dates']==len(dates) and got['dates_primary_better']==sum(x<0 for x in diff)
            np.testing.assert_allclose(got['mean_mse_difference'],sum(diff)/len(diff),atol=1e-14,rtol=1e-11)
        for mode in ['band','point']:
            assert summary[mode+'_different_proposal_ticks']==sum(getattr(r,'long_price_'+mode+'_action')!=getattr(r,k+'_'+mode+'_action') for r in table.itertuples())
    primary=read(run/'registration.json')['preregistration']['protocol']['primary'];entrychecks=0
    for route in ROUTES:
        try:trades=pd.read_csv(run/(route+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        clocks={};symbols={}
        for trade in trades.itertuples():
            c=pd.Timestamp(trade.decision_at).tz_convert(core.ET).strftime('%H:%M:%S');clocks[c]=clocks.get(c,0)+1
            symbols[trade.symbol]=symbols.get(trade.symbol,0)+1
        assert report['entries'][route]=={'actual_entries':len(trades),'by_clock':clocks,'by_symbol':symbols}
        if route!=primary:continue
        try:entryrows=pd.read_csv(run/'actual_entry_forecasts.csv')
        except pd.errors.EmptyDataError:entryrows=pd.DataFrame()
        assert len(entryrows)==len(trades)
        for i,t in enumerate(trades.itertuples()):
            r=entryrows.iloc[i];rec=read(run/'training'/(tag(pd.Timestamp(t.decision_at))+'.json'));j=core.ETFS.index(t.symbol)
            assert r.decision_at==t.decision_at and r.symbol==t.symbol and r.reason==t.reason
            np.testing.assert_allclose([r.raw_selected,r.point_selected,r.lower_selected,r.net_return,r.actual_log],
                [rec['raws']['long_price'][j],rec['estimates']['long_price']['point'][j],rec['estimates']['long_price']['lower'][j],t.net_return,math.log1p(t.net_return)],atol=1e-14,rtol=1e-11)
            entrychecks+=1
        s=report['primary_actual_entry_forecasts'];assert s['entries']==len(trades)
        assert s['completed']==(int(entryrows.actual_log.notna().sum()) if len(entryrows) else 0)
        for k,v in s['mean'].items():np.testing.assert_allclose(v,entryrows[k].mean(),atol=1e-14,rtol=1e-11)
    mass=pd.read_csv(run/'sample_mass.csv');masschecks=0
    for date,rows in mass.groupby('date'):
        with gzip.open(run/'models'/(date[:10]+'.json.gz'),'rt') as f:models=json.load(f)
        for row in rows.itertuples():
            model=models[row.clock][row.variant]
            assert row.real_dates==model['distinct_dates']
            np.testing.assert_allclose(row.date_ess,model['date_effective_sample_size'],atol=1e-10,rtol=1e-10)
            for field in ['reference_mass','penalty_scale']:
                if field in model:np.testing.assert_allclose(getattr(row,field),model[field],atol=1e-12,rtol=1e-12)
                else:assert pd.isna(getattr(row,field))
            masschecks+=1
    for kind,rows in mass.groupby('variant'):
        for field in ['real_dates','reference_mass','penalty_scale','date_ess']:
            v=[x for x in rows[field] if np.isfinite(x)];got=report['sample_mass_range'][kind][field]
            if v:np.testing.assert_allclose([got['min'],got['max']],[min(v),max(v)],atol=1e-10,rtol=1e-10)
            else:assert got=={'min':None,'max':None}
    results=read(run/'results.json');primary_eq=pd.read_csv(run/(primary+'_equity.csv'));pairedchecks=0
    def increments(eq):
        nav=[1.,*eq.equity.tolist()]
        return [math.log(nav[i+1]/nav[i]) for i in range(len(eq))]
    main=increments(primary_eq)
    for name,got in results['paired_increments'].items():
        eq=pd.read_csv(run/(name+'_equity.csv'));assert list(eq.date)==list(primary_eq.date)
        other=increments(eq);delta=[x-y for x,y in zip(main,other)];n=len(delta)
        draws=np.random.default_rng(707).integers(0,n,size=(5000,(n+9)//10));boot=[]
        for draw in draws:
            ids=[(int(start)+j)%n for start in draw for j in range(10)][:n]
            boot.append(sum(delta[j] for j in ids)/n)
        boot.sort()
        def percentile(q):
            rank=q*(len(boot)-1);low=int(rank);fraction=rank-low
            return boot[low]*(1-fraction)+boot[min(low+1,len(boot)-1)]*fraction
        bounds=[percentile(.025),percentile(.975)]
        np.testing.assert_allclose(got['interval95'],bounds,atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(got['mean_daily_log_difference'],sum(delta)/n,atol=1e-13,rtol=1e-10)
        assert got['dates']==n and got['seed']==707 and got['block_sessions']==10 and got['replicates']==5000
        assert got['positive_increment_supported']==(bounds[0]>0);pairedchecks+=1
    assert report['results_sha256']==sha(run/'results.json') and not report['changes_primary_or_gate']
    result={'passed':True,'error_values_checked':checks,'actual_entry_forecasts_checked':entrychecks,
        'diagnostics_sha256':sha(run/'diagnostics.json'),'source_sha256':sha(Path(__file__)),
        'sample_mass_rows_checked':masschecks,'paired_bootstrap_comparisons_checked':pairedchecks,
        'sample_mass_model_values_checked_separately_by_daily_model_verifier':True,'policy_changes':False}
    dump(run/'diagnostic_audit.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);audit(p.parse_args().run)
