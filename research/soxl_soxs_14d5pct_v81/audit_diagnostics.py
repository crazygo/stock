"""Independent paired-target arithmetic, score populations, fills and capital intervals."""
import argparse,math
from wait_learning import *
import reference_wait as ref


def audit(run):
    run=Path(run);report=read(run/'diagnostics.json');result=read(run/'results.json');table=pd.read_csv(run/'diagnostic_rows.csv')
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');market=extended_market();records={at:read(PARENT/'training'/(tag(at)+'.json')) for at in panel.index}
    bank={p['date']:p for p in ref.bank(panel,market,records,CLOCKS)};saved={};buckets={};checks=0
    end=pd.Timestamp(read(SOURCE/'protocol.json')['development']['evaluation_end'],tz=core.ET)+pd.Timedelta(days=1)
    fields=['now_prediction','wait_prediction','advantage','now_actual','wait_actual','actual_advantage','now_squared_error','wait_squared_error','advantage_squared_error','source_squared_error','source_advantage_squared_error']
    for row in table.itertuples():
        at=pd.Timestamp(row.at);truth=panel.loc[at];src=records[at]['events']['recent_ratio']
        if at not in saved:saved[at]=read(run/'training'/(tag(at)+'.json'))
        e=saved[at]['events'][row.route];pair=bank.get(truth.date);mature=bool(pair and pair['release'] is not None and pair['release']<end)
        assert row.ready==e['ready'] and row.mature==mature and row.action==e['action'] and row.source_action==src['action']
        assert row.source_eligible==(src['action'] in core.ETFS)
        if row.ready:
            np.testing.assert_allclose([row.now_prediction,row.wait_prediction,row.advantage,row.source_value],
                [e['values'][0],e['values'][1],e['values'][0]-e['values'][1],e['source_value']],atol=1e-13,rtol=1e-10)
        if row.ready and row.mature:
            symbol=records[pair['next_at']]['events']['recent_ratio']['action'];j=0 if src['top_symbol']=='SOXL' else 1
            now=float(truth['L_log' if j==0 else 'S_log']);wait=0 if symbol=='CASH' else float(panel.loc[pair['next_at'],'L_log' if symbol=='SOXL' else 'S_log']);delta=now-wait
            assert row.wait_action==symbol and pd.Timestamp(row.wait_at)==pair['next_at']
            values={'now_prediction':e['values'][0],'wait_prediction':e['values'][1],'advantage':e['values'][0]-e['values'][1],
                'now_actual':now,'wait_actual':wait,'actual_advantage':delta,'now_squared_error':(e['values'][0]-now)**2,
                'wait_squared_error':(e['values'][1]-wait)**2,'advantage_squared_error':(e['values'][0]-e['values'][1]-delta)**2,
                'source_squared_error':(e['source_value']-now)**2,'source_advantage_squared_error':(e['source_value']-delta)**2}
            for k,v in values.items():np.testing.assert_allclose(getattr(row,k),v,atol=1e-13,rtol=1e-10);checks+=1
            for pop,include in [('all_top',True),('source_eligible',row.source_eligible),('accepted',row.action!='CASH')]:
                if include:buckets.setdefault((row.route,pop),{}).setdefault(row.date,[]).append(values)
    for name,pops in report['scores'].items():
        g=table[table.route==name]
        for pop,score in pops.items():
            selected=g if pop=='all_top' else g[g.source_eligible] if pop=='source_eligible' else g[g.action!='CASH'];known=selected[selected.ready & selected.mature];days=buckets.get((name,pop),{})
            assert score['rows']==len(selected) and score['ready_rows']==int(selected.ready.sum()) and score['unknown_rows']==int((~selected.mature).sum())
            assert score['dates']==len(days) and score['known_rows']==len(known)
            for k,v in score['date_equal_means'].items():np.testing.assert_allclose(v,sum(sum(x[k] for x in a)/len(a) for a in days.values())/len(days),atol=1e-13,rtol=1e-10)
    actual=pd.read_csv(run/'actual_entry_forecasts.csv');entrychecks=0
    for name in ROUTES:
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        names={};clocks={}
        for t in trades.itertuples():
            names[t.symbol]=names.get(t.symbol,0)+1;c=pd.Timestamp(t.decision_at).tz_convert(core.ET).strftime('%H:%M:%S');clocks[c]=clocks.get(c,0)+1
        assert report['actual_entry_counts'][name]=={'entries':len(trades),'by_symbol':names,'by_clock':clocks}
        if name not in report['actual_entry_summary']:continue
        rows=actual[actual.route==name].reset_index(drop=True);assert len(rows)==len(trades)
        for i,t in enumerate(trades.itertuples()):
            got=rows.iloc[i];at=pd.Timestamp(t.decision_at);e=saved[at]['events'][name]
            assert pd.Timestamp(got['at'])==at and got.symbol==t.symbol and got.reason==t.reason and got.known==np.isfinite(t.net_return)
            np.testing.assert_allclose([got.now_prediction,got.wait_prediction,got.advantage,got.source_value],[*e['values'],e['advantage'],e['source_value']],atol=1e-13)
            if got.known:np.testing.assert_allclose([got.net_return,got.actual_log_return],[t.net_return,math.log1p(t.net_return)],atol=1e-13)
            entrychecks+=1
        known=rows[rows.known] if len(rows) else rows;s=report['actual_entry_summary'][name]
        assert s['entries']==len(rows) and s['completed']==len(known) and s['pending']==len(rows)-len(known)
        if len(known):np.testing.assert_allclose([s['mean_now_prediction'],s['mean_actual_log_return']],[known.now_prediction.mean(),known.actual_log_return.mean()],atol=1e-13)
        else:assert s['mean_now_prediction'] is None and s['mean_actual_log_return'] is None
    main=[saved[t]['events']['paired_state']['action'] for t in saved]
    for name,out in report['action_comparisons'].items():
        other=[saved[t]['events'][name]['action'] for t in saved]
        assert out=={'different_actions':sum(a!=b for a,b in zip(main,other)),'main_buys':sum(x!='CASH' for x in main),'other_buys':sum(x!='CASH' for x in other)}
    primary=pd.read_csv(run/'paired_state_equity.csv');m=[1.,*primary.equity];paired=0
    for name,out in result['paired_increments'].items():
        other=pd.read_csv(run/(name+'_equity.csv'));assert other.date.tolist()==primary.date.tolist();o=[1.,*other.equity];n=len(primary)
        delta=[math.log(m[i+1]/m[i])-math.log(o[i+1]/o[i]) for i in range(n)];starts=np.random.default_rng(707).integers(0,n,size=(5000,(n+9)//10));values=[]
        for blocks in starts:
            ids=[(int(s)+j)%n for s in blocks for j in range(10)][:n];values.append(sum(delta[i] for i in ids)/n)
        np.testing.assert_allclose(out['interval95'],np.quantile(values,[.025,.975]),atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(out['mean_daily_log_difference'],sum(delta)/n,atol=1e-13);paired+=1
    assert report['results_sha256']==sha(run/'results.json') and report['no_policy_changes']
    out={'passed':True,'prediction_values_checked':checks,'actual_entry_forecasts_checked':entrychecks,'paired_bootstrap_comparisons_checked':paired,
        'diagnostics_sha256':sha(run/'diagnostics.json'),'source_sha256':sha(Path(__file__)),'no_policy_changes':True};dump(run/'diagnostic_audit.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);audit(p.parse_args().run)
