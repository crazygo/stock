"""Separate arithmetic audit for populations, actual fills, and paired capital intervals."""
import argparse,math
from tilt_learning import *


def audit(run):
    run=Path(run);report=read(run/'diagnostics.json');result=read(run/'results.json');table=pd.read_csv(run/'diagnostic_rows.csv')
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');records={};oldrecords={};buckets={};checks=0
    fields=['brier','log_loss','predicted_log_return','actual_log_return']+[f'p{k}' for k in range(4)]+[f'y{k}' for k in range(4)]
    for row in table.itertuples():
        at=pd.Timestamp(row.at);j=0 if row.side=='L' else 1
        if at not in records:
            records[at]=read(run/'training'/(tag(at)+'.json'));oldrecords[at]=read(PARENT/'training'/(tag(at)+'.json'))
        e=records[at]['events'][row.route];src=oldrecords[at]['events']['gap_blend'];info=oldrecords[at]['estimates']['gap_blend']
        winner=int(info['values'][1]>info['values'][0]) if info['ready'] else None;truth=panel.loc[at]
        assert row.ready==e['ready'] and row.mature==pd.notna(truth.terminal_release) and row.action==e['action']
        assert row.source_top==(j==winner) and row.source_eligible_top==(j==winner and src['action'] in core.ETFS)
        if e['ready']:
            q=e['probabilities'][j]
            np.testing.assert_allclose([getattr(row,f'p{k}') for k in range(4)],q,atol=1e-13,rtol=1e-11)
            np.testing.assert_allclose(row.predicted_log_return,e['values'][j],atol=1e-13,rtol=1e-11)
        if row.ready and row.mature:
            y=int(truth[row.side+'_gap']);actual=float(truth[row.side+'_log'])
            values={'brier':sum(v*v for v in q)+1-2*q[y],'log_loss':-math.log(max(q[y],1e-8)),
                'predicted_log_return':e['values'][j],'actual_log_return':actual,**{f'p{k}':q[k] for k in range(4)},**{f'y{k}':int(k==y) for k in range(4)}}
            assert row.label==y
            for f,v in values.items():np.testing.assert_allclose(getattr(row,f),v,atol=1e-13,rtol=1e-10);checks+=1
            for population,include in [('both',True),('top',row.source_top),('eligible',row.source_eligible_top)]:
                if include:buckets.setdefault((row.route,population),{}).setdefault(row.date,[]).append(values)
    for route,sub in report['scores'].items():
        g=table[table.route==route]
        for population,out in sub.items():
            selected=g if population=='both' else g[g.source_top] if population=='top' else g[g.source_eligible_top]
            known=selected[selected.ready&selected.mature];days=buckets.get((route,population),{})
            assert out['rows']==len(selected) and out['ready_rows']==int(selected.ready.sum()) and out['unknown_rows']==int((~selected.mature).sum())
            assert out['dates']==len(days) and out['known_rows']==len(known)
            for f,v in out['date_equal_means'].items():
                expected=sum(sum(x[f] for x in rows)/len(rows) for rows in days.values())/len(days)
                np.testing.assert_allclose(v,expected,atol=1e-13,rtol=1e-10)
    history=pd.read_csv(run/'calibrator_history.csv')
    for row in history.itertuples():
        d=read(run/'daily'/(row.date[:10]+'.json'));m=d['models'][row.mode]
        assert row.ready==d['ready'] and row.real_dates==m['real_dates'];np.testing.assert_allclose(row.date_ess,m['date_ess'],atol=1e-12)
        if m['ready']:np.testing.assert_allclose([getattr(row,f'beta{k}') for k in range(4)],m['beta'],atol=1e-12)
    entries=pd.read_csv(run/'actual_entry_forecasts.csv');entry_checks=0
    for name in ROUTES:
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        names={};clocks={}
        for t in trades.itertuples():
            names[t.symbol]=names.get(t.symbol,0)+1;c=pd.Timestamp(t.decision_at).tz_convert(core.ET).strftime('%H:%M:%S');clocks[c]=clocks.get(c,0)+1
        assert report['actual_entry_counts'][name]=={'entries':len(trades),'by_symbol':names,'by_clock':clocks}
        if name not in ['eligible_tilt','top_tilt','both_tilt','identity']:continue
        actual=entries[entries.route==name].reset_index(drop=True);assert len(actual)==len(trades)
        for i,t in enumerate(trades.itertuples()):
            row=actual.iloc[i];at=pd.Timestamp(t.decision_at);j=core.ETFS.index(t.symbol);e=records[at]['events'][name];info=oldrecords[at]['estimates']['gap_blend']
            assert pd.Timestamp(row['at'])==at and row.symbol==t.symbol and row.reason==t.reason
            np.testing.assert_allclose([row.predicted_log_return,row.source_predicted_log_return],[e['values'][j],info['values'][j]],atol=1e-13,rtol=1e-10)
            for k in range(4):np.testing.assert_allclose([row[f'p{k}'],row[f'source_p{k}']],[e['probabilities'][j][k],info['probabilities'][j][k]],atol=1e-13)
            assert row.known==np.isfinite(t.net_return)
            if row.known:
                cls=0 if t.reason=='target' else (2 if t.gap_below_stop else 1) if t.reason=='risk_stop' else 3
                assert int(row['class'])==cls;np.testing.assert_allclose([row.net_return,row.actual_log_return],[t.net_return,math.log1p(t.net_return)],atol=1e-13)
            entry_checks+=1
        if len(actual):
            s=report['actual_entry_summary'][name];known=actual[actual.known]
            assert s['entries']==len(actual) and s['completed']==len(known) and s['pending']==len(actual)-len(known)
            assert {int(float(k)):v for k,v in s['class_counts'].items()}=={int(k):v for k,v in known['class'].value_counts().items()}
            for key in ['predicted_log_return','actual_log_return']:np.testing.assert_allclose(s['mean_'+key],known[key].mean(),atol=1e-13)
    for name,got in report['action_comparisons'].items():
        a=table[(table.route=='eligible_tilt')&(table.side=='L')]
        if name=='source_blend':other=[oldrecords[pd.Timestamp(t)]['events']['gap_blend']['action'] for t in a['at']]
        else:other=table[(table.route==name)&(table.side=='L')].action.tolist()
        assert got=={'different_actions':sum(x!=y for x,y in zip(a.action,other)),'main_buys':int((a.action!='CASH').sum()),'other_buys':sum(x!='CASH' for x in other)}
    primary=pd.read_csv(run/'eligible_tilt_equity.csv');mainnav=[1.,*primary.equity];paired=0
    for name,s in result['paired_increments'].items():
        other=pd.read_csv(run/(name+'_equity.csv'));assert other.date.tolist()==primary.date.tolist();nav=[1.,*other.equity];n=len(primary)
        delta=[math.log(mainnav[i+1]/mainnav[i])-math.log(nav[i+1]/nav[i]) for i in range(n)]
        samples=np.random.default_rng(707).integers(0,n,size=(5000,(n+9)//10));values=[]
        for starts in samples:
            ids=[(int(start)+j)%n for start in starts for j in range(10)][:n];values.append(sum(delta[i] for i in ids)/n)
        np.testing.assert_allclose(s['interval95'],np.quantile(values,[.025,.975]),atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(s['mean_daily_log_difference'],sum(delta)/n,atol=1e-13);paired+=1
    assert report['results_sha256']==sha(run/'results.json') and report['no_policy_changes']
    out={'passed':True,'prediction_values_checked':checks,'actual_entry_forecasts_checked':entry_checks,'calibrator_daily_rows_checked':len(history),
        'paired_bootstrap_comparisons_checked':paired,'diagnostics_sha256':sha(run/'diagnostics.json'),'source_sha256':sha(Path(__file__)),
        'no_policy_changes':True};dump(run/'diagnostic_audit.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);audit(p.parse_args().run)
