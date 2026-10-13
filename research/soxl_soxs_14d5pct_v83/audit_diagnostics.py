"""Independent probability diagnostics, executed rows and capital bootstrap."""
import argparse,math
from state_calibration import *


def audit(run):
    run=Path(run);report=read(run/'diagnostics.json');result=read(run/'results.json');table=pd.read_csv(run/'diagnostic_rows.csv')
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');p=read(SOURCE/'protocol.json');s=p['development'];test=panel[panel.date.between(s['evaluation_start'],s['evaluation_end'])]
    end=pd.Timestamp(s['evaluation_end'],tz=core.ET)+pd.Timedelta(days=1);records={};buckets={};checks=0
    names=list(MODELS)+['identity_matched'];fields=['value','actual','squared_error','brier','log_loss']+[f'q{k}' for k in range(4)]+[f'y{k}' for k in range(4)]
    assert len(table)==len(test)*len(names) and not table.duplicated(['at','route']).any()
    for row in table.itertuples():
        at=pd.Timestamp(row.at);truth=panel.loc[at]
        if at not in records:records[at]=read(run/'training'/(tag(at)+'.json'))
        e=records[at]['events'][row.route];src=read(PARENT/'training'/(tag(at)+'.json'))['events']['recent_ratio']
        mature=bool(pd.notna(truth.terminal_release) and truth.terminal_release<end)
        assert row.route in names and row.ready==e['ready'] and row.mature==mature
        assert row.action==e['action'] and row.source_action==src['action'] and row.source_eligible==(src['action'] in core.ETFS)
        assert row.outside==bool(e.get('outside_state_fields')) and pd.Timestamp(row.date)==truth.date and row.clock==truth.clock
        if row.ready:
            assert row.symbol==e['top_symbol'];np.testing.assert_allclose(row.value,e['value'],atol=1e-13)
            for k in range(4):np.testing.assert_allclose(getattr(row,f'q{k}'),e['probabilities'][k],atol=1e-13);checks+=1
        else:assert pd.isna(row.value) and all(pd.isna(getattr(row,f'q{k}')) for k in range(4))
        if row.ready and row.mature:
            side='L' if e['top_symbol']=='SOXL' else 'S';actual=float(truth[side+'_log']);y=int(truth[side+'_gap']);q=e['probabilities']
            assert row.truth==y
            values=dict(value=e['value'],actual=actual,squared_error=(e['value']-actual)**2,
                        brier=sum((q[k]-int(y==k))**2 for k in range(4)),log_loss=-math.log(max(q[y],1e-8)))
            values.update({f'q{k}':q[k] for k in range(4)});values.update({f'y{k}':int(y==k) for k in range(4)})
            for k,v in values.items():np.testing.assert_allclose(getattr(row,k),v,atol=1e-13,rtol=1e-10);checks+=1
            for pop,include in [('all_top',True),('source_eligible',row.source_eligible),('accepted',row.action!='CASH')]:
                if include:buckets.setdefault((row.route,pop),{}).setdefault(row.date,[]).append(values)
        else:assert pd.isna(row.actual) and pd.isna(row.brier) and pd.isna(row.truth)
    for name,pops in report['scores'].items():
        g=table[table.route==name]
        for pop,score in pops.items():
            selected=g if pop=='all_top' else g[g.source_eligible] if pop=='source_eligible' else g[g.action!='CASH']
            known=selected[selected.ready & selected.mature];days=buckets.get((name,pop),{})
            assert score['rows']==len(selected) and score['ready_rows']==selected.ready.sum() and score['known_rows']==len(known)
            assert score['unknown_rows']==(~selected.mature).sum() and score['known_dates']==len(days) and score['outside_rows']==selected.outside.sum()
            if not days:assert score['date_equal_means']=={}
            for k,v in score['date_equal_means'].items():np.testing.assert_allclose(v,sum(sum(x[k] for x in a)/len(a) for a in days.values())/len(days),atol=1e-13,rtol=1e-10)
    actual=pd.read_csv(run/'actual_entry_forecasts.csv');entrychecks=0
    for name in ROUTES:
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        symbols={};clocks={}
        for t in trades.itertuples():
            symbols[t.symbol]=symbols.get(t.symbol,0)+1;c=pd.Timestamp(t.decision_at).tz_convert(core.ET).strftime('%H:%M:%S');clocks[c]=clocks.get(c,0)+1
        assert report['actual_entry_counts'][name]==dict(entries=len(trades),by_symbol=symbols,by_clock=clocks)
        if name not in names:continue
        rows=actual[actual.route==name].reset_index(drop=True);assert len(rows)==len(trades)
        for i,t in enumerate(trades.itertuples()):
            got=rows.iloc[i];at=pd.Timestamp(t.decision_at);e=records[at]['events'][name]
            assert pd.Timestamp(got['at'])==at and got.symbol==t.symbol and got.reason==t.reason
            assert got.known==np.isfinite(t.net_return) and got.outside==bool(e['outside_state_fields'])
            np.testing.assert_allclose(got.value,e['value'],atol=1e-13)
            for k in range(4):np.testing.assert_allclose(got[f'q{k}'],e['probabilities'][k],atol=1e-13)
            if got.known:
                y=0 if t.reason=='target' else (2 if t.gap_below_stop else 1) if t.reason=='risk_stop' else 3
                assert got.truth==y;np.testing.assert_allclose([got.net_return,got.actual_log_return],[t.net_return,math.log1p(t.net_return)],atol=1e-13)
            entrychecks+=1
        known=rows[rows.known] if len(rows) else rows;st=report['actual_entry_summary'][name]
        assert st['entries']==len(rows) and st['completed']==len(known) and st['pending']==len(rows)-len(known) and st['outside_entries']==rows.outside.sum()
        assert st['class_counts']==[int((known.truth==k).sum()) for k in range(4)]
        if len(known):
            np.testing.assert_allclose([st['mean_value'],st['mean_actual']],[known.value.mean(),known.actual_log_return.mean()],atol=1e-13)
            np.testing.assert_allclose(st['class_probabilities'],[known[f'q{k}'].mean() for k in range(4)],atol=1e-13)
        else:assert st['mean_value'] is None and st['mean_actual'] is None and st['class_probabilities'] is None
    assert set(records)==set(test.index)
    ordered=list(test.index);main=[records[t]['events']['eligible_state']['action'] for t in ordered]
    for name,out in report['action_comparisons'].items():
        other=[records[t]['events'][name]['action'] for t in ordered]
        assert out==dict(different_actions=sum(a!=b for a,b in zip(main,other)),main_buys=sum(x!='CASH' for x in main),other_buys=sum(x!='CASH' for x in other))
    assert len(report['daily_readiness'])==test.date.nunique()
    for d in report['daily_readiness']:
        day=read(run/'daily'/(d['date'][:10]+'.json'));m=day['models']['eligible_state']
        assert d['ready']==day['ready'] and d['common_dates']==len(day['parents']) and d['eligible_dates']==m['real_dates']
        np.testing.assert_allclose(d['eligible_date_ess'],m['date_ess'],atol=1e-13)
    primary=pd.read_csv(run/'eligible_state_equity.csv');m=[1.,*primary.equity];paired=0
    for name,out in result['paired_increments'].items():
        other=pd.read_csv(run/(name+'_equity.csv'));assert other.date.tolist()==primary.date.tolist();o=[1.,*other.equity];n=len(primary)
        delta=[math.log(m[i+1]/m[i])-math.log(o[i+1]/o[i]) for i in range(n)];starts=np.random.default_rng(707).integers(0,n,size=(5000,(n+9)//10));values=[]
        for blocks in starts:
            ids=[(int(s)+j)%n for s in blocks for j in range(10)][:n];values.append(sum(delta[i] for i in ids)/n)
        np.testing.assert_allclose(out['interval95'],np.quantile(values,[.025,.975]),atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(out['mean_daily_log_difference'],sum(delta)/n,atol=1e-13);paired+=1
    assert report['results_sha256']==sha(run/'results.json') and report['no_policy_changes']
    out=dict(passed=True,prediction_values_checked=checks,actual_entry_forecasts_checked=entrychecks,paired_bootstrap_comparisons_checked=paired,
             diagnostics_sha256=sha(run/'diagnostics.json'),source_sha256=sha(Path(__file__)),no_policy_changes=True)
    dump(run/'diagnostic_audit.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);audit(p.parse_args().run)
