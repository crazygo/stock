"""Separate arithmetic for proper scores, mixture summaries, entries and paired intervals."""
import argparse,math
from class_learning import *


def audit(run):
    run=Path(run);report=read(run/'diagnostics.json');result=read(run/'results.json');table=pd.read_csv(run/'diagnostic_rows.csv')
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');daily={};checks=0;records={}
    for row in table.itertuples():
        at=pd.Timestamp(row.at)
        if at not in records:records[at]=read(run/'training'/(tag(at)+'.json'))
        info=records[at]['estimates'][row.route];j=0 if row.side=='L' else 1;truthrow=panel.loc[at]
        assert row.mature==pd.notna(truthrow.terminal_release) and row.ready==info['ready']
        assert row.action==records[at]['events'][row.route]['action']
        if not row.ready:continue
        probs=info['probabilities'][j]
        np.testing.assert_allclose([row.predicted_log_return,row.alpha,row.target_probability],
            [info['values'][j],info['alpha'],probs[0]],atol=1e-13,rtol=1e-11)
        if row.mode=='gap':np.testing.assert_allclose(row.gap_probability,probs[2],atol=1e-13,rtol=1e-11)
        if not row.mature:continue
        y=int(truthrow[row.side+'_'+row.mode]);actual=float(truthrow[row.side+'_log']);common=int(truthrow[row.side+'_merged'])
        cp=[probs[0],probs[1]+probs[2],probs[3]] if row.mode=='gap' else probs
        wanted={'brier':1-2*probs[y]+sum(v*v for v in probs),'log_loss':-math.log(max(probs[y],1e-8)),
            'common_three_brier':1-2*cp[common]+sum(v*v for v in cp),'payoff_squared_error':(info['values'][j]-actual)**2}
        for k,v in wanted.items():np.testing.assert_allclose(getattr(row,k),v,atol=1e-13,rtol=1e-10);checks+=1
        assert row.label==y and row.target_actual==int(y==0)
        np.testing.assert_allclose(row.actual_log_return,actual,atol=1e-13,rtol=1e-10)
        daily.setdefault(row.route,{}).setdefault(row.date,[]).append(wanted)
    for name,dates in daily.items():
        wanted=report['scores'][name];g=table[table.route==name]
        assert wanted['dates']==len(dates) and wanted['etf_clock_rows']==sum(len(rs) for rs in dates.values())
        assert wanted['ready_rows']==int(g.ready.sum()) and wanted['unknown_rows']==int((~g.mature).sum())
        for metric in ['brier','log_loss','common_three_brier','payoff_squared_error']:
            value=sum(sum(r[metric] for r in rs)/len(rs) for rs in dates.values())/len(dates)
            np.testing.assert_allclose(wanted[metric],value,atol=1e-13,rtol=1e-10)
    weightrows=pd.read_csv(run/'mixture_weights.csv');weightchecks=0
    for r in weightrows.itertuples():
        d=read(run/'daily'/(r.date[:10]+'.json'))['mixture'][r.mode]
        assert r.ready==d['ready'] and r.dates==len(d['parents'])
        np.testing.assert_allclose([r.alpha,r.logodds],[d['alpha'],d['logodds']],atol=1e-12,rtol=1e-10);weightchecks+=1
    for mode,g in weightrows.groupby('mode'):
        summary=report['mixture_summary'][mode];assert summary['dates']==len(g)
        np.testing.assert_allclose([summary['alpha_min'],summary['alpha_max'],summary['alpha_mean']],[min(g.alpha),max(g.alpha),sum(g.alpha)/len(g)],atol=1e-12,rtol=1e-10)
    entrychecks=0;primary=result['primary_name']
    for name in ROUTES:
        try:t=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        names={};clocks={}
        for trade in t.itertuples():
            names[trade.symbol]=names.get(trade.symbol,0)+1;c=pd.Timestamp(trade.decision_at).tz_convert(core.ET).strftime('%H:%M:%S');clocks[c]=clocks.get(c,0)+1
        assert report['entries'][name]=={'entries':len(t),'by_symbol':names,'by_clock':clocks}
        if name!=primary:continue
        try:chosen=pd.read_csv(run/'actual_entry_forecasts.csv')
        except pd.errors.EmptyDataError:chosen=pd.DataFrame()
        assert len(chosen)==len(t)
        for i,trade in enumerate(t.itertuples()):
            row=chosen.iloc[i];rec=records[pd.Timestamp(trade.decision_at)];j=core.ETFS.index(trade.symbol);info=rec['estimates']['gap_blend']
            assert pd.Timestamp(row.decision_at)==pd.Timestamp(trade.decision_at) and row.symbol==trade.symbol and row.reason==trade.reason
            assert bool(row.gap_below_stop)==bool(trade.gap_below_stop)
            np.testing.assert_allclose([row.predicted_log_return,row.actual_log_return,row.target_probability,row.gap_probability,row.alpha,row.net_return],
                [info['values'][j],math.log1p(trade.net_return),info['probabilities'][j][0],info['probabilities'][j][2],info['alpha'],trade.net_return],atol=1e-13,rtol=1e-10)
            entrychecks+=1
        s=report['actual_primary_entries'];assert s['entries']==len(t) and s['completed']==(int(chosen.actual_log_return.notna().sum()) if len(chosen) else 0)
        for k,v in s['means'].items():np.testing.assert_allclose(v,chosen[k].mean(),atol=1e-13,rtol=1e-10)
    prior_comparisons=0
    for at,record in records.items():
        a=record['estimates']['gap_prior'];b=record['estimates']['merged_prior'];assert a['ready']==b['ready']
        if a['ready']:
            np.testing.assert_allclose(a['values'],b['values'],atol=1e-13,rtol=1e-10)
            for qa,qb in zip(a['probabilities'],b['probabilities']):
                np.testing.assert_allclose([qa[0],qa[1]+qa[2],qa[3]],qb,atol=1e-13,rtol=1e-10)
            prior_comparisons+=1
    for suffix in ['equity.csv','trades.csv','decisions.csv']:assert sha(run/('gap_prior_'+suffix))==sha(run/('merged_prior_'+suffix))
    main=pd.read_csv(run/(primary+'_equity.csv'));mainnav=[1.,*main.equity];paired=0
    for name,got in result['paired_increments'].items():
        other=pd.read_csv(run/(name+'_equity.csv'));assert list(other.date)==list(main.date);nav=[1.,*other.equity];n=len(main)
        differences=[math.log(mainnav[i+1]/mainnav[i])-math.log(nav[i+1]/nav[i]) for i in range(n)]
        starts=np.random.default_rng(707).integers(0,n,size=(5000,(n+9)//10));boot=[]
        for draw in starts:
            ids=[(int(start)+j)%n for start in draw for j in range(10)][:n];boot.append(sum(differences[j] for j in ids)/n)
        boot.sort()
        def quantile(q):
            pos=q*(len(boot)-1);a=int(pos);return boot[a]+(boot[min(a+1,len(boot)-1)]-boot[a])*(pos-a)
        interval=[quantile(.025),quantile(.975)]
        np.testing.assert_allclose(got['interval95'],interval,atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(got['mean_daily_log_difference'],sum(differences)/n,atol=1e-13,rtol=1e-10)
        assert got['positive_increment_supported']==(interval[0]>0);paired+=1
    assert report['results_sha256']==sha(run/'results.json') and not report['changes_primary_or_gate']
    out={'passed':True,'proper_score_and_payoff_values_checked':checks,'daily_mixture_rows_checked':weightchecks,
        'actual_entry_forecasts_checked':entrychecks,'paired_bootstrap_comparisons_checked':paired,
        'prior_coarsening_query_checks':prior_comparisons,'gap_and_merged_prior_books_identical':True,
        'diagnostics_sha256':sha(run/'diagnostics.json'),'source_sha256':sha(Path(__file__)),'policy_changes':False}
    dump(run/'diagnostic_audit.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);audit(p.parse_args().run)
