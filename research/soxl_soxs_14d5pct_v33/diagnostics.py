"""Read-only cost-path sensitivity, maturity changes and issued-choice audit."""
from cost_robust import *
from verify_continuation import near


def describe(frame):
    rows=[]
    for s in ['L','S']:
        a=frame['base_'+s];b=frame['stress_'+s];valid=a.notna()&b.notna()
        for t in frame.index[valid]:rows.append({'at':t,'date':frame.loc[t,'date'],'symbol':'SOXL' if s=='L' else 'SOXS',
            'baseline':a.loc[t],'stress':b.loc[t],'loss_from_worst':a.loc[t]-min(a.loc[t],b.loc[t])})
    f=pd.DataFrame(rows)
    return {'known_scenario_pairs':len(f),'unknown_pairs':len(frame)*2-len(f),'dates':f.date.nunique() if len(f) else 0,
        'stress_improves_count':int((f.stress>f.baseline).sum()),'worst_target_equals_baseline_count':int(f.loss_from_worst.eq(0).sum()),
        'date_equal_mean_log_reduction':float(f.groupby('date').loss_from_worst.mean().mean()),
        'log_reduction_quantiles':{str(q):float(f.loss_from_worst.quantile(q)) for q in [0,.5,.9,.99,1]}}


def diagnose(run):
    run=Path(run);result=read(run/'results.json');verify=read(run/'verification.json');reg=read(run/'registration.json')
    assert verify['passed'] and verify['results_sha256']==sha(run/'results.json')
    settings=reg['preregistration']['protocol'][reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end']
    p=read(run/'predictions.json');bank=pd.read_parquet(run/'joint_feedback.parquet');panel=pd.read_parquet(run/'panel.parquet')
    sel=bank[bank.date.between(start,end)];base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json')
    counts={n:{s:sum(e['action']==s and start<=e['date'][:10]<=end for e in es) for s in [*core.ETFS,'CASH']} for n,es in p.items()}
    changes={n:[{'at':a['at'],'primary':a['action'],'control':b['action']} for a,b in zip(p['worst_band'],p[n])
        if start<=a['date'][:10]<=end and a['action']!=b['action']] for n in ['matched_baseline','average_band','source_v32']}
    rows=[]
    for at,r in bank.iterrows():
        for s in core.ETFS:
            a=base[key(at,s)];b=stress[key(at,s)];x=a.get('net_return');y=b.get('net_return')
            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):continue
            if a['reason']!=b['reason']:rows.append({'at':at,'date':r.date,'symbol':s,'baseline_reason':a['reason'],'stress_reason':b['reason'],
                'baseline_return':x,'stress_return':y,'feature_ready':bool(r.feature_ready),'joint_feedback_known':pd.notna(r.terminal_release)})
    support=[];predicted_up=0;candidate_rows=0;err=[]
    for at,r in sel.iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));info=rec['estimates']
        if r.clock==CLOCKS[0]:
            d=read(run/'daily'/rec['daily_record']);oldroot=core.ROOT/reg['preregistration']['protocol']['control_source']
            source=oldroot/'daily'/rec['daily_record'] if r.date<pd.Timestamp('2026-05-01') else run/'source_daily'/rec['daily_record']
            old=read(source);newids=d['members'];oldids=old['members']
            support.append({'date':r.date,'new_dates':len(d['parents']),'old_dates':len(old['parents']),
                'members_equal':newids==oldids,'newest_joint':d['parents'][-1]['date'] if d['parents'] else None,
                'newest_baseline':old['parents'][-1]['date'] if old['parents'] else None})
        if info['worst']['ready']:
            a=np.array(info['worst']['point']);b=np.array(info['baseline']['point']);predicted_up+=int((a>b+1e-12).sum());candidate_rows+=2
            if pd.notna(r.terminal_release):
                truth=np.array([r.base_L,r.base_S]);worst=np.minimum(truth,np.array([r.stress_L,r.stress_S]))
                err.append({'date':r.date,**{k:float(np.mean((np.array(info[k]['point'])-truth)**2)) for k in VARIANTS},
                    'worst_target_error':float(np.mean((a-worst)**2))})
    f=pd.DataFrame(err);mse={k:float(f.groupby('date')[k].mean().mean()) for k in [*VARIANTS,'worst_target_error']} if len(f) else {}
    ds=pd.read_csv(run/'worst_band_decisions.csv');ds.index=pd.to_datetime(ds.decision_at,utc=True);proposals=ds[ds.requested_action.isin(core.ETFS)]
    monthly=[]
    for month,g in ds.groupby(ds.index.tz_convert(core.ET).tz_localize(None).to_period('M')):
        active=g[g.requested_action.isin(core.ETFS)];buy=g[g.decision.str.startswith('buy_')]
        monthly.append({'month':str(month),'proposals':len(active),'actual_buys':len(buy),'blocked':active[~active.decision.str.startswith('buy_')].decision.value_counts().to_dict()})
    actual={}
    for n in ROUTES:
        try:t=pd.read_csv(run/(n+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        actual[n]=[{k:getattr(x,k) for k in ['decision_at','symbol','entry_at','exit_at','reason','net_return']} for x in t.itertuples()]
        lab=stress if n=='double_cost' else base
        for x in t.itertuples():near(x.net_return,lab[key(pd.Timestamp(x.decision_at),x.symbol)]['net_return'])
    primary_book={(pd.Timestamp(t['decision_at']),t['symbol']):t for t in actual['worst_band']}
    control_book={(pd.Timestamp(t['decision_at']),t['symbol']):t for t in actual['matched_baseline']}
    control_ds=pd.read_csv(run/'matched_baseline_decisions.csv');control_ds.index=pd.to_datetime(control_ds.decision_at,utc=True)
    event_maps={n:{pd.Timestamp(e['at']):e for e in p[n]} for n in ['worst_band','matched_baseline']}
    trade_changes={}
    for label,keys,book in [('added',set(primary_book)-set(control_book),primary_book),('removed',set(control_book)-set(primary_book),control_book)]:
        values=[]
        for at,s in sorted(keys):
            candidate_primary=event_maps['worst_band'][at]['action'];candidate_control=event_maps['matched_baseline'][at]['action']
            values.append({**book[(at,s)],'primary_candidate':candidate_primary,'control_candidate':candidate_control,
                'primary_ledger_decision':ds.loc[at,'decision'],'control_ledger_decision':control_ds.loc[at,'decision'],
                'same_issued_action':candidate_primary==candidate_control})
        trade_changes[label]=values
    out={'passed':True,'post_result_only':True,'phase':reg['phase'],'scenario_sensitivity_before_evaluation':describe(bank[(bank.date<start)&bank.feature_ready]),
        'scenario_sensitivity_evaluation':describe(sel[sel.feature_ready]),'all_reason_flips':rows,'candidate_counts':counts,'changed_actions':changes,
        'daily_support':support,'different_maturity_members_days':sum(not d['members_equal'] for d in support),
        'worst_prediction_above_matched_baseline_count':predicted_up,'comparable_etf_queries':candidate_rows,
        'mature_error_ticks':len(err),'date_equal_candidate_mse':mse,'monthly_opportunities':monthly,'actual_trades':actual,
        'actual_entry_changes_against_matched_baseline':trade_changes,
        'primary_total_proposals':len(proposals),'primary_blocked':proposals[~proposals.decision.str.startswith('buy_')].decision.value_counts().to_dict(),
        'multiple_scenarios_are_not_independent_samples':True,'historical_worst_target_is_not_future_bound':True,
        'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'diagnostics.json',out);print({k:v for k,v in out.items() if k in ['different_maturity_members_days','worst_prediction_above_matched_baseline_count','mature_error_ticks','date_equal_candidate_mse','monthly_opportunities']},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
