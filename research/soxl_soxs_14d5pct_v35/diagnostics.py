"""Post-result coverage, feature ablations, prediction errors and capital choices."""
from financial_learning import *
from verify_continuation import key,near


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');reg=read(run/'registration.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    settings=reg['preregistration']['protocol'][reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end']
    p=pd.read_parquet(run/'financial_panel.parquet');sel=p[p.date.between(start,end)];pred=read(run/'predictions.json')
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');errors=[];support=[];equalities=[];unknown=0;slopes=[]
    for at,row in sel.iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));daily=read(run/'daily'/rec['daily_record']) if row.clock==CLOCKS[0] else None
        if daily:
            support.append({'date':row.date,**daily['feedback']})
            if daily['ready']:
                item={'date':row.date,'coefficient_layout':'clock,input ETF,output ETF'}
                for k in VARIANTS:
                    m=daily['variants'][k]['point_model']
                    # Residual correction derivative in original raw-score units.
                    item[k]=(np.array(m['coefficients'])/np.array(m['input_scale'])[:,:,None]).tolist()
                slopes.append(item)
        if rec['raws']['baseline'] is not None and rec['source_raw'] is not None:
            equalities.append(float(np.max(np.abs(np.array(rec['raws']['baseline'])-rec['source_raw']))))
        if not all(e['ready'] for e in rec['estimates'].values()):continue
        if pd.isna(row.terminal_release):unknown+=1;continue
        truth=np.array([row.C+row.D,row.C-row.D]);one={'date':row.date,'at':at}
        for k in VARIANTS:
            one['raw_'+k]=float(np.mean((np.array(rec['raws'][k])-truth)**2))
            one['adjusted_'+k]=float(np.mean((np.array(rec['estimates'][k]['point'])-truth)**2))
        errors.append(one)
    f=pd.DataFrame(errors);mse={k:float(f.groupby('date')[k].mean().mean()) for k in f if k not in ['date','at']} if len(f) else {}
    counts={n:{s:sum(e['action']==s and start<=e['date'][:10]<=end for e in es) for s in [*core.ETFS,'CASH']} for n,es in pred.items()}
    changed={n:[{'at':a['at'],'primary':a['action'],'control':b['action']} for a,b in zip(pred['full_band'],pred[n]) if start<=a['date'][:10]<=end and a['action']!=b['action']]
             for n in ['state_band','baseline_band','source_v32']}
    ranges={c:{k:{'min':float(g[k].min()),'max':float(g[k].max()),'mean':float(g[k].mean())} for k in FINANCIAL} for c,g in sel[sel.financial_ready].groupby('clock')}
    actual={};books={};decisions={}
    for n in ROUTES:
        try:t=pd.read_csv(run/(n+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        rows=[]
        for x in t.itertuples():
            near(x.net_return,(stress if n=='double_cost' else base)[key(pd.Timestamp(x.decision_at),x.symbol)]['net_return'])
            rows.append({k:getattr(x,k) for k in ['decision_at','symbol','entry_at','exit_at','reason','net_return']})
        actual[n]=rows;books[n]={(pd.Timestamp(x['decision_at']),x['symbol']):x for x in rows}
        ds=pd.read_csv(run/(n+'_decisions.csv'));ds.index=pd.to_datetime(ds.decision_at,utc=True);decisions[n]=ds
    ds=decisions['full_band'];proposals=ds[ds.requested_action.isin(core.ETFS)];monthly=[]
    for month,g in ds.groupby(ds.index.tz_convert(core.ET).tz_localize(None).to_period('M')):
        active=g[g.requested_action.isin(core.ETFS)]
        monthly.append({'month':str(month),'proposals':len(active),'actual_buys':int(g.decision.str.startswith('buy_').sum()),
            'blocked':active[~active.decision.str.startswith('buy_')].decision.value_counts().to_dict()})
    trades_changed={}
    for control in ['state_band','baseline_band']:
        out={}
        for label,keys,n in [('added',set(books['full_band'])-set(books[control]),'full_band'),('removed',set(books[control])-set(books['full_band']),control)]:
            out[label]=[{**books[n][item],'primary_ledger_decision':decisions['full_band'].loc[item[0],'decision'],
                'control_ledger_decision':decisions[control].loc[item[0],'decision'],
                'primary_candidate':decisions['full_band'].loc[item[0],'requested_action'],
                'control_candidate':decisions[control].loc[item[0],'requested_action']} for item in sorted(keys)]
        trades_changed[control]=out
    temporal={}
    for col in FINANCIAL:
        counts_per_day=sel.groupby('date')[col].nunique()
        temporal[col]={'dates':len(counts_per_day),'dates_with_intraday_change':int((counts_per_day>1).sum())}
    for col in FINANCIAL:
        assert temporal[col]['dates_with_intraday_change']==0
    shifts={k:{'max_absolute_derivative_difference_from_baseline':float(max(np.max(np.abs(np.array(x[k])-x['baseline'])) for x in slopes)),
        'interpretation':'Different raw learners have different score distributions; a slope difference is descriptive, not a causal attribution of capital loss.'}
        for k in ['full','state']}
    source=read(run/'feature_receipts.json');disclosures={};semantic_mismatches=[]
    for symbol in PANEL:
        valid=[x['states'][symbol] for x in source['days'] if x['states'][symbol] is not None]
        comparisons=[]
        for day in source['days']:
            state=day['states'][symbol]
            if state is None:continue
            now=sorted({x['tag'] for x in state['sources']['net_income']['sources']})
            previous=sorted({x['tag'] for x in state['sources']['year_net_income']['sources']})
            comparisons.append({'date':day['date'],'current_tags':now,'previous_year_tags':previous})
            if now!=previous:semantic_mismatches.append({'symbol':symbol,**comparisons[-1]})
        assert not semantic_mismatches, 'Different net income scopes require a new input protocol before acceptance'
        disclosures[symbol]={'known_state_days':len(valid),'distinct_latest_quarters':len({x['latest_period'] for x in valid}),
            'distinct_used_disclosure_dates':len({x['release_date'] for x in valid}),
            'net_income_tag_pairs':sorted({(tuple(x['current_tags']),tuple(x['previous_year_tags'])) for x in comparisons}),
            'same_scope_year_comparisons_checked':len(comparisons),
            'missing_days':[x['date'] for x in source['days'] if x['states'][symbol] is None]}
    receipt={'financial_state_coverage':disclosures,'excluded_financial_rows':source['excluded_raw_rows'],
        'net_income_scope_mismatches':semantic_mismatches,
        'net_income_scope_note':'AVGO ProfitLoss includes noncontrolling interests; other7issuers use NetIncomeLoss attributable to parent. Within-issuer year pairs match in this run; cross-company scopes are not identical.',
        'historical_availability_assumption':source['availability'],'passed':True,'post_result_only':True,'phase':reg['phase'],'total_input_ticks':len(p),'financial_ready_ticks':int(p.financial_ready.sum()),
        'development_ticks':len(sel),'development_financial_ready':int(sel.financial_ready.sum()),
        'development_raw_feature_ready':int(sel.feature_ready.sum()),'feature_ranges_by_clock':ranges,
        'baseline_raw_vs_source_max_difference':max(equalities) if equalities else None,
        'within_day_feature_changes':temporal,'daily_residual_derivatives':slopes,'calibration_shift_summary':shifts,
        'candidate_counts':counts,'changed_actions':changed,'parent_support':support,
        'mature_error_ticks':len(errors),'unknown_error_ticks_retained':unknown,'date_equal_mse':mse,
        'actual_trades':actual,'monthly_opportunities':monthly,'actual_entry_changes':trades_changed,
        'primary_total_proposals':len(proposals),'primary_blocked':proposals[~proposals.decision.str.startswith('buy_')].decision.value_counts().to_dict(),
        'candidate_error_is_not_capital_acceptance':True,'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'diagnostics.json',receipt);print({'mse':mse,'changed_actions':{k:len(v) for k,v in changed.items()},'monthly':monthly,
        'baseline_raw_source_error':receipt['baseline_raw_vs_source_max_difference']},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
