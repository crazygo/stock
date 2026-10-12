"""Post-result date-weighted errors and frozen-action attribution."""
from shared_error import *
from verify_continuation import key,near


def diagnose(run):
    run=Path(run);result=read(run/'results.json');verification=read(run/'verification.json')
    assert verification['passed'] and verification['results_sha256']==sha(run/'results.json')
    panel=pd.read_parquet(run/'panel.parquet');base=labels(run/'baseline_labels.json');pred=read(run/'predictions.json')
    select=lambda d:'2026-01-01'<=str(d)[:10]<='2026-04-30'
    issued={n:[e for e in es if select(e['date'])] for n,es in pred.items()}
    action_counts={n:{s:sum(e['action']==s for e in es) for s in [*core.ETFS,'CASH']} for n,es in issued.items()}
    changes={};outcomes={};derivatives=[];parent_counts=[];errors=[];old_slopes=[]
    oldroot=core.ROOT/read(run/'registration.json')['preregistration']['protocol']['control_source']
    for n in ['additive_band','shared_band','source_v27']:
        records=[]
        for a,b in zip(issued['interaction_band'],issued[n]):
            assert a['at']==b['at']
            if a['action']!=b['action']:records.append({'at':a['at'],'primary':a['action'],'control':b['action']})
        changes[n]=records
    for e in issued['interaction_band']:
        at=pd.Timestamp(e['at']);rec=read(run/'training'/(tag(at)+'.json'));d=read(run/'daily'/rec['daily_record'])
        if rec['clock']==CLOCKS[0]:parent_counts.append({'date':rec['date'],**d['feedback']})
        source=read(oldroot/'training'/(tag(at)+'.json'));old=source['estimate']
        if old['ready']:
            gain=np.eye(2)+np.asarray(old['point_model']['coef'])/np.asarray(old['point_model']['scale'])[None,:]
            old_slopes.append({'date':rec['date'],'clock':rec['clock'],'spread_gain':float(np.array([1.,-1.])@gain@np.array([.5,-.5])),
                'total_gain':gain.tolist()})
        if not rec['estimates']['interaction']['ready']:continue
        # Same daily coefficients, observed state versus training-center state; no new actions.
        m=d['variants']['interaction']['point_model'];X=encode(panel,{at:rec['raw']},[at])[0];z=(X-m['input_center'])/m['input_scale']
        coef=np.array(m['coef']);scale=np.array(m['basis_scale']);gain=np.eye(2);center=np.eye(2)
        for k in range(2):
            center[:,k]+=coef[:,k]/scale[k]/m['input_scale'][k]
            gain[:,k]+=(coef[:,k]/scale[k]+sum(coef[:,11+k*6+j]/scale[11+k*6+j]*z[2+j] for j in range(6)))/m['input_scale'][k]
        derivatives.append({'at':at,'observed_gain':gain.tolist(),'training_center_gain':center.tolist(),'state_slope_change_max':float(abs(gain-center).max())})
        if pd.notna(panel.loc[at,'terminal_release']):
            truth=panel.loc[at,['C','D']].to_numpy(float);truth=np.array([sum(truth),truth[0]-truth[1]])
            row={'at':at,'date':rec['date'],'clock':rec['clock'],'raw_error_L':truth[0]-rec['raw'][0],
                'raw':float(np.mean((truth-np.array(rec['raw']))**2))}
            for k in VARIANTS:row[k]=float(np.mean((truth-np.array(rec['estimates'][k]['point']))**2))
            errors.append(row)
    frame=pd.DataFrame(errors);mses={k:float(frame.groupby('date')[k].mean().mean()) for k in ['raw',*VARIANTS]} if len(frame) else {}
    corr=frame.pivot(index='date',columns='clock',values='raw_error_L').corr().to_dict() if len(frame) else {}
    for n in ['interaction_band','interaction_point','additive_band','shared_band','source_v27']:
        try:t=pd.read_csv(run/(n+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        rows=[]
        for r in t.itertuples():
            lab=base[key(pd.Timestamp(r.decision_at),r.symbol)];near(r.net_return,lab['net_return'])
            rows.append({k:getattr(r,k) for k in ['decision_at','symbol','entry_at','exit_at','reason','net_return']})
        outcomes[n]=rows
    output={'passed':True,'post_result_diagnostic_only':True,'action_counts':action_counts,'changed_actions':changes,
        'date_equal_mse':mses,'mature_error_rows':len(errors),'mature_error_dates':frame.date.nunique() if len(frame) else 0,
        'source_v27_local_slopes':old_slopes,'crossclock_SOXL_raw_error_correlation':corr,'parent_support':parent_counts,'state_dependent_score_slopes':derivatives,
        'actual_trades':outcomes,'candidate_MSE_is_not_trade_or_monthly_acceptance':True,
        'slope_counterfactual_is_not_causal_market_effect':True,'results_sha256':sha(run/'results.json'),
        'verification_sha256':sha(run/'verification.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'diagnostics.json',output);print({'MSE':mses,'changed_actions':{n:len(r) for n,r in changes.items()},
        'candidate_error_dates':output['mature_error_dates'],'actual_trades':{n:len(r) for n,r in outcomes.items()}},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
