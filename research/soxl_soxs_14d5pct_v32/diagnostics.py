"""Post-result slope shrinkage, date-weighted errors and executable choices."""
from hierarchical_error import *
from audit_hierarchical import predict as ref_predict
from verify_reliability import close
from verify_continuation import key,near


def diagnose(run):
    run=Path(run);r=read(run/'results.json');v=read(run/'verification.json');reg=read(run/'registration.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    settings=reg['preregistration']['protocol'][reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end']
    select=lambda d:start<=str(d)[:10]<=end
    panel=pd.read_parquet(run/'panel.parquet');labels_base=labels(run/'baseline_labels.json');pred=read(run/'predictions.json')
    issued={n:[e for e in es if select(e['date'])] for n,es in pred.items()};changes={}
    for name in ['shared_band','independent_band','source_v27']:
        changes[name]=[{'at':a['at'],'primary':a['action'],'control':b['action']} for a,b in zip(issued['partial_band'],issued[name]) if a['action']!=b['action']]
    counts={n:{s:sum(e['action']==s for e in es) for s in [*core.ETFS,'CASH']} for n,es in issued.items()}
    daily_models={};slopes=[];errors=[];support=[];ready_count=0;gradient_checks=0
    for at,row in panel[panel.date.between(start,end)].iterrows():
        rec=read(run/'training'/(tag(at)+'.json'));filename=rec['daily_record']
        if filename not in daily_models:
            d=read(run/'daily'/filename);daily_models[filename]=d;support.append({'date':rec['date'],**d['feedback']})
            if d['ready']:
                measures={}
                for kind in VARIANTS:
                    beta=np.array(d['variants'][kind]['point_model']['coefficients']);mean=beta.mean(axis=0)
                    measures[kind]={'contrast_norm2':float(np.sum((beta-mean)**2)),'mean_norm2':float(np.sum(mean**2))}
                assert measures['partial']['contrast_norm2']<=measures['independent']['contrast_norm2']+1e-12
                near(measures['shared']['contrast_norm2'],0.)
                slopes.append({'date':rec['date'],'models':measures})
        d=daily_models[filename]
        if not rec['estimates']['partial']['ready']:continue
        ready_count+=1;x=np.array([*rec['raw'],CLOCKS.index(row.clock)]);c=int(x[2])
        for kind in VARIANTS:
            m=d['variants'][kind]['point_model'];gain=np.eye(2)+np.array(m['coefficients'][c]).T/np.array(m['input_scale'][c])[None,:]
            for k in range(2):
                a=x.copy();b=x.copy();a[k]+=1e-6;b[k]-=1e-6
                numerical=(ref_predict(m,a)[0]-ref_predict(m,b)[0])/(2e-6)+np.eye(2)[:,k]
                close(numerical,gain[:,k]);gradient_checks+=2
        if pd.isna(row.terminal_release):continue
        truth=np.array([row.C+row.D,row.C-row.D]);one={'date':row.date,'at':at,'raw':float(np.mean((truth-np.array(rec['raw']))**2))}
        for kind in VARIANTS:one[kind]=float(np.mean((truth-np.array(rec['estimates'][kind]['point']))**2))
        errors.append(one)
    frame=pd.DataFrame(errors);mses={k:float(frame.groupby('date')[k].mean().mean()) for k in ['raw',*VARIANTS]} if len(frame) else {}
    actual={};rejections={}
    for name in ROUTES:
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        rows=[]
        for t in trades.itertuples():
            if name!='double_cost':near(t.net_return,labels_base[key(pd.Timestamp(t.decision_at),t.symbol)]['net_return'])
            rows.append({k:getattr(t,k) for k in ['decision_at','symbol','entry_at','exit_at','reason','net_return']})
        actual[name]=rows
        ds=pd.read_csv(run/(name+'_decisions.csv'));rejections[name]=ds.decision.value_counts().to_dict()
    receipt={'passed':True,'phase':reg['phase'],'post_result_diagnostic_only':True,'candidate_counts':counts,'changed_actions':changes,
        'parent_support':support,'slope_contrasts':slopes,'ready_ticks':ready_count,'mature_error_ticks':len(errors),
        'mature_error_dates':frame.date.nunique() if len(frame) else 0,'unknown_pair_errors_retained':ready_count-len(errors),
        'date_equal_mse':mses,'independent_local_derivative_checks':gradient_checks,'actual_trades':actual,'execution_decisions':rejections,
        'candidate_errors_are_not_monthly_acceptance':True,'results_sha256':sha(run/'results.json'),
        'verification_sha256':sha(run/'verification.json'),'source_sha256':sha(Path(__file__))}
    dump(run/'diagnostics.json',receipt);print({'MSE':mses,'changed_actions':{k:len(v) for k,v in changes.items()},
        'actual_trades':{k:len(v) for k,v in actual.items()},'gradient_checks':gradient_checks},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
