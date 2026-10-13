"""Independent calendar, mature members, domain odds, class fits, actions and capital."""
import argparse
from shift_learning import *
from verify_transfer import reference_parents,canonical
from audit_capital import book_check
from audit_statistics import audit_statistics
import reference_shift as ref
import reference_tilt as cal


def verify(run):
    run=Path(run);registration=read(run/'registration.json');pre=registration['preregistration'];p=pre['protocol']
    assert pre['protocol_sha256']==registration['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in pre['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert pre['parent_delivery_sha256']==verify_parent()
    for f,h in pre['input_files_sha256'].items():assert sha(run/f)==h,f
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');market=extended_market();ledger=load_market();history={}
    predictions=read(run/'predictions.json');previous=None;days=checks=actions=members=0;max_gradient=0.;max_class_gradient=0.
    for i,(at,row) in enumerate(panel.iterrows()):
        oldpath=BASE/'training'/(tag(at)+'.json');old=read(oldpath)
        previouspath=PARENT/'training'/(tag(at)+'.json');prev=read(previouspath);record=read(run/'training'/(tag(at)+'.json'))
        assert record['base_source_sha256']==sha(oldpath) and record['parent_source_sha256']==sha(previouspath)
        assert pd.Timestamp(record['at'])==at and pd.Timestamp(record['cutoff'])==day_cutoff(at)
        if previous!=row.date:
            saved=read(run/'daily'/(str(row.date.date())+'.json'));old_daily=read(PARENT/'daily'/(str(row.date.date())+'.json'))
            parents,fb=reference_parents({'semiconductor':panel},market,at)
            chosen=[q for q in parents if all(t in history and history[t]['events']['gap_blend']['ready'] for t in q['members'])][-120:]
            assert old_daily['parents']==canonical(chosen) and old_daily['feedback']==canonical(fb)
            assert saved['cutoff']==canonical(day_cutoff(at)) and saved['at']==canonical(at) and saved['source_ready']==old_daily['ready']
            windows=ref.windows(panel,market.sessions,market.calendar,history,at,CLOCKS)
            if old_daily['ready']:
                P,y,w,meta=cal.data(panel,chosen,history,'top');X,keys,_=ref.contexts(panel,chosen,history,FIELDS)
                assert keys==meta['members'];np.testing.assert_allclose(w,old_daily['models']['top']['weights'],atol=1e-11,rtol=1e-11)
            for mode in VARIANTS:
                got=saved['variants'][mode];assert got['window']==canonical(windows[mode])
                if not old_daily['ready'] or not windows[mode]['ready']:
                    assert not got['ready'] and got['reason']=='source_or_context_unready';continue
                Z,targetkeys,v=ref.contexts(panel,windows[mode]['parents'],history,FIELDS)
                assert got['members']==canonical(keys) and got['target_members']==canonical(targetkeys)
                assert all(t<day_cutoff(at) for t,_ in targetkeys)
                A,dy,dw,center,scale=ref.design(X,Z,w,v);dm=got['domain'];beta=np.asarray(dm['beta'])
                for k,val in [('center',center),('scale',scale),('old_domain_weights',dw[:len(X)]),('new_domain_weights',dw[len(X):])]:
                    np.testing.assert_allclose(dm[k],val,atol=1e-11,rtol=1e-10)
                loss,g=ref.objective(beta,A,dy,dw);norm=float(max(abs(g)));assert norm<=1e-7
                np.testing.assert_allclose(dm['objective'],loss,atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(dm['gradient_inf'],norm,atol=1e-10,rtol=1e-4)
                logratio=np.sum(A[:len(X)]*beta,axis=1);ratio=np.exp(np.clip(logratio,-np.log(5),np.log(5)))
                np.testing.assert_allclose(dm['raw_log_ratios'],logratio,atol=1e-11,rtol=1e-10)
                np.testing.assert_allclose(dm['ratios'],ratio,atol=1e-11,rtol=1e-10)
                assert dm['clipped_low']==int((logratio< -np.log(5)).sum()) and dm['clipped_high']==int((logratio>np.log(5)).sum())
                adjusted,ess=ref.weights(w,ratio,keys);np.testing.assert_allclose(got['weights'],adjusted,atol=1e-11,rtol=1e-10)
                np.testing.assert_allclose([got['weighted_date_ess'],got['original_weight_sum']],[ess,sum(w)],atol=1e-10,rtol=1e-10)
                assert got['overlap_dates']==len(set(meta['dates'])&set(windows[mode]['dates']))
                unsupported=int(sum(P[j,y[j]]==0 for j in range(len(y))));ready=ess>=20 and not unsupported
                assert got['ready']==ready and got['model']['ready']==ready
                if ready:
                    model=got['model'];loss,g=cal.objective(np.array(model['beta']),P,y,adjusted);cnorm=float(max(abs(g)));assert cnorm<=1e-7
                    np.testing.assert_allclose(model['objective'],loss,atol=1e-10,rtol=1e-10)
                    np.testing.assert_allclose(model['gradient_inf'],cnorm,atol=1e-10,rtol=1e-4)
                    assert model['rows']==len(y);max_class_gradient=max(max_class_gradient,cnorm)
                elif ess<20:assert got['model']['reason']=='weighted_date_ess_below20'
                else:assert got['model']['reason']=='issued_true_class_had_zero_support'
                checks+=1;members+=len(y);max_gradient=max(max_gradient,norm)
            previous=row.date;days+=1
        expected={};src=old['events']['gap_blend'];info=old['estimates']['gap_blend']
        for mode in VARIANTS:
            e={'ready':False,'action':'CASH','allocation':0.,'probabilities':None,'values':None}
            if saved['variants'][mode]['ready'] and src['ready']:
                j=0 if info['values'][0]>=info['values'][1] else 1
                q=cal.probabilities(info['probabilities'],saved['variants'][mode]['model']['beta'])
                means=old['distributions']['gap']['payoff_means'];values=[sum(a*b for a,b in zip(q[s],means[s])) for s in range(2)]
                eligible=src['action'] in ['SOXL','SOXS'];buy=eligible and values[j]>1e-12
                e.update(ready=True,action=['SOXL','SOXS'][j] if buy else 'CASH',allocation=float(buy),probabilities=q,values=values,
                    top_symbol=['SOXL','SOXS'][j],buy_advantage=values[j],source_eligible=eligible)
                assert e['action']=='CASH' or e['action']==src['action']
            expected[mode+'_ratio']={'at':at,'date':row.date,'clock':row.clock,**e}
        expected['uniform_top']=prev['events']['top_tilt'] if saved['variants']['recent']['ready'] else {
            'at':at,'date':row.date,'clock':row.clock,'ready':False,'action':'CASH','allocation':0.,'probabilities':None,'values':None}
        for name,source in CONTROL_MAP.items():expected[name]=prev['events'][source]
        cash={'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None}
        expected['cash']=cash;expected['double_cost']=dict(expected['recent_ratio'])
        expected['premarket_only']=dict(expected['recent_ratio']) if row.clock==CLOCKS[0] else cash
        for name,e in expected.items():
            got=record['events'][name];assert got==predictions[name][i]
            for k in ['at','date','clock','ready','action','allocation','top_symbol','source_eligible']:assert got.get(k)==canonical(e.get(k))
            for k in ['values','probabilities','buy_advantage']:
                if e.get(k) is None:assert got.get(k) is None
                else:np.testing.assert_allclose(got[k],e[k],atol=1e-11,rtol=1e-10)
            actions+=1
        history[at]=old
        if (i+1)%800==0:print('Verified shifts',i+1,'/',len(panel),at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json') and not frozen['new_capital_outcomes_seen']
    settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings['months']
    features=pd.read_parquet(run/'features.parquet');test=features[features.date.between(start,end)]
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in ledger.sessions[(ledger.sessions>=start)&(ledger.sessions<=end)]
        for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(ledger.calendar[d]['close_at'])]
    assert list(test.index)==expected and result['coverage']['complete'];baseline=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');books=[]
    for name in ROUTES:
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert frozen['routes'][name]==es
        books.append(book_check(run,name,ledger,test,stress if name=='double_cost' else baseline,es,2 if name=='double_cost' else 1,result['routes'][name],months,require_full_cash=False))
    stats=audit_statistics(run);original=read(PARENT/'results.json')
    for name,oldname in CONTROL_MAP.items():
        assert result['routes'][name]==original['routes'][oldname]
        for suffix in ['equity.csv','trades.csv','decisions.csv']:assert sha(run/(name+'_'+suffix))==sha(PARENT/(oldname+'_'+suffix))
    gate=development_gate(result['primary'],True,p);assert gate==result['development_gate']
    assert result['primary_name']==p['primary'] and result['primary']==result['routes'][p['primary']]
    assert not result['goal_complete'] and not result['case_regression']['executed']
    out={'passed':True,'results_sha256':sha(run/'results.json'),'prediction_rows':len(panel),'dates':days,'domain_and_calibrator_gradient_checks':checks,
        'historical_candidate_members_checked':members,'maximum_domain_gradient_inf':max_gradient,'maximum_class_gradient_inf':max_class_gradient,'route_actions':actions,'capital_books':books,
        'statistics_checks':stats,'four_parent_books_byte_identical':True,'source_sha256':sha(Path(__file__)),'development_gate':gate,
        'strategy_independent_validation':False,'goal_complete':False};dump(run/'verification.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);verify(p.parse_args().run)
