"""Independent mature populations, convex calibration, actions and capital checks."""
import argparse
from tilt_learning import *
from verify_transfer import reference_parents,canonical
from audit_capital import book_check
from audit_statistics import audit_statistics
import reference_tilt as ref


def verify(run):
    run=Path(run);reg=read(run/'registration.json');pre=reg['preregistration'];p=pre['protocol']
    assert pre['protocol_sha256']==reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in pre['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert pre['parent_delivery_sha256']==verify_parent()
    for f,h in pre['input_files_sha256'].items():assert sha(run/f)==h,f
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');market=extended_market();ledger=load_market();history={}
    predictions=read(run/'predictions.json');previous=None;days=checks=actions=members=0;max_gradient=0.
    for i,(at,row) in enumerate(panel.iterrows()):
        oldpath=PARENT/'training'/(tag(at)+'.json');old=read(oldpath);record=read(run/'training'/(tag(at)+'.json'))
        assert record['parent_source_sha256']==sha(oldpath) and pd.Timestamp(record['at'])==at and pd.Timestamp(record['cutoff'])==day_cutoff(at)
        if previous!=row.date:
            saved=read(run/'daily'/(str(row.date.date())+'.json'));parents,fb=reference_parents({'semiconductor':panel},market,at)
            chosen=[q for q in parents if all(t in history and history[t]['events']['gap_blend']['ready'] for t in q['members'])][-120:]
            assert saved['parents']==canonical(chosen) and saved['feedback']==canonical(fb)
            assert saved['cutoff']==canonical(day_cutoff(at)) and saved['at']==canonical(at)
            refs={m:ref.data(panel,chosen,history,m) for m in MODES}
            population_ready=len(chosen)>=60 and refs['eligible'][3]['real_dates']>=30
            assert population_ready==saved['population_ready'];allfit=True
            for mode,(P,y,w,meta) in refs.items():
                model=saved['models'][mode]
                for k in ['dates','real_dates','members']:assert model[k]==canonical(meta[k])
                np.testing.assert_allclose(model['weights'],w,atol=1e-11,rtol=1e-11);np.testing.assert_allclose(model['date_ess'],meta['date_ess'],atol=1e-11,rtol=1e-11)
                unsupported=int(sum(P[j,y[j]]==0 for j in range(len(y))))
                fitready=population_ready and len(y)>0 and unsupported==0
                assert model['ready']==fitready;allfit=allfit and fitready;members+=len(y)
                if fitready:
                    loss,g=ref.objective(np.array(model['beta']),P,y,w);norm=float(max(abs(g)));assert norm<=1e-7
                    max_gradient=max(max_gradient,norm);np.testing.assert_allclose(model['objective'],loss,atol=1e-10,rtol=1e-10)
                    np.testing.assert_allclose(model['gradient_inf'],norm,atol=1e-10,rtol=1e-4);assert model['rows']==len(y);checks+=1
                elif not population_ready:assert model['reason']=='insufficient_dates'
                else:assert model['reason']=='issued_true_class_had_zero_support' and model['unsupported_rows']==unsupported
            ready=population_ready and allfit and not fb['expired_unresolved_prefix'];assert saved['ready']==ready
            previous=row.date;days+=1
        estimates={};src=old['events']['gap_blend'];info=old['estimates']['gap_blend']
        for mode in MODES+['identity']:
            e={'ready':False,'action':'CASH','allocation':0.,'probabilities':None,'values':None}
            if ready and src['ready']:
                j=0 if info['values'][0]>=info['values'][1] else 1
                q=ref.probabilities(info['probabilities'],[0]*4 if mode=='identity' else saved['models'][mode]['beta'])
                means=old['distributions']['gap']['payoff_means'];values=[sum(a*b for a,b in zip(q[s],means[s])) for s in range(2)]
                eligible=src['action'] in ['SOXL','SOXS'];buy=eligible and values[j]>1e-12
                e.update(ready=True,action=['SOXL','SOXS'][j] if buy else 'CASH',allocation=float(buy),probabilities=q,values=values,
                    top_symbol=['SOXL','SOXS'][j],buy_advantage=values[j],source_eligible=eligible)
                assert e['action']=='CASH' or e['action']==src['action']
            name=mode if mode=='identity' else mode+'_tilt';estimates[name]={'at':at,'date':row.date,'clock':row.clock,**e}
        expected=dict(estimates)
        for name,source in CONTROL_MAP.items():expected[name]=old['events'][source]
        cash={'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None}
        expected['cash']=cash;expected['double_cost']=dict(expected['eligible_tilt'])
        expected['premarket_only']=dict(expected['eligible_tilt']) if row.clock==CLOCKS[0] else cash
        for name,e in expected.items():
            got=record['events'][name];assert got==predictions[name][i]
            for k in ['at','date','clock','ready','action','allocation','top_symbol','source_eligible']:assert got.get(k)==canonical(e.get(k))
            for k in ['values','probabilities','buy_advantage']:
                if e.get(k) is None:assert got.get(k) is None
                else:np.testing.assert_allclose(got[k],e[k],atol=1e-11,rtol=1e-10)
            actions+=1
        history[at]=old
        if (i+1)%800==0:print('Verified tilts',i+1,'/',len(panel),at,flush=True)
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
    out={'passed':True,'results_sha256':sha(run/'results.json'),'prediction_rows':len(panel),'dates':days,'calibrator_gradient_checks':checks,
        'historical_candidate_members_checked':members,'maximum_gradient_inf':max_gradient,'route_actions':actions,'capital_books':books,
        'statistics_checks':stats,'four_parent_books_byte_identical':True,'source_sha256':sha(Path(__file__)),'development_gate':gate,
        'strategy_independent_validation':False,'goal_complete':False};dump(run/'verification.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);verify(p.parse_args().run)
