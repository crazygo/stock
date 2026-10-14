"""Independent paired fitting equations, candidate choices and capital audit."""
import argparse
from paired93 import *
from registration93 import verify_registration
from verify_transfer import canonical,reference_parents
from reference92 import certificate
import reference93 as ref
from audit_capital import book_check
from audit_statistics import audit_statistics


def verify(run):
    run=Path(run);registration=read(run/'registration.json');pre=registration['preregistration'];p=pre['protocol'];verify_registration(pre)
    assert registration['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in pre['input_files_sha256'].items():assert sha(run/f)==h,f
    for f in ['event_features.csv','fundamental_features.csv']:assert sha(run/f)==sha(PREVIOUS/f),f
    market=extended_market();panel=prepare_panel(pd.read_parquet(run/'semiconductor_panel.parquet'),market);ledger=load_market()
    predictions=read(run/'predictions.json');history={};previous_day=None;days=actions=checks=contexts=fits=members=0;maxgap=maxgradient=0.;projection_rows=[]
    for n,(at,row) in enumerate(panel.iterrows()):
        paths={k:root/'training'/(tag(at)+'.json') for k,root in [('parent',PARENT),('base',BASE),('wait',WAIT),('previous',PREVIOUS)]}
        original=read(paths['parent']);payoff=read(paths['base']);old=read(paths['previous']);got=read(run/'training'/(tag(at)+'.json'))
        for name,path in paths.items():assert got[name+'_source_sha256']==sha(path)
        assert pd.Timestamp(got['at'])==at and pd.Timestamp(got['cutoff'])==day_cutoff(at)
        day=str(row.date.date())
        if day!=previous_day:
            saved=read(run/'daily'/(day+'.json'));prior=read(PREVIOUS/'daily'/(day+'.json'))['base']
            parents,fb=reference_parents({'semiconductor':panel},market,at)
            chosen=[a for a in parents if all(t in history and history[t]['events']['recent_ratio']['ready'] for t in a['members'])][-120:]
            assert saved['parents']==canonical(chosen)==prior['parents'] and saved['feedback']==canonical(fb)
            assert pd.Timestamp(saved['at'])==at and pd.Timestamp(saved['cutoff'])==day_cutoff(at)
            eligible=sum(any(history[t]['events']['recent_ratio']['action'] in ['SOXL','SOXS'] for t in a['members']) for a in chosen)
            population_ready=len(chosen)>=60 and eligible>=30
            assert saved['eligible_dates']==eligible and saved['population_ready']==population_ready and saved['baseline_ready']==prior['ready']
            P,y,w,X,sides,clocks,meta=ref.data(panel,chosen,history,CLOCKS);groups={};allfit=True;values_by_mode={}
            for mode in PAIRED_MODELS:
                m=saved['models'][mode]
                for key in ['dates','real_dates','members']:assert m[key]==canonical(meta[key])
                np.testing.assert_allclose(m['weights'],w,atol=1e-11,rtol=1e-10);np.testing.assert_allclose(m['date_ess'],meta['date_ess'],atol=1e-11,rtol=1e-10)
                unsupported=int(sum(P[i,y[i]]==0 for i in range(len(y))));fitready=population_ready and len(y)>0 and not unsupported
                assert m['ready']==fitready;allfit=allfit and fitready;members+=len(y)
                if fitready:
                    Z,mu,sd=ref.matrix(X,sides,clocks,w,mode)
                    for key,value in [('center',mu),('scale',sd),('input_min',X.min(axis=0)),('input_max',X.max(axis=0))]:np.testing.assert_allclose(m[key],value,atol=1e-11,rtol=1e-10)
                    B=np.asarray(m['beta']);assert B.shape==(Z.shape[1],4)
                    loss,g=ref.loss_gradient(B.ravel(),P,y,w,Z);norm=float(max(abs(g)));assert norm<=1e-7;maxgradient=max(maxgradient,norm)
                    np.testing.assert_allclose(m['objective'],loss,atol=1e-10,rtol=1e-10);np.testing.assert_allclose(m['gradient_inf'],norm,atol=1e-10,rtol=1e-4)
                    assert m['rows']==len(y) and m['mode']==mode;fits+=1
                    scores=Z@B;values_by_mode[mode]=scores-scores.mean(axis=1)[:,None]
                elif not population_ready:assert m['reason']=='insufficient_dates'
                else:assert m['reason']=='issued_true_class_had_zero_support' and m['unsupported_rows']==unsupported
            ready=bool(population_ready and prior['ready'] and allfit and not fb['expired_unresolved_prefix']);assert saved['ready']==ready
            if ready:
                for mode in PAIRED_MODELS:
                    groups[mode]={}
                    for side in range(2):
                        for clock in range(8):
                            ids=[i for i in range(len(y)) if sides[i]==side and clocks[i]==clock]
                            mm=[meta['members'][i] for i in ids];dates=sorted({str(pd.Timestamp(t).tz_convert(core.ET).date()) for t,j in mm})
                            group=saved['supports'][mode][f'{side}:{clock}'];assert group['members']==canonical(mm) and group['dates']==dates
                            expected=values_by_mode[mode][ids].reshape(-1,4);np.testing.assert_allclose(np.asarray(group['corrections']).reshape(-1,4),expected,atol=1e-10,rtol=1e-10)
                            groups[mode][f'{side}:{clock}']=(expected,dates);contexts+=1
            else:assert saved['supports']=={}
            previous_day=day;days+=1
        source=original['events']['recent_ratio'];pair_expected={};estimates_ready=ready and source['ready'] and row.calendar_known and row.fundamental_known and payoff['distributions']['gap'] is not None
        if estimates_ready:
            means=np.asarray(payoff['distributions']['gap']['payoff_means']);Q=np.asarray(source['probabilities'])
            pair_expected['identity']=dict(probabilities=Q,values=np.sum(Q*means,axis=1),outside=[[],[]])
            for mode in PAIRED_MODELS:
                bounded=[];rawprob=[];proofs=[];outside=[];m=saved['models'][mode]
                for side in range(2):
                    clock=CLOCKS.index(row.clock);x=ref.coordinates(row,side);indices=range(19) if mode=='fundamental' else range(13)
                    z=[1.,float(side)]+[float(clock==k) for k in range(1,8)]+[(x[k]-m['center'][k])/m['scale'][k] for k in indices]
                    raw=np.array([sum(z[k]*m['beta'][k][j] for k in range(len(z))) for j in range(4)]);raw-=raw.mean()
                    proof=got['pair_estimates'][mode]['projections'][side];observed,dates=groups[mode][f'{side}:{clock}']
                    np.testing.assert_allclose(proof['raw'],raw,atol=1e-10,rtol=1e-10)
                    audit=certificate(raw,observed,proof);maxgap=max(maxgap,audit['optimality_gap']);checks+=1
                    assert proof['context_dates']==len(dates)
                    singular=np.linalg.svd(np.vstack([np.zeros(4),observed]),compute_uv=False);assert proof['rank']==sum(singular>1e-12*max(1.,singular[0]))
                    P=np.array(source['probabilities'][side]);c=np.array(proof['correction'])
                    q=P*np.exp(c-max(c));q/=sum(q);qr=P*np.exp(raw-max(raw));qr/=sum(qr)
                    bounded.append(q);rawprob.append(qr);proofs.append(proof)
                    outside.append([STATE_FIELDS[k] for k in indices if x[k]<m['input_min'][k]-1e-12 or x[k]>m['input_max'][k]+1e-12])
                    projection_rows.append(dict(at=at,date=day,model=mode,symbol=['SOXL','SOXS'][side],clock=row.clock,
                        context_dates=len(dates),context_rows=len(observed),distance=proof['distance'],optimality_gap=audit['optimality_gap'],
                        raw_norm=float(np.linalg.norm(raw)),bounded_norm=float(np.linalg.norm(c)),raw_value=float(qr@means[side]),bounded_value=float(q@means[side])))
                Q=np.array(bounded);pair_expected[mode]=dict(probabilities=Q,values=np.sum(Q*means,axis=1),projections=proofs,outside=outside)
                if mode=='fundamental':
                    Q=np.array(rawprob);pair_expected['raw_fundamental']=dict(probabilities=Q,values=np.sum(Q*means,axis=1),outside=outside)
            for key,expected in pair_expected.items():
                actual=got['pair_estimates'][key];assert set(actual)==set(expected)
                for field,value in expected.items():
                    if field in ['probabilities','values']:np.testing.assert_allclose(actual[field],value,atol=1e-11,rtol=1e-10)
                    else:assert actual[field]==canonical(value)
        assert set(got['pair_estimates'])==set(pair_expected)
        header=dict(at=at,date=row.date,clock=row.clock);expected=dict(old['events'])
        for name,(model,policy) in POLICIES.items():
            e=dict(ready=False,action='CASH',allocation=0.,probabilities=None,value=None)
            if pair_expected:
                pair=pair_expected[model];j=int(pair['values'][1]>pair['values'][0]) if policy=='reselect' else ['SOXL','SOXS'].index(source['top_symbol'])
                buy=pair['values'][j]>1e-12 and (policy!='fixed_source' or source['action'] in ['SOXL','SOXS'])
                e=dict(ready=True,action=['SOXL','SOXS'][j] if buy else 'CASH',allocation=float(buy),probabilities=pair['probabilities'][j],value=float(pair['values'][j]),
                    top_symbol=['SOXL','SOXS'][j],source_eligible=source['action'] in ['SOXL','SOXS'],source_top_symbol=source['top_symbol'],
                    source_value=source['values'][j],outside_state_fields=pair['outside'][j])
            expected[name]={**header,**e}
        expected['paired_premarket_only']=expected['paired_fundamental'] if row.clock==CLOCKS[0] else expected['cash'];expected['paired_double_cost']=expected['paired_fundamental']
        assert set(expected)==set(ROUTES)==set(got['events'])
        for name,e in expected.items():
            actual=got['events'][name];assert actual==predictions[name][n]
            if name in previous.ROUTES:assert actual==old['events'][name]
            assert set(actual)==set(e)
            for field,value in e.items():
                if field in ['probabilities','value'] and value is not None:np.testing.assert_allclose(actual[field],value,atol=1e-11,rtol=1e-10)
                else:assert actual[field]==canonical(value),(at,name,field)
            actions+=1
        history[at]=original
        if (n+1)%1200==0:print('Verified paired fits and actions',n+1,'/',len(panel),flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings['months']
    assert frozen['predictions_sha256']==sha(run/'predictions.json') and not frozen['new_capital_outcomes_seen']
    features=pd.read_parquet(run/'features.parquet');test=features[features.date.between(start,end)]
    ticks=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in ledger.sessions[(ledger.sessions>=start)&(ledger.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(ledger.calendar[d]['close_at'])]
    assert list(test.index)==ticks and result['coverage']['complete']
    baseline=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');books=[]
    for name in ROUTES:
        events=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert events==frozen['routes'][name]
        multiple=2 if name.endswith('double_cost') else 1
        books.append(book_check(run,name,ledger,test,stress if multiple==2 else baseline,events,multiple,result['routes'][name],months,require_full_cash=False))
    oldresult=read(PREVIOUS/'results.json')
    for name in previous.ROUTES:
        assert result['routes'][name]==oldresult['routes'][name]
        for suffix in ['equity.csv','trades.csv','decisions.csv']:assert sha(run/(name+'_'+suffix))==sha(PREVIOUS/(name+'_'+suffix))
    gate=development_gate(result['primary'],True,p);assert gate==result['development_gate']
    assert result['primary_name']==p['primary'] and result['primary']==result['routes'][p['primary']]
    assert not result['goal_complete'] and not result['case_regression']['executed']
    pd.DataFrame(projection_rows).to_csv(run/'projection_queries.csv',index=False)
    out=dict(passed=True,results_sha256=sha(run/'results.json'),prediction_rows=len(panel),dates=days,paired_model_gradient_checks=fits,
        maximum_gradient_inf=maxgradient,historical_candidate_members_checked=members,context_groups_checked=contexts,
        convex_projection_certificates=checks,maximum_optimality_gap=maxgap,route_actions=actions,capital_books=books,
        statistics_checks=audit_statistics(run),eighteen_parent_books_byte_identical=True,
        projection_rows_sha256=sha(run/'projection_queries.csv'),source_sha256=sha(Path(__file__)),development_gate=gate,goal_complete=False)
    dump(run/'verification.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);verify(p.parse_args().run)
