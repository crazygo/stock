"""Independent paired labels, normal equations, actions and complete capital audit."""
import argparse
from wait_learning import *
from verify_transfer import canonical
from audit_capital import book_check
from audit_statistics import audit_statistics
import reference_wait as ref
from verification_format import serialized_field


def verify(run):
    run=Path(run);registration=read(run/'registration.json');pre=registration['preregistration'];p=pre['protocol']
    assert pre['protocol_sha256']==registration['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in pre['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert pre['parent_delivery_sha256']==verify_parent()
    for f,h in pre['input_files_sha256'].items():assert sha(run/f)==h,f
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');market=extended_market();ledger=load_market()
    records={at:read(PARENT/'training'/(tag(at)+'.json')) for at in panel.index};bank=ref.bank(panel,market,records,CLOCKS)
    assert read(run/'paired_registry.json')==canonical(bank)
    predictions=read(run/'predictions.json');previous=None;days=checks=actions=members=0;max_residual=0.;negative_wait=cash_wait=0
    for i,(at,row) in enumerate(panel.iterrows()):
        oldpath=PARENT/'training'/(tag(at)+'.json');old=records[at];record=read(run/'training'/(tag(at)+'.json'))
        assert record['parent_source_sha256']==sha(oldpath) and pd.Timestamp(record['at'])==at and pd.Timestamp(record['cutoff'])==day_cutoff(at)
        if previous!=row.date:
            saved=read(run/'daily'/(str(row.date.date())+'.json'));parents,fb=ref.parents(bank,day_cutoff(at))
            assert saved['parents']==canonical(parents) and saved['feedback']==canonical(fb)
            assert saved['at']==canonical(at) and saved['cutoff']==canonical(day_cutoff(at))
            ready=len(parents)>=60 and not fb['expired_unresolved_prefix'];assert saved['ready']==ready;models={}
            if ready:
                X,Y,c,w,meta=ref.data(panel,parents,records,FIELDS,CLOCKS);assert np.isfinite(Y).all()
                assert all(q['release']<day_cutoff(at) and q['next_at']<day_cutoff(at) for q in parents)
                for key in ['dates','real_dates','members']:assert saved['metadata'][key]==canonical(meta[key])
                for key in ['date_ess','date_weights']:np.testing.assert_allclose(saved['metadata'][key],meta[key],atol=1e-11,rtol=1e-10)
                for mode in VARIANTS:
                    expected,A=ref.solve(X,Y,c,w,mode);m=saved['models'][mode];assert m['mode']==mode and m['ids']==expected['ids'] and m['rows']==len(Y)
                    for key in ['center','scale','design_center','intercept','coefficients','penalty','weights']:np.testing.assert_allclose(serialized_field(m,key),expected[key],atol=1e-11,rtol=1e-10)
                    coef=np.asarray(m['coefficients']).reshape(-1,2);Z=A-m['design_center'];res=Y-m['intercept']-Z@coef
                    g=Z.T@(w[:,None]*res)-np.asarray(m['penalty'])[:,None]*coef;norm=float(max(abs(g).flat)) if g.size else 0.
                    assert norm<1e-9 and max(abs(w@res))<1e-9;max_residual=max(max_residual,norm);checks+=1
                    models[mode]=expected
                members+=len(Y);negative_wait+=int((Y[:,1]<0).sum());cash_wait+=sum(m['wait_action']=='CASH' for m in meta['members'])
            else:assert saved['models']=={} and saved['metadata']=={}
            previous=row.date;days+=1
        src=old['events']['recent_ratio'];current_ready=ready and src['ready'];base={'at':at,'date':row.date,'clock':row.clock}
        cash={**base,'ready':True,'action':'CASH','allocation':0.,'values':None};inactive={**cash,'ready':False};expected={}
        for mode in VARIANTS:
            e=dict(inactive)
            if current_ready:
                x,j=ref.x(row,src,FIELDS);values=ref.forecast(models[mode],x,CLOCKS.index(row.clock));delta=values[0]-values[1]
                buy=src['action'] in core.ETFS and delta>1e-12
                e={**base,'ready':True,'action':core.ETFS[j] if buy else 'CASH','allocation':float(buy),'values':values,
                    'advantage':float(delta),'top_symbol':src['top_symbol'],'source_eligible':src['action'] in core.ETFS,'source_value':src['values'][j]}
            expected['paired_'+mode]=e
        for name,compare in [('absolute_pair',True),('absolute_now',False)]:
            e=dict(expected['paired_state'])
            if current_ready:
                buy=e['source_eligible'] and e['values'][0]>1e-12 and (e['advantage']>1e-12 or not compare)
                e.update(action=e['top_symbol'] if buy else 'CASH',allocation=float(buy))
            expected[name]=e
        expected['source_matched']=src if current_ready else inactive
        for name,source in CONTROL_MAP.items():expected[name]=old['events'][source]
        expected['cash']=cash;expected['double_cost']=dict(expected['paired_state']);expected['premarket_only']=dict(expected['paired_state']) if row.clock==CLOCKS[0] else cash
        for name,e in expected.items():
            got=record['events'][name];assert got==predictions[name][i]
            for k in ['at','date','clock','ready','action','allocation','top_symbol','source_eligible']:assert got.get(k)==canonical(e.get(k))
            for k in ['values','probabilities','advantage','source_value']:
                if e.get(k) is None:assert got.get(k) is None
                else:np.testing.assert_allclose(got[k],e[k],atol=1e-11,rtol=1e-10)
            if name.startswith('paired_') and got['action']!='CASH':assert got['action']==src['action']
            actions+=1
        if (i+1)%800==0:print('Verified waiting',i+1,'/',len(panel),at,flush=True)
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
    out={'passed':True,'results_sha256':sha(run/'results.json'),'prediction_rows':len(panel),'dates':days,'ridge_systems_checked':checks,
        'historical_pair_members_checked':members,'negative_wait_rows_retained':negative_wait,'cash_wait_rows_retained':cash_wait,
        'maximum_normal_equation_residual':max_residual,'route_actions':actions,'capital_books':books,'statistics_checks':stats,
        'four_parent_books_byte_identical':True,'source_sha256':sha(Path(__file__)),'development_gate':gate,
        'strategy_independent_validation':False,'goal_complete':False};dump(run/'verification.json',out);print(out,flush=True)



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);verify(p.parse_args().run)
