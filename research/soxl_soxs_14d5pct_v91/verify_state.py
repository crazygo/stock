"""Independent whole-date samples, convex equations, forecasts and capital audit."""
import argparse
from calibration91 import *
from verify_transfer import reference_parents,canonical
from audit_capital import book_check
from audit_statistics import audit_statistics
import reference91 as ref
from verify_calendar import reference as calendar_reference
from reference_reaction import verify_reaction_table
from reference_features91 import verify_table
from features91 import DATA


def verify(run):
    run=Path(run);reg=read(run/'registration.json');pre=reg['preregistration'];p=pre['protocol']
    assert pre['protocol_sha256']==reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    from registration91 import verify_registration
    verify_registration(pre)
    for f,h in pre['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert pre['parent_delivery_sha256']==verify_parent()
    for f,h in pre['input_files_sha256'].items():assert sha(run/f)==h,f
    panel=pd.read_parquet(run/'semiconductor_panel.parquet');market=extended_market();ledger=load_market();history={};panel=prepare_panel(panel,market)
    calendar=read(CATALOG);event_table=pd.read_csv(run/'event_features.csv');assert len(event_table)==len(panel)
    reaction_check=verify_reaction_table(event_table,panel)
    fundamental_check=verify_table(pd.read_csv(run/'fundamental_features.csv'),panel,read(DATA))
    predictions=read(run/'predictions.json');previous=None;days=checks=actions=members=0;max_gradient=0.
    for i,(at,row) in enumerate(panel.iterrows()):
        calendar_row=event_table.iloc[i];assert pd.Timestamp(calendar_row['at'])==at and calendar_row.known
        calendar_days=[d for d in market.sessions if row.date<=d<=row.date+pd.Timedelta(days=13)]
        expiry=pd.Timestamp(market.calendar[max(calendar_days)]['close_at']);assert pd.Timestamp(calendar_row.expiry)==expiry
        reference_event=calendar_reference(calendar,at,expiry)
        assert reference_event['known']
        names=['event_future_count','event_future_proximity','event_recent_count','event_recent_proximity']
        np.testing.assert_allclose([calendar_row[k] for k in names],reference_event['values'],atol=1e-14,rtol=0)
        np.testing.assert_allclose([row[k] for k in names],reference_event['values'],atol=1e-14,rtol=0)
        for field in ['future_days','previous_days','visible_record_ids']:
            assert json.loads(calendar_row[field])==reference_event[field]
        paths={v:folder/'training'/(tag(at)+'.json') for v,folder in [(80,PARENT),(78,BASE),(81,WAIT)]}
        old=read(paths[80]);base=read(paths[78]);wait=read(paths[81]);record=read(run/'training'/(tag(at)+'.json'))
        for key,v in [('parent_source_sha256',80),('base_source_sha256',78),('wait_source_sha256',81)]:assert record[key]==sha(paths[v])
        assert old['base_source_sha256']==sha(paths[78])
        assert pd.Timestamp(record['at'])==at and pd.Timestamp(record['cutoff'])==day_cutoff(at)
        if previous!=row.date:
            saved=read(run/'daily'/(str(row.date.date())+'.json'));parents,fb=reference_parents({'semiconductor':panel},market,at)
            chosen=[q for q in parents if all(t in history and history[t]['events']['recent_ratio']['ready'] for t in q['members'])][-120:]
            assert saved['parents']==canonical(chosen) and saved['feedback']==canonical(fb)
            assert saved['cutoff']==canonical(day_cutoff(at)) and saved['at']==canonical(at)
            arrays={pop:ref.data(panel,chosen,history,pop,CLOCKS) for pop in ['eligible','top']}
            population_ready=len(chosen)>=60 and arrays['eligible'][-1]['real_dates']>=30
            assert population_ready==saved['population_ready'];allfit=True
            for name,(population,mode) in MODELS.items():
                P,y,w,X,sides,clocks,meta=arrays[population];model=saved['models'][name]
                for key in ['dates','real_dates','members']:assert model[key]==canonical(meta[key])
                np.testing.assert_allclose(model['weights'],w,atol=1e-11,rtol=1e-11);np.testing.assert_allclose(model['date_ess'],meta['date_ess'],atol=1e-11,rtol=1e-11)
                unsupported=sum(P[j,y[j]]==0 for j in range(len(y)))
                fitready=population_ready and len(y)>0 and unsupported==0
                assert model['ready']==fitready;allfit=allfit and fitready;members+=len(y)
                if fitready:
                    Z,center,scale=ref.matrix(X,sides,clocks,w,mode)
                    for key,value in [('center',center),('scale',scale),('input_min',X.min(axis=0)),('input_max',X.max(axis=0))]:np.testing.assert_allclose(model[key],value,atol=1e-11,rtol=1e-10)
                    B=np.asarray(model['beta']);assert B.shape==(Z.shape[1],4)
                    loss,g=ref.loss_gradient(B.ravel(),P,y,w,Z);norm=float(max(abs(g)));assert norm<=1e-7
                    max_gradient=max(max_gradient,norm);np.testing.assert_allclose(model['objective'],loss,atol=1e-10,rtol=1e-10)
                    np.testing.assert_allclose(model['gradient_inf'],norm,atol=1e-10,rtol=1e-4)
                    assert model['rows']==len(y) and model['mode']==mode;checks+=1
                elif not population_ready:assert model['reason']=='insufficient_dates'
                else:assert model['reason']=='issued_true_class_had_zero_support' and model['unsupported_rows']==unsupported
            ready=population_ready and allfit and not fb['expired_unresolved_prefix'];assert saved['ready']==ready
            previous=row.date;days+=1
        src=old['events']['recent_ratio'];header={'at':at,'date':row.date,'clock':row.clock}
        cash={**header,'ready':True,'action':'CASH','allocation':0.,'probabilities':None,'value':None};expected={}
        for name in list(MODELS)+['identity_matched']:
            e={**cash,'ready':False}
            if ready and src['ready'] and row.calendar_known and row.fundamental_known:
                j=['SOXL','SOXS'].index(src['top_symbol']);x=ref.coordinates(row,j);P=src['probabilities'][j]
                q=np.array(P) if name=='identity_matched' else ref.predict(saved['models'][name],P,x,j,CLOCKS.index(row.clock))
                value=float(sum(a*b for a,b in zip(q,base['distributions']['gap']['payoff_means'][j])))
                buy=src['action'] in ['SOXL','SOXS'] and value>1e-12
                outside=[] if name=='identity_matched' else [STATE_FIELDS[k] for k in (range(19) if saved['models'][name]['mode']=='fundamental' else range(13) if saved['models'][name]['mode']=='reaction' else range(9) if saved['models'][name]['mode']=='event' else range(5)) if x[k]<saved['models'][name]['input_min'][k]-1e-12 or x[k]>saved['models'][name]['input_max'][k]+1e-12]
                e.update(ready=True,action=src['top_symbol'] if buy else 'CASH',allocation=float(buy),probabilities=q,value=value,
                         top_symbol=src['top_symbol'],source_eligible=src['action'] in ['SOXL','SOXS'],source_value=src['values'][j],outside_state_fields=outside)
                assert e['action']=='CASH' or e['action']==src['action']
            expected[name]=e
        for name,source in CONTROL_MAP.items():expected[name]=old['events'][source]
        expected['source_wait']=wait['events']['paired_state'];expected['cash']=cash;expected['double_cost']=dict(expected['eligible_fundamental'])
        expected['premarket_only']=dict(expected['eligible_fundamental']) if row.clock==CLOCKS[0] else cash
        for name,e in expected.items():
            got=record['events'][name];assert got==predictions[name][i]
            for key in ['at','date','clock','ready','action','allocation','top_symbol','source_eligible','outside_state_fields']:assert got.get(key)==canonical(e.get(key))
            for key in ['value','values','probabilities','source_value']:
                if e.get(key) is None:assert got.get(key) is None
                else:np.testing.assert_allclose(got[key],e[key],atol=1e-11,rtol=1e-10)
            actions+=1
        history[at]=old
        if (i+1)%800==0:print('Verified state probabilities',i+1,'/',len(panel),at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json')
    assert frozen['predictions_sha256']==sha(run/'predictions.json') and not frozen['new_capital_outcomes_seen']
    settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings['months']
    features=pd.read_parquet(run/'features.parquet');test=features[features.date.between(start,end)]
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in ledger.sessions[(ledger.sessions>=start)&(ledger.sessions<=end)]
        for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(ledger.calendar[d]['close_at'])]
    assert list(test.index)==expected and result['coverage']['complete']
    baseline=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');books=[]
    for name in ROUTES:
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert frozen['routes'][name]==es
        books.append(book_check(run,name,ledger,test,stress if name=='double_cost' else baseline,es,2 if name=='double_cost' else 1,result['routes'][name],months,require_full_cash=False))
    stats=audit_statistics(run)
    for name,oldname in {**CONTROL_MAP,'source_wait':'paired_state'}.items():
        parent_run=WAIT if name=='source_wait' else PARENT
        assert result['routes'][name]==read(parent_run/'results.json')['routes'][oldname]
        for suffix in ['equity.csv','trades.csv','decisions.csv']:assert sha(run/(name+'_'+suffix))==sha(parent_run/(oldname+'_'+suffix))
    for suffix in ['equity.csv','trades.csv','decisions.csv']:
        assert sha(run/('eligible_event_'+suffix))==sha(PRIOR/('eligible_event_'+suffix))
    assert result['routes']['eligible_event']==read(PRIOR/'results.json')['routes']['eligible_event']
    for suffix in ['equity.csv','trades.csv','decisions.csv']:
        assert sha(run/('eligible_state_'+suffix))==sha(PRICE_PRIOR/('eligible_state_'+suffix))
    assert result['routes']['eligible_state']==read(PRICE_PRIOR/'results.json')['routes']['eligible_state']
    for suffix in ['equity.csv','trades.csv','decisions.csv']:
        assert sha(run/('eligible_reaction_'+suffix))==sha(REACTION_PRIOR/('eligible_reaction_'+suffix))
    assert result['routes']['eligible_reaction']==read(REACTION_PRIOR/'results.json')['routes']['eligible_reaction']
    gate=development_gate(result['primary'],True,p);assert gate==result['development_gate']
    assert result['primary_name']==p['primary'] and result['primary']==result['routes'][p['primary']]
    assert not result['goal_complete'] and not result['case_regression']['executed']
    out=dict(passed=True,results_sha256=sha(run/'results.json'),prediction_rows=len(panel),dates=days,calibrator_gradient_checks=checks,
             historical_candidate_members_checked=members,maximum_gradient_inf=max_gradient,route_actions=actions,capital_books=books,
             statistics_checks=stats,seven_parent_books_byte_identical=True,fundamental_input_checks=fundamental_check,reaction_input_checks=reaction_check,source_sha256=sha(Path(__file__)),development_gate=gate,
             strategy_independent_validation=False,goal_complete=False)
    dump(run/'verification.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);verify(p.parse_args().run)
