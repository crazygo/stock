"""Independent convex optimality, chronological expert mixtures and capital audit."""
import argparse,gzip,json,math
from threadpoolctl import threadpool_limits
from scipy.special import softmax
from class_learning import *
from reference_class import arrays,setup,evaluate
from verify_transfer import reference_parents,canonical
from audit_capital import book_check
from audit_statistics import audit_statistics


def reference_mixture(panel,parents,feedback,history,at,mode):
    selected=[r for r in parents if all(history.get(t) is not None for t in r['members'])][-120:];n=len(selected)
    out={'ready':n>=60,'cutoff':day_cutoff(at),'feedback':feedback,'parents':selected,'alpha':None}
    if not n:return out
    weights=np.array([2.**((j-n+1)/63) for j in range(n)]);weights*=n/sum(weights);differences=[]
    for r in selected:
        terms=[]
        for t in r['members']:
            for j,side in enumerate(['L','S']):
                y=int(panel.loc[t,side+'_'+mode]);assert y>=0
                p=history[t]['raw'][j][y];q=history[t]['prior'][j][y]
                terms.append(math.log(max(float(p),1e-8)/max(float(q),1e-8)))
        differences.append(sum(terms)/len(terms))
    odds=sum(float(w)*d for w,d in zip(weights,differences))/10
    alpha=math.exp(odds)/(1+math.exp(odds)) if odds<0 else 1/(1+math.exp(-odds))
    out.update(date_weights=weights,date_logscore_differences=differences,logodds=odds,alpha=alpha if n>=60 else None)
    return out


def verify(run):
    run=Path(run);registration=read(run/'registration.json');pre=registration['preregistration'];p=pre['protocol']
    assert registration['protocol_sha256']==sha(SOURCE/'protocol.json')==pre['protocol_sha256']
    for f,h in pre['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert pre['parent_delivery_sha256']==verify_parent();manifest=read(run/'input_manifest.json')
    for f,h in manifest['derived_files_sha256'].items():assert sha(run/f)==h,f
    for f in ['input_manifest','input_audit']:assert pre[f+'_sha256']==sha(run/(f+'.json'))
    assert read(run/'input_audit.json')['passed'];market=extended_market();ledger=load_market()
    origin=read(PARENT/'input_manifest.json');assert market.manifest==origin['expanded_market_manifest'] and ledger.manifest==origin['ledger_market_manifest']
    panels={k:pd.read_parquet(run/(k+'_panel.parquet')) for k in PAIRS};panel=panels['semiconductor']
    predictions=read(run/'predictions.json');hist={k:{} for k in MODES};previous=None;models_checked=days=mixtures=actions=0;max_gradient=0.
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            record=read(run/'training'/(tag(at)+'.json'));oldpath=PARENT/'training'/(tag(at)+'.json');old=read(oldpath)
            assert record['parent_source_sha256']==sha(oldpath) and pd.Timestamp(record['at'])==at and pd.Timestamp(record['cutoff'])==day_cutoff(at)
            if previous!=row.date:
                filename=str(row.date.date());saved=read(run/'daily'/(filename+'.json'))
                parents,feedback=reference_parents(panels,market,at);own,ownfb=reference_parents({'semiconductor':panel},market,at)
                assert saved['parents']==canonical(parents) and saved['feedback']==canonical(feedback)
                with gzip.open(run/'models'/(filename+'.json.gz'),'rt') as f:stored=json.load(f)
                refs={}
                for clock in CLOCKS:
                    ids=[t for r in parents for t in r['members'] if panel.loc[t,'clock']==clock] if len(parents)>=120 else []
                    assert saved['members'][clock]==canonical(ids);refs[clock]={}
                    if len(ids)<120:assert stored[clock]=={};continue
                    for mode in MODES:
                        model=stored[clock][mode];X,y,g,groups,dw,ess=arrays(panels,ids,FIELDS,mode);ref=setup(X,y,g,groups,dw,CLASSES[mode])
                        assert model['real_dates']==len(ids) and model['classes']==CLASSES[mode]
                        np.testing.assert_allclose(model['date_weights'],dw,atol=1e-11,rtol=1e-11)
                        np.testing.assert_allclose(model['date_ess'],ess,atol=1e-10,rtol=1e-11)
                        for k in ['center','scale','weights','prior','payoff_means','pooled_prior','pooled_payoff_sum','own_class_mass','own_payoff_sum','offset']:
                            np.testing.assert_allclose(model[k],ref[k],atol=1e-10,rtol=1e-10)
                        np.testing.assert_array_equal(model['active'],ref['active']);B=np.array(model['coefficients']);value,grad=evaluate(B,ref)
                        gnorm=float(abs(grad).max());assert gnorm<=1e-6,(filename,clock,mode,gnorm)
                        max_gradient=max(max_gradient,gnorm);np.testing.assert_allclose(model['objective'],value,atol=1e-10,rtol=1e-11)
                        np.testing.assert_allclose(model['gradient_inf'],gnorm,atol=1e-10,rtol=1e-5)
                        ref.update(coefficients=B,classes=CLASSES[mode]);refs[clock][mode]=ref;models_checked+=1
                daily={}
                for mode in MODES:
                    expected=reference_mixture(panel,own,ownfb,hist[mode],at,mode);got=saved['mixture'][mode]
                    for key in ['ready','cutoff','feedback','parents']:assert got[key]==canonical(expected[key])
                    for key in ['date_weights','date_logscore_differences','logodds','alpha']:
                        if expected.get(key) is None:assert got.get(key) is None
                        else:np.testing.assert_allclose(got[key],expected[key],atol=1e-10,rtol=1e-10)
                    daily[mode]=expected;mixtures+=1
                previous=row.date;days+=1
            estimates={};raw={}
            for mode in MODES:
                distribution=None
                if row.feature_ready and mode in refs[row.clock]:
                    model=refs[row.clock][mode];prob=[]
                    for g,sign in [(0,1),(1,-1)]:
                        x=row[FIELDS].to_numpy(float)[[0,1,2,3,4,5,6,7,8,9,10,16,17]].copy()
                        x[[0,1,2,5,6,7,8,9,10]]*=sign
                        a=np.r_[np.eye(6)[g],(x-model['center'])/model['scale']];pr=np.zeros(model['classes'])
                        pr[model['active']]=softmax(a@model['coefficients']+model['offset']);prob.append(pr)
                    distribution={'raw':np.array(prob),'prior':model['prior'][:2],'payoff_means':model['payoff_means'][:2]}
                got=record['distributions'][mode]
                if distribution is None:assert got is None
                else:
                    for field in distribution:np.testing.assert_allclose(got[field],distribution[field],atol=1e-10,rtol=1e-10)
                raw[mode]=distribution
                for variant in ['blend','raw','prior','equal']:
                    e={'ready':False,'probabilities':None,'values':None}
                    if distribution is not None and daily[mode]['ready'] and not daily[mode]['feedback']['expired_unresolved_prefix']:
                        alpha={'blend':daily[mode]['alpha'],'raw':1.,'prior':0.,'equal':.5}[variant]
                        prob=np.array([[alpha*a+(1-alpha)*b for a,b in zip(pa,pb)] for pa,pb in zip(distribution['raw'],distribution['prior'])])
                        values=np.array([sum(a*b for a,b in zip(pr,means)) for pr,means in zip(prob,distribution['payoff_means'])])
                        e.update(ready=True,probabilities=prob,values=values,alpha=alpha)
                    name=mode+'_'+variant;info=record['estimates'][name];assert info['ready']==e['ready']
                    for field in ['probabilities','values','alpha']:
                        if e.get(field) is None:assert info.get(field) is None
                        else:np.testing.assert_allclose(info[field],e[field],atol=1e-10,rtol=1e-10)
                    estimates[name]=e
                hist[mode][at]=distribution
            cash={'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None};expected={}
            for name,info in estimates.items():
                e={'ready':False,'action':'CASH','allocation':0.,'values':None}
                if info['ready'] and not feedback['expired_unresolved_prefix']:
                    v=info['values'];j=int(v[1]>v[0]);buy=v[j]>1e-12
                    e.update(ready=True,action=['SOXL','SOXS'][j] if buy else 'CASH',allocation=float(buy),values=v,
                        top_symbol=['SOXL','SOXS'][j],buy_advantage=float(v[j]))
                expected[name]={'at':at,'date':row.date,'clock':row.clock,**e}
            for name,source in CONTROL_MAP.items():expected[name]=old['events'][source]
            expected['cash']=cash;expected['double_cost']=dict(expected['gap_blend'])
            expected['premarket_only']=dict(expected['gap_blend']) if row.clock==CLOCKS[0] else dict(cash)
            for name,e in expected.items():
                got=record['events'][name];assert got==predictions[name][i]
                for key in ['at','date','clock','ready','action','allocation','top_symbol']:assert got.get(key)==canonical(e.get(key))
                for key in ['values','buy_advantage']:
                    if e.get(key) is None:assert got.get(key) is None
                    else:np.testing.assert_allclose(got[key],e[key],atol=1e-10,rtol=1e-10)
                actions+=1
            if (i+1)%400==0:print('Verified classes',i+1,'/',len(panel),at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json')
    assert frozen['predictions_sha256']==sha(run/'predictions.json') and not frozen['new_capital_outcomes_seen']
    settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings['months']
    features=pd.read_parquet(run/'features.parquet');test=features[features.date.between(start,end)]
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in ledger.sessions[(ledger.sessions>=start)&(ledger.sessions<=end)]
        for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(ledger.calendar[d]['close_at'])]
    assert list(test.index)==expected and result['coverage']['complete'];base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');books=[]
    for name in ROUTES:
        es=[e for e in predictions[name] if start<=str(pd.Timestamp(e['date']).date())<=end];assert frozen['routes'][name]==es
        books.append(book_check(run,name,ledger,test,stress if name=='double_cost' else base,es,2 if name=='double_cost' else 1,result['routes'][name],months,require_full_cash=False))
    stats=audit_statistics(run);oldresult=read(PARENT/'results.json')
    for name,parent_name in CONTROL_MAP.items():
        assert result['routes'][name]==oldresult['routes'][parent_name]
        for suffix in ['equity.csv','trades.csv','decisions.csv']:assert sha(run/(name+'_'+suffix))==sha(PARENT/(parent_name+'_'+suffix))
    gate=development_gate(result['primary'],True,p)
    assert result['primary_name']==p['primary'] and result['primary']==result['routes'][p['primary']] and result['development_gate']==gate
    assert not result['goal_complete'] and not result['case_regression']['executed']
    out={'passed':True,'results_sha256':sha(run/'results.json'),'prediction_rows':len(panel),'dates':days,'convex_model_gradient_checks':models_checked,
        'maximum_gradient_inf':max_gradient,'daily_mixtures_checked':mixtures,'route_actions':actions,'capital_books':books,'statistics_checks':stats,
        'four_old_controls_byte_identical':True,'source_sha256':sha(Path(__file__)),'development_gate':gate,'strategy_independent_validation':False,'goal_complete':False}
    dump(run/'verification.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);verify(p.parse_args().run)
