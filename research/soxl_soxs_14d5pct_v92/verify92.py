"""Independent context/convex-optimality/action audit plus complete capital checks."""
import argparse
from bounded92 import *
from registration92 import verify_registration
from verify_transfer import canonical
from reference92 import certificate
import reference91 as ref
from audit_capital import book_check
from audit_statistics import audit_statistics


def verify(run):
    run=Path(run);registration=read(run/'registration.json');pre=registration['preregistration'];p=pre['protocol'];verify_registration(pre)
    assert registration['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in pre['input_files_sha256'].items():assert sha(run/f)==h,f
    for f in ['event_features.csv','fundamental_features.csv']:assert sha(run/f)==sha(PREVIOUS/f),f
    panel=prepare_panel(pd.read_parquet(run/'semiconductor_panel.parquet'),extended_market());ledger=load_market()
    predictions=read(run/'predictions.json');history={};previous_day=None;days=actions=checks=contexts=oldfits=0;maxgap=0.;projection_rows=[]
    for n,(at,row) in enumerate(panel.iterrows()):
        oldpaths={k:root/'training'/(tag(at)+'.json') for k,root in [('parent',PARENT),('base',BASE),('wait',WAIT),('previous',PREVIOUS)]}
        original=read(oldpaths['parent']);payoff=read(oldpaths['base']);old=read(oldpaths['previous']);got=read(run/'training'/(tag(at)+'.json'))
        for name,path in oldpaths.items():assert got[name+'_source_sha256']==sha(path)
        assert pd.Timestamp(got['at'])==at and pd.Timestamp(got['cutoff'])==day_cutoff(at)
        day=str(row.date.date())
        if day!=previous_day:
            saved=read(run/'daily'/(day+'.json'));daily=saved['base'];assert daily==read(PREVIOUS/'daily'/(day+'.json'))
            groups={};oldfits+=sum(m['ready'] for m in daily['models'].values())
            if daily['ready']:
                parents=[{**p,'members':[pd.Timestamp(t) for t in p['members']]} for p in daily['parents']]
                arrays={pop:ref.data(panel,parents,history,pop,CLOCKS) for pop in ['eligible','top']}
                for name,parent_name in BMODELS.items():
                    model=daily['models'][parent_name];pop,mode=MODELS[parent_name];P,y,w,X,sides,clocks,meta=arrays[pop]
                    Z,mu,sd=ref.matrix(X,sides,clocks,w,mode);assert canonical(meta['members'])==model['members']
                    np.testing.assert_allclose(mu,model['center'],atol=1e-11,rtol=1e-10);np.testing.assert_allclose(sd,model['scale'],atol=1e-11,rtol=1e-10)
                    values=Z@np.asarray(model['beta']);values-=values.mean(axis=1)[:,None];groups[name]={}
                    for side in range(2):
                        for clock in range(8):
                            ids=[i for i in range(len(y)) if sides[i]==side and clocks[i]==clock]
                            members=[meta['members'][i] for i in ids];dates=sorted({str(pd.Timestamp(t).tz_convert(core.ET).date()) for t,j in members})
                            group=saved['supports'][name][f'{side}:{clock}'];assert group['members']==canonical(members) and group['dates']==dates
                            expected=values[ids].reshape(-1,4);np.testing.assert_allclose(np.asarray(group['corrections']).reshape(-1,4),expected,atol=1e-10,rtol=1e-10)
                            groups[name][f'{side}:{clock}']=(expected,dates);contexts+=1
            else:assert saved['supports']=={}
            previous_day=day;days+=1
        expected=dict(old['events']);source=original['events']['recent_ratio']
        for name,parent_name in BMODELS.items():
            e=dict(expected[parent_name])
            if e['ready']:
                side=['SOXL','SOXS'].index(e['top_symbol']);clock=CLOCKS.index(row.clock);m=daily['models'][parent_name]
                x=ref.coordinates(row,side);indices=list(range(19)) if m['mode']=='fundamental' else list(range(13))
                z=[1.,float(side)]+[float(clock==k) for k in range(1,8)]+[(x[k]-m['center'][k])/m['scale'][k] for k in indices]
                raw=np.array([sum(z[k]*m['beta'][k][j] for k in range(len(z))) for j in range(4)]);raw-=raw.mean()
                info=got['events'][name]['projection'];observed,dates=groups[name][f'{side}:{clock}']
                np.testing.assert_allclose(info['raw'],raw,atol=1e-10,rtol=1e-10)
                audit=certificate(raw,observed,info);maxgap=max(maxgap,audit['optimality_gap']);checks+=1
                assert info['parent_model']==parent_name and info['context_dates']==len(dates)
                singular=np.linalg.svd(np.vstack([np.zeros(4),observed]),compute_uv=False)
                assert info['rank']==sum(singular>1e-12*max(1.,singular[0]))
                c=np.asarray(info['correction']);P=np.asarray(source['probabilities'][side]);q=P*np.exp(c-max(c));q/=sum(q)
                value=float(sum(a*b for a,b in zip(q,payoff['distributions']['gap']['payoff_means'][side])))
                buy=e['source_eligible'] and value>1e-12
                e.update(probabilities=q,value=value,action=e['top_symbol'] if buy else 'CASH',allocation=float(buy),projection=info)
                projection_rows.append(dict(at=at,date=day,route=name,symbol=e['top_symbol'],clock=row.clock,source_eligible=e['source_eligible'],
                    context_dates=len(dates),context_rows=len(observed),rank=info['rank'],distance=info['distance'],optimality_gap=audit['optimality_gap'],
                    raw_norm=float(np.linalg.norm(raw)),bounded_norm=float(np.linalg.norm(c)),zero_anchor_weight=info['zero_anchor_weight'],
                    raw_value=old['events'][parent_name]['value'],bounded_value=value,raw_action=old['events'][parent_name]['action'],bounded_action=e['action']))
            expected[name]=e
        expected['bounded_premarket_only']=expected['bounded_fundamental'] if row.clock==CLOCKS[0] else expected['cash']
        expected['bounded_double_cost']=expected['bounded_fundamental']
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
        if (n+1)%1600==0:print('Verified bounded actions',n+1,'/',len(panel),flush=True)
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
    out=dict(passed=True,results_sha256=sha(run/'results.json'),prediction_rows=len(panel),dates=days,old_daily_fits_exact=oldfits,
        context_groups_checked=contexts,convex_projection_certificates=checks,maximum_optimality_gap=maxgap,route_actions=actions,
        capital_books=books,statistics_checks=audit_statistics(run),thirteen_parent_books_byte_identical=True,
        projection_rows_sha256=sha(run/'projection_queries.csv'),source_sha256=sha(Path(__file__)),development_gate=gate,goal_complete=False)
    dump(run/'verification.json',out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);verify(p.parse_args().run)
