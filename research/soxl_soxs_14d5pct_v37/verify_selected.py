"""Independent selected features, convex fit equations, actions and capital."""
import argparse,json
from threadpoolctl import threadpool_limits
from selected_risk import core,np,pd,Path,SOURCE,read,dump,sha,CLOCKS,tag,restrict_market,labels
from audit_shared import reference_pool
from verify_reliability import close
from verify_continuation import near
from audit_books import book_check
from run_planning import json_ready

canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))
SCORE_COLUMNS=['top_lower','top_gap','raw_direction_agreement','is_soxs']+['clock_'+c for c in CLOCKS[1:]]
STATE_COLUMNS=['aligned_relative5','aligned_breadth5','aligned_volume_diffusion','volatility','aligned_growth_acceleration','recent_earnings']
cols=lambda k:SCORE_COLUMNS+STATE_COLUMNS if k=='context' else SCORE_COLUMNS if k=='scores' else []


def verify_panel(original,panel,records):
    pd.testing.assert_frame_equal(panel[original.columns],original);count=0
    for at,row in panel.iterrows():
        r=records[at];v=r['estimates']['full'];assert row.top_ready==v['ready']
        if not v['ready']:
            assert pd.isna(row.selected_y) and pd.isna(row.top_symbol) and row[SCORE_COLUMNS+STATE_COLUMNS].isna().all();continue
        j=int(v['point'][1]>v['point'][0]);direction=1-2*j;assert row.top_symbol==core.ETFS[j]
        if np.isfinite(original.loc[at,['C','D']].to_numpy(float)).all():near(row.selected_y,original.loc[at,'C']+direction*original.loc[at,'D'])
        else:assert pd.isna(row.selected_y)
        expected=[v['lower'][j],abs(v['point'][0]-v['point'][1]),direction*(r['raws']['full'][0]-r['raws']['full'][1]),float(j)]
        expected += [float(row.clock==c) for c in CLOCKS[1:]]
        expected += [direction*original.loc[at,c] for c in ['relative_5','breadth_5','volume_direction_diffusion']]
        expected += [original.loc[at,'log_sector_volatility'],direction*original.loc[at,'growth_acceleration'],original.loc[at,'recent_earnings_share']]
        close(row[SCORE_COLUMNS+STATE_COLUMNS].to_numpy(float),expected);count+=14
    return count


def fit_audit(model,X,returns,w,n):
    keep=np.asarray(w)>0;X=np.array(X)[keep];r=np.array(returns)[keep];w=np.array(w)[keep];w*=n/w.sum()
    mean=np.sum(w[:,None]*X,axis=0)/sum(w);sd=np.sqrt(np.sum(w[:,None]*(X-mean)**2,axis=0)/sum(w));sd[sd<1e-12]=1
    close(model['center'],mean);close(model['scale'],sd)
    m=w*np.abs(r);total=sum(m);positive=sum(m[r>0]);q=positive/total if total else .5
    near(model['positive_share'],q);near(model['magnitude_total'],total);near(model['mean_return'],sum(w*r)/sum(w));near(model['raw_win_fraction'],sum(w*(r>0))/sum(w));near(model['weight_sum'],n)
    if q in [0,1] or total<=1e-16:
        assert model['mode']=='constant' and model['iterations']==0;near(model['quality'],q);close(model['coef'],np.zeros(X.shape[1]));near(model['intercept'],0)
    else:
        assert model['mode']=='logit';m*=n/total;Z=(X-mean)/sd;b=np.array(model['coef']);z=Z@b+model['intercept']
        prob=np.where(z>=0,1/(1+np.exp(-np.abs(z))),np.exp(-np.abs(z))/(1+np.exp(-np.abs(z))))
        residual=m*(prob-(r>0));gradient=np.r_[sum(residual),Z.T@residual+10*b]
        assert np.max(np.abs(gradient))<2e-8,(np.max(np.abs(gradient)),model['gradient_max'])
        near(model['gradient_max'],np.max(np.abs(gradient)));assert model['iterations']<100
        A=np.column_stack((np.ones(len(X)),Z));h=A.T@((m*prob*(1-prob))[:,None]*A)+np.diag(np.r_[0.,np.repeat(10.,X.shape[1])])
        assert np.linalg.eigvalsh(h)[0]>0 # Unique global optimum for the convex objective.
    return 1


def reference_quality(model,x):
    if model['mode']=='constant':return model['quality']
    z=model['intercept']+sum((x[i]-model['center'][i])/model['scale'][i]*model['coef'][i] for i in range(len(x)))
    return 1/(1+np.exp(-z)) if z>=0 else np.exp(z)/(1+np.exp(z))


def daily_audit(panel,market,records,at,saved):
    cutoff=pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
    history={t:r['raws']['full'] if r['estimates']['full']['ready'] else None for t,r in records.items() if t<cutoff}
    parents,fb=reference_pool(panel,market,history,at);assert saved['parents']==canonical(parents) and saved['feedback']==canonical(fb)
    ids=[t for p in parents for t in p['members']];assert list(pd.to_datetime(saved['members'],utc=True))==ids
    assert pd.Timestamp(saved['cutoff'])==cutoff and pd.Timestamp(saved['date'])==panel.loc[at,'date']
    n=len(parents);assert saved['ready']==(n>=60)
    if n<60:assert saved['models']=={} and saved['bootstrap']==[];return 0
    dw=np.array([2**((i-n+1)/63) for i in range(n)]);dw*=n/dw.sum()
    rw=np.array([dw[i]/len(p['members']) for i,p in enumerate(parents) for _ in p['members']]);groups=np.array([i for i,p in enumerate(parents) for _ in p['members']])
    close(saved['row_weights'],rw);assert saved['row_groups']==groups.tolist()
    r=panel.loc[ids,'selected_y'].to_numpy(float);X={k:panel.loc[ids,cols(k)].to_numpy(float) for k in ['context','scores','frequency']}
    checks=sum(fit_audit(saved['models'][k],x,r,rw,n) for k,x in X.items())
    seed=826+at.tz_convert(core.ET).date().toordinal();assert saved['seed']==seed
    starts=np.random.default_rng(seed).integers(0,n,size=(199,(n+9)//10));assert saved['block_starts']==starts.tolist()
    bydate=[np.flatnonzero(groups==j) for j in range(n)]
    for b,draw in enumerate(starts):
        days=[(int(s)+j)%n for s in draw for j in range(10)][:n];indices=np.concatenate([bydate[j] for j in days])
        # Expand repeated dates, independently checking the count-weight shortcut.
        for k,x in X.items():checks+=fit_audit(saved['bootstrap'][b][k],x[indices],r[indices],rw[indices],n)
    return checks


def check_event(name,e,record,daily,estimates):
    old=record['events']['full_band']
    if name=='source_v35':assert e==old;return
    if name=='cash':assert e['action']=='CASH' and e['allocation']==0.;return
    kind={'scores_gate':'scores','frequency_gate':'frequency'}.get(name,'context');info=estimates[kind]
    ready=info['ready'] and not daily['feedback']['expired_unresolved_prefix'];assert e['ready']==ready
    q=None;reason='unready';take=False
    if ready:
        if info['source_fallback']:take=old['action'] in core.ETFS;reason='source_warmup'
        else:
            q=info['lower_quality'] if name=='context_band' else info['quality']
            take=q>.5+1e-12 and (name=='standalone' or old['action'] in core.ETFS)
            reason='positive_gain_loss_balance' if take else 'rejected'
    point=record['estimates']['full']['point'];j=int(point is not None and point[1]>point[0])
    assert e['action']==(core.ETFS[j] if take else 'CASH') and e['allocation']==float(take)
    assert e['reason']==reason and e['quality_used']==q


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    root=Path(manifest['verified_source']);delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v35/evidence/delivery_manifest.json')
    want={str(Path(folder)/f):h for field,folder in [('cache_outputs_sha256',''),('daily_records_sha256','daily'),('training_records_sha256','training')] for f,h in delivery[field].items()}
    assert manifest['source_files_sha256']==want
    for f,h in want.items():assert sha(root/f)==h
    oldv=read(root/'verification.json');assert oldv['passed'] and oldv['results_sha256']==sha(root/'results.json')
    market=core.load_market();assert market.manifest==manifest['market']['full_market']
    market=restrict_market(market,pd.Timestamp(manifest['market']['observation_cutoff_exclusive']))
    original=pd.read_parquet(root/'financial_panel.parquet');panel=pd.read_parquet(run/'selected_panel.parquet');features=pd.read_parquet(root/'features.parquet')
    records={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))}
    feature_checks=verify_panel(original,panel,records);predictions=read(run/'predictions.json');previous=None;daily=None;checks=actions=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            if row.date!=previous:
                daily=read(run/'daily'/(str(row.date.date())+'.json'));checks+=daily_audit(panel,market,records,at,daily);previous=row.date
            saved=read(run/'training'/(tag(at)+'.json'));record=records[at]
            assert saved['source_record_sha256']==sha(root/'training'/(tag(at)+'.json')) and pd.Timestamp(saved['at'])==at and saved['clock']==row.clock
            assert pd.Timestamp(saved['date'])==row.date and pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
            ready=bool(record['estimates']['full']['ready'] and panel.loc[at,'feature_ready'])
            for k,info in saved['estimates'].items():
                assert info['ready']==ready and info['source_fallback']==(ready and not daily['ready'])
                if not ready or not daily['ready']:assert info['quality'] is None and info['lower_quality'] is None;continue
                x=panel.loc[at,cols(k)].to_numpy(float);point=reference_quality(daily['models'][k],x)
                draws=np.array([reference_quality(b[k],x) for b in daily['bootstrap']]);radius=max(0.,sorted(draws-point)[179])
                near(info['quality'],point);close(info['bootstrap_qualities'],draws);near(info['upward_radius'],radius);near(info['lower_quality'],point-radius)
            for name,es in predictions.items():
                e=es[i];assert e==saved['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                check_event(name,e,record,daily,saved['estimates']);actions+=1
            if (i+1)%200==0:print('verified selection through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    start,end=p['development']['evaluation_start'],p['development']['evaluation_end'];months=p['development']['months']
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected
    assert result['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date']))};books=[]
    for name,stats in result['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name]
        books.append(book_check(run,name,market,features,labels(root/('stress_labels.json' if name=='double_cost' else 'baseline_labels.json')),es,2 if name=='double_cost' else 1,stats,months))
    for suffix in ['equity','trades','decisions','orders']:
        pd.testing.assert_frame_equal(pd.read_csv(run/('source_v35_'+suffix+'.csv')),pd.read_csv(root/('full_band_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    receipt={'passed':True,'phase':'development','clock_ticks':len(features),'daily_fits':features.date.nunique(),
        'selected_feature_equations':feature_checks,'independent_logistic_equations':checks,'action_checks':actions,
        'capital_books':books,'source_v35_exact':True,'source_verified_hashes':len(want),'results_sha256':sha(run/'results.json'),
        'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
