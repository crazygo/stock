"""Independent qualification membership, support bounds, fits and capital books."""
import argparse
from threadpoolctl import threadpool_limits
from eligible_risk import core,np,pd,Path,SOURCE,read,dump,sha,CLOCKS,tag,restrict_market,labels
from verify_selected import verify_panel,fit_audit,reference_quality,canonical,SCORE_COLUMNS,STATE_COLUMNS
from audit_shared import reference_pool
from verify_reliability import close
from verify_continuation import near
from audit_books import book_check

COLS={'eligible_context':SCORE_COLUMNS+STATE_COLUMNS,'eligible_scores':SCORE_COLUMNS,'all_context':SCORE_COLUMNS+STATE_COLUMNS}


def daily_audit(panel,market,records,at,saved):
    cutoff=pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
    history={t:r['raws']['full'] if r['estimates']['full']['ready'] else None for t,r in records.items() if t<cutoff}
    parents,fb=reference_pool(panel,market,history,at);assert saved['parents']==canonical(parents) and saved['feedback']==canonical(fb)
    assert pd.Timestamp(saved['date'])==panel.loc[at,'date'] and pd.Timestamp(saved['cutoff'])==cutoff
    allrows=[{'parent_index':j,'date':p['date'],'members':p['members']} for j,p in enumerate(parents)];eligible=[]
    for j,p in enumerate(parents):
        ids=[]
        for t in p['members']:
            e=records[t]['events']['full_band']
            if e['action'] in core.ETFS:
                assert e['ready'] and e['allocation']==1.;ids.append(t)
        if ids:eligible.append({'parent_index':j,'date':p['date'],'members':ids})
    assert saved['subsets']==canonical({'all':allrows,'eligible':eligible})
    n=len(parents);ready=n>=60 and len(eligible)>=30;assert saved['ready']==ready
    if not ready:assert saved['models']=={} and saved['bootstrap']==[] and saved['inputs']=={};return 0
    matrices={};checks=0
    for kind,cols in COLS.items():
        pp=allrows if kind=='all_context' else eligible;ids=[t for p in pp for t in p['members']];nn=len(pp)
        age=np.array([n-1-p['parent_index'] for p in pp]);dw=2.**(-age/63);dw*=nn/dw.sum()
        w=np.array([dw[j]/len(p['members']) for j,p in enumerate(pp) for _ in p['members']]);groups=np.array([p['parent_index'] for p in pp for _ in p['members']])
        info=saved['inputs'][kind];assert info['columns']==cols and info['distinct_dates']==nn and list(pd.to_datetime(info['members'],utc=True))==ids
        close(info['weights'],w);assert info['row_parent_groups']==groups.tolist()
        X=panel.loc[ids,cols].to_numpy(float);r=panel.loc[ids,'selected_y'].to_numpy(float)
        close(info['minimum'],np.min(X,axis=0));close(info['maximum'],np.max(X,axis=0))
        checks+=fit_audit(saved['models'][kind],X,r,w,nn);matrices[kind]=(X,r,w,groups,nn)
    seed=826+at.tz_convert(core.ET).date().toordinal();assert seed==saved['seed']
    starts=np.random.default_rng(seed).integers(0,n,size=(199,(n+9)//10));assert saved['block_starts']==starts.tolist()
    for b,draw in enumerate(starts):
        dates=[(int(s)+j)%n for s in draw for j in range(10)][:n]
        for kind,(X,r,w,g,nn) in matrices.items():
            ids=np.array([i for day in dates for i in np.flatnonzero(g==day)],int)
            model=saved['bootstrap'][b][kind]
            if not len(ids):assert model is None
            else:checks+=fit_audit(model,X[ids],r[ids],w[ids],nn)
    return checks


def check_event(name,e,record,daily,estimates,old_v37):
    old=record['events']['full_band']
    if name=='source_v35':assert e==old;return
    if name=='source_v37':assert e==old_v37;return
    if name=='cash':assert e['action']=='CASH' and e['allocation']==0;return
    kind='all_context' if name in ['all_supported','all_gate'] else 'eligible_scores' if name=='eligible_scores' else 'eligible_context'
    supported=name in ['eligible_supported','all_supported','eligible_scores','eligible_band','double_cost'];band=name=='eligible_band'
    v=estimates[kind];ready=v['ready'] and not daily['feedback']['expired_unresolved_prefix'];assert e['ready']==ready
    take=False;q=None;reason='unready'
    if ready:
        if old['action']=='CASH':reason='source_rejected'
        elif v['source_fallback']:take=True;reason='source_warmup'
        elif supported and v['outside_columns']:take=True;reason='source_outside_support'
        elif band and not v['bootstrap_complete']:reason='bootstrap_incomplete'
        else:
            q=v['lower_quality'] if band else v['quality'];take=q>.5+1e-12
            reason='positive_gain_loss_balance' if take else 'risk_rejected'
    assert e['action']==(old['action'] if take else 'CASH') and e['allocation']==float(take)
    assert e['quality_used']==q and e['reason']==reason


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    root=Path(manifest['verified_source']);delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v35/evidence/delivery_manifest.json')
    want={str(Path(folder)/f):h for field,folder in [('cache_outputs_sha256',''),('daily_records_sha256','daily'),('training_records_sha256','training')] for f,h in delivery[field].items()}
    assert manifest['source_files_sha256']==want
    for f,h in want.items():assert sha(root/f)==h
    oldv=read(root/'verification.json');assert oldv['passed'] and oldv['results_sha256']==sha(root/'results.json')
    control=Path(manifest['control_source']);cd=read(core.ROOT/'research/soxl_soxs_14d5pct_v37/evidence/delivery_manifest.json')
    for f,h in manifest['control_sha256'].items():assert sha(control/f)==h==cd['cache_outputs_sha256'][f]
    control_events={pd.Timestamp(e['at']):e for e in read(control/'predictions.json')['context_gate']}
    market=core.load_market();assert market.manifest==manifest['market']['full_market'];market=restrict_market(market,pd.Timestamp(manifest['market']['observation_cutoff_exclusive']))
    original=pd.read_parquet(root/'financial_panel.parquet');panel=pd.read_parquet(run/'selected_panel.parquet');features=pd.read_parquet(root/'features.parquet')
    records={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))}
    feature_checks=verify_panel(original,panel,records);predictions=read(run/'predictions.json');previous=None;daily=None;checks=actions=readydays=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            if row.date!=previous:
                daily=read(run/'daily'/(str(row.date.date())+'.json'));checks+=daily_audit(panel,market,records,at,daily);readydays+=int(daily['ready']);previous=row.date
            saved=read(run/'training'/(tag(at)+'.json'));record=records[at]
            assert saved['source_record_sha256']==sha(root/'training'/(tag(at)+'.json')) and pd.Timestamp(saved['at'])==at and saved['clock']==row.clock
            assert pd.Timestamp(saved['date'])==row.date and pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
            ready=bool(record['estimates']['full']['ready'] and panel.loc[at,'feature_ready'])
            for kind,v in saved['estimates'].items():
                assert v['ready']==ready and v['source_fallback']==(ready and not daily['ready'])
                if not ready or not daily['ready']:assert v['quality'] is None and v['lower_quality'] is None;continue
                x=panel.loc[at,COLS[kind]].to_numpy(float);info=daily['inputs'][kind]
                outside=[c for j,c in enumerate(COLS[kind]) if x[j]<info['minimum'][j]-1e-12 or x[j]>info['maximum'][j]+1e-12]
                assert v['outside_columns']==outside;point=reference_quality(daily['models'][kind],x);near(v['quality'],point)
                draws=[reference_quality(b[kind],x) if b[kind] is not None else None for b in daily['bootstrap']]
                assert v['bootstrap_complete']==all(z is not None for z in draws)
                for actual,wanted in zip(v['bootstrap_qualities'],draws):
                    if wanted is None:assert actual is None
                    else:near(actual,wanted)
                if v['bootstrap_complete']:
                    radius=max(0.,sorted(np.array(draws)-point)[179]);near(v['upward_radius'],radius);near(v['lower_quality'],point-radius)
                else:assert v['upward_radius'] is None and v['lower_quality'] is None
            for name,es in predictions.items():
                e=es[i];assert e==saved['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                check_event(name,e,record,daily,saved['estimates'],control_events[at]);actions+=1
            if (i+1)%200==0:print('verified eligibility through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    start,end=p['development']['evaluation_start'],p['development']['evaluation_end'];months=p['development']['months']
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected
    assert result['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date']))};books=[]
    for name,stats in result['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name]
        books.append(book_check(run,name,market,features,labels(root/('stress_labels.json' if name=='double_cost' else 'baseline_labels.json')),es,2 if name=='double_cost' else 1,stats,months))
    for name,source,prefix in [('source_v35',root,'full_band'),('source_v37',control,'context_gate')]:
        for suffix in ['equity','trades','decisions','orders']:
            pd.testing.assert_frame_equal(pd.read_csv(run/(name+'_'+suffix+'.csv')),pd.read_csv(source/(prefix+'_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    receipt={'passed':True,'phase':'development','clock_ticks':len(features),'daily_records':features.date.nunique(),'ready_fit_dates':readydays,
        'selected_feature_equations':feature_checks,'independent_logistic_equations':checks,'action_checks':actions,'capital_books':books,
        'v35_v37_controls_exact':True,'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
