"""Independent loss optima, calendar resamples, issued actions and capital books."""
import argparse,json
from collections import Counter
from threadpoolctl import threadpool_limits
from adaptive_correction import core,np,pd,Path,SOURCE,read,dump,sha,CLOCKS,tag,restrict_market,labels
from audit_shared import reference_pool
from audit_hierarchical import solve,model_check,predict
from verify_reliability import close
from verify_continuation import near
from verify_financial import check_action
from audit_books import book_check
from run_planning import json_ready

canonical=lambda x:json.loads(json.dumps(json_ready(x),default=str))


def loss_audit(saved,raw,cal,truth,clocks,groups,w,kind):
    """Verify KKT conditions directly; no training blend function is called."""
    to_cd=lambda a:np.array([[(r[0]+r[1])/2,(r[0]-r[1])/2] for r in a])
    raw,cal,truth=[to_cd(x) for x in [raw,cal,truth]];want=np.zeros((4,2));heads=[]
    for c in range(4) if kind=='clock' else [-1]:
        ids=[i for i in range(len(w)) if w[i]>0 and (c<0 or clocks[i]==c)]
        v=np.array([w[i] for i in ids]);v/=sum(v)
        bydate={}
        for val,i in zip(v,ids):bydate[groups[i]]=bydate.get(groups[i],0.)+val
        eff=1/sum(x*x for x in bydate.values())
        for k in range(2):
            d=cal[ids,k]-raw[ids,k];r=truth[ids,k]-raw[ids,k]
            A=sum(v*d*d);B=sum(v*d*r);lam=2*A/eff
            head=saved['heads'][len(heads)];alpha=head['alpha']
            for name,val in [('curvature',A),('linear',B),('effective_dates',eff),('penalty',lam)]:near(head[name],val)
            assert head['clock']==c and head['component']==k and 0<=alpha<=1
            grad=2*((A+lam)*alpha-B-lam/2)
            if A<=1e-16:near(alpha,.5)
            elif alpha<=1e-12:assert grad>=-1e-12
            elif alpha>=1-1e-12:assert grad<=1e-12
            else:assert abs(grad)<1e-11
            obj=lambda a:float(sum(v*(r-a*d)**2)+lam*(a-.5)**2)
            assert obj(alpha)<=min(obj(0),obj(1),obj(.5))+1e-12
            want[c if c>=0 else slice(None),k]=alpha;heads.append(head)
    assert len(heads)==len(saved['heads']);close(saved['alpha'],want)
    return want,len(heads)


def daily_audit(panel,market,records,at,source_day,saved):
    cutoff=pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
    history={t:r['raws']['full'] if r['estimates']['full']['ready'] else None for t,r in records.items() if t<cutoff}
    parents,fb=reference_pool(panel,market,history,at);assert canonical(parents)==saved['parents'] and canonical(fb)==saved['feedback']
    ids=[t for p in parents for t in p['members']];assert list(pd.to_datetime(saved['members'],utc=True))==ids
    assert pd.Timestamp(saved['cutoff'])==cutoff and pd.Timestamp(saved['date'])==panel.loc[at,'date']
    ready=len(parents)>=60 and source_day['ready'];assert ready==saved['ready']
    if not ready:assert saved['models']=={} and saved['bootstrap']==[];return {},0
    n=len(parents);dw=np.array([2**((i-n+1)/63) for i in range(n)]);dw*=n/dw.sum()
    w=np.array([dw[j]/len(p['members']) for j,p in enumerate(parents) for _ in p['members']]);groups=np.array([j for j,p in enumerate(parents) for _ in p['members']])
    close(saved['row_weights'],w);assert saved['row_groups']==groups.tolist()
    raw=np.array([records[t]['raws']['full'] for t in ids]);cal=np.array([records[t]['estimates']['full']['point'] for t in ids])
    truth=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in ids]);clocks=np.array([CLOCKS.index(panel.loc[t,'clock']) for t in ids])
    out={'models':{},'bootstrap':[]};checks=0
    for kind in ['clock','shared']:
        out['models'][kind],k=loss_audit(saved['models'][kind],raw,cal,truth,clocks,groups,w,kind);checks+=k
    bp=source_day['parents'];bid=list(pd.to_datetime(source_day['members'],utc=True));nb=len(bp)
    bw=np.array([2**((i-nb+1)/63) for i in range(nb)]);bw*=nb/bw.sum()
    brw=np.array([bw[j]/len(p['members']) for j,p in enumerate(bp) for _ in p['members']])
    X=np.array([[*records[t]['raws']['full'],CLOCKS.index(panel.loc[t,'clock'])] for t in bid]);Y=np.array([[panel.loc[t,'C']+panel.loc[t,'D'],panel.loc[t,'C']-panel.loc[t,'D']] for t in bid])-X[:,:2]
    m=solve(X,Y,brw,'partial',nb);model_check(saved['calibration_model'],m)
    model_check(source_day['variants']['full']['point_model'],m);out['calibration']=m;checks+=2
    union=sorted(set(pd.Timestamp(p['date']) for p in bp)|set(p['date'] for p in parents));assert list(pd.to_datetime(saved['union_dates']))==union
    seed=826+at.tz_convert(core.ET).date().toordinal();assert saved['seed']==seed
    starts=np.random.default_rng(seed).integers(0,len(union),size=(199,(len(union)+9)//10));assert starts.tolist()==saved['block_starts']
    base_bydate={};meta_rowdates=list(panel.loc[ids,'date'])
    for i,d in enumerate(panel.loc[bid,'date']):base_bydate.setdefault(d,[]).append(i)
    for b,draw in enumerate(starts):
        sampled=[union[(int(s)+j)%len(union)] for s in draw for j in range(10)][:len(union)]
        bi=np.array([i for d in sampled for i in base_bydate.get(d,[])],int)
        # Explicit repeated rows verify the calibration weighted-count shortcut.
        m=solve(X[bi],Y[bi],brw[bi],'partial',nb);model_check(saved['bootstrap'][b]['calibration'],m)
        counts=Counter(sampled);mw=np.array([v*counts[d] for v,d in zip(w,meta_rowdates)])
        item={'calibration':m};checks+=2
        for kind in ['clock','shared']:
            item[kind],k=loss_audit(saved['bootstrap'][b][kind],raw,cal,truth,clocks,groups,mw,kind);checks+=k
        out['bootstrap'].append(item)
    return out,checks


def mix(raw,cal,alpha):
    common=(raw[0]+raw[1])/2+alpha[0]*((cal[0]+cal[1]-raw[0]-raw[1])/2)
    direction=(raw[0]-raw[1])/2+alpha[1]*((cal[0]-cal[1]-raw[0]+raw[1])/2)
    return np.array([common+direction,common-direction])


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    root=Path(manifest['verified_source']);delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v35/evidence/delivery_manifest.json')
    want={str(Path(folder)/f):h for field,folder in [('cache_outputs_sha256',''),('daily_records_sha256','daily'),('training_records_sha256','training')] for f,h in delivery[field].items()}
    assert want==manifest['source_files_sha256']
    for f,h in want.items():assert sha(root/f)==h
    oldv=read(root/'verification.json');assert oldv['passed'] and oldv['results_sha256']==sha(root/'results.json')
    market=core.load_market();assert market.manifest==manifest['market']['full_market']
    market=restrict_market(market,pd.Timestamp(manifest['market']['observation_cutoff_exclusive']))
    panel=pd.read_parquet(root/'financial_panel.parquet');features=pd.read_parquet(root/'features.parquet')
    records={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))};predictions=read(run/'predictions.json')
    previous=None;daily=None;models={};checks=actions=readyticks=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            if row.date!=previous:
                daily=read(run/'daily'/(str(row.date.date())+'.json'));source_day=read(root/'daily'/(str(row.date.date())+'.json'))
                models,k=daily_audit(panel,market,records,at,source_day,daily);checks+=k;previous=row.date
            saved=read(run/'training'/(tag(at)+'.json'));record=records[at]
            assert saved['source_record_sha256']==sha(root/'training'/(tag(at)+'.json')) and pd.Timestamp(saved['at'])==at and saved['clock']==row.clock
            assert pd.Timestamp(saved['date'])==row.date and pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
            ready=bool(record['estimates']['full']['ready'] and panel.loc[at,'feature_ready'])
            raw=record['raws']['full'];cal=record['estimates']['full']['point'];c=CLOCKS.index(row.clock)
            if ready:readyticks+=1
            for kind,info in saved['estimates'].items():
                assert info['ready']==ready
                if not ready:assert info['point'] is None and info['lower'] is None;continue
                assert info['source_fallback']==(not daily['ready'])
                if not daily['ready']:
                    old=record['estimates']['full'];assert info['alpha']==[1.,1.]
                    for k in ['point','lower','radius','bootstrap_predictions']:assert info[k]==old[k]
                    assert info['source_radius']==info['joint_radius']==old['radius']
                    continue
                alpha=models['models'][kind][c] if kind in ['clock','shared'] else np.full(2,.5 if kind=='equal' else 0.)
                close(info['alpha'],alpha);point=mix(raw,cal,alpha);draws=[]
                for b in models['bootstrap']:
                    z=np.array(raw)+predict(b['calibration'],[*raw,c])[0]
                    draws.append(mix(raw,z,b[kind][c] if kind in ['clock','shared'] else alpha))
                draws=np.array(draws);radius=max(0.,sorted(np.max(draws-point,axis=1))[179]);floor=record['estimates']['full']['radius']
                close(info['point'],point);close(info['bootstrap_predictions'],draws);near(info['joint_radius'],radius);near(info['source_radius'],floor)
                near(info['radius'],max(radius,floor));close(info['lower'],point-max(radius,floor))
            for name,es in predictions.items():
                e=es[i];assert e==saved['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                if name=='source_v35':assert e==record['events']['full_band']
                elif name=='cash':assert e['action']=='CASH' and e['allocation']==0
                else:
                    kind={'shared_band':'shared','equal_band':'equal','raw_band':'raw'}.get(name,'clock')
                    check_action(saved['estimates'][kind],daily['feedback']['expired_unresolved_prefix'],name!='adaptive_point',e)
                actions+=1
            if (i+1)%200==0:print('verified correction through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    start,end=p['development']['evaluation_start'],p['development']['evaluation_end'];months=p['development']['months']
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected
    assert result['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date']))}
    books=[]
    for name,stats in result['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name]
        books.append(book_check(run,name,market,features,labels(root/('stress_labels.json' if name=='double_cost' else 'baseline_labels.json')),es,2 if name=='double_cost' else 1,stats,months))
    for suffix in ['equity','trades','decisions','orders']:
        pd.testing.assert_frame_equal(pd.read_csv(run/('source_v35_'+suffix+'.csv')),pd.read_csv(root/('full_band_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    receipt={'passed':True,'phase':'development','clock_ticks':len(features),'daily_fits':features.date.nunique(),'ready_ticks':readyticks,
        'independent_equations':checks,'action_checks':actions,'capital_books':books,'source_v35_exact':True,
        'source_verified_hashes':len(want),'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
