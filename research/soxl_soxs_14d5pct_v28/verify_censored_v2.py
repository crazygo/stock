"""Independent risk-set sums, prefix receipts, decisions and capital books."""
import argparse, math, gzip, pickle
from threadpoolctl import threadpool_limits
from censored_learning import *
from verify_continuation import key,near
from audit_books import book_check
from audit_models import model_equations
from clock_reliability import columns,training_dates
from risk import first_exit


def independent_paths(market,features,base):
    paths={};checks=0
    for at in features.index:
        day=pd.Timestamp(at.tz_convert(core.ET).date())
        days=market.sessions[(market.sessions>=day)&(market.sessions<=day+pd.Timedelta(days=13))]
        grid=[]
        for d in days:
            session=market.calendar[d];opening=pd.Timestamp(session['open_at']);closing=pd.Timestamp(session['close_at'])
            grid.extend(t for t in pd.date_range(opening+pd.Timedelta(minutes=5),closing,freq='5min') if t-pd.Timedelta(minutes=5)>at+pd.Timedelta(seconds=30))
        ends=pd.DatetimeIndex(grid)
        for symbol in core.ETFS:
            f=market.bars[symbol].reindex(ends);valid=(f.start_at==ends-pd.Timedelta(minutes=5))&(f.end_at==ends)&f.session_type.eq('regular')
            valid &= f.available_at.notna()&(f.available_at>=f.end_at)
            for col in ['open','high','low','close','share_factor']:valid &= np.isfinite(f[col])&(f[col]>0)
            # Release includes every preceding required bar. A hole remains a hole.
            releases=[];latest=pd.Timestamp.min.tz_localize('UTC');broken=False
            for i,(ok,av) in enumerate(zip(valid,f.available_at)):
                broken |= not ok
                latest=pd.Timestamp.max.tz_localize('UTC') if broken else max(latest,av,ends[i])
                releases.append(latest.value)
            label=base[key(at,symbol)];event_index=None;event=0;log=None
            if valid.iloc[0]:
                entry=float(f.open.iloc[0])*(1+.0005)*float(f.share_factor.iloc[0])
                target=math.ceil(float(f.open.iloc[0])*1.0005*1.05*100-1e-10)/100*float(f.share_factor.iloc[0])
                holes=np.flatnonzero(~valid.to_numpy());prefix=int(holes[0]) if len(holes) else len(f)
                event_data=first_exit(f.iloc[:prefix],entry,target,.06,None,.0005,.0005)
                if event_data is not None:
                    event_index=event_data['offset'];event=1 if event_data['reason']=='target' else 2;fill=event_data['fill']
                elif valid.all():event_index=len(f)-1;event=3;fill=float(f.econ_close.iloc[-1])*.9995
                if event:
                    log=math.log(fill*.9995/(entry*1.0005))
                    if label['exit_at'] is not None:
                        assert ends[event_index]==label['exit_at'];near(fill,label['exit_fill_economic'])
                        near(log,math.log((label['exit_fill_economic']*.9995)/(label['entry_fill_economic']*1.0005)))
                        checks+=1
            paths[(at,symbol)]={'ends':ends,'release':np.array(releases,dtype=np.int64),'event_index':event_index,'event':event,'log':log,'expiry':days[-1]}
    return paths,checks


def reference_observation(path,cutoff):
    n=int(np.searchsorted(path['release'],cutoff.value,side='left'));i=path['event_index']
    known=i is not None and i<n;used=i+1 if known else n
    missing=not known and n<len(path['ends']) and path['ends'][n]<cutoff
    return {'duration':used/len(path['ends']),'event':path['event'] if known else 0,'price_log_return':path['log'] if known else None,
        'observed_bars':used,'expected_bars':len(path['ends']),'last_end':str(path['ends'][used-1]) if used else None,
        'release':str(pd.Timestamp(path['release'][used-1],tz='UTC')) if used else None,
        'censor_reason':None if known else ('missing_or_late_prefix' if missing else 'administrative_cutoff'),
        'expired_missing':bool(missing and path['expiry']<pd.Timestamp(cutoff.tz_convert(core.ET).date()))}


def independent_aj(obs,W):
    n=len(W);survival=np.ones(n);cif=np.zeros((n,3));values=np.zeros(n)
    for time in sorted({o['duration'] for o in obs if o['event']}):
        risk=W[:,[i for i,o in enumerate(obs) if o['duration']>=time]].sum(axis=1)
        dying=[i for i,o in enumerate(obs) if o['duration']==time and o['event']]
        total=np.zeros(n)
        for cause in (1,2,3):
            indexes=[i for i in dying if obs[i]['event']==cause]
            if indexes:
                delta=np.divide(W[:,indexes].sum(axis=1),risk,out=np.zeros(n),where=risk>0)
                cif[:,cause-1]+=survival*delta;total+=delta
                payoff=sum(W[:,i]*obs[i]['price_log_return'] for i in indexes)
                values+=survival*np.divide(payoff,risk,out=np.zeros(n),where=risk>0)
        survival*=np.maximum(0,1-total)
    return cif,survival,values


def check_estimate(history,paths,at,raw,terminal,out):
    cutoff=day_cutoff(at);ids=[t for t,v in history.items() if t<cutoff and v is not None][-252:]
    assert out['members']==[str(t) for t in ids] and out['terminal_only']==terminal and pd.Timestamp(out['cutoff'])==cutoff
    if raw is None or len(ids)<60:assert not out['ready'] and out['observations']=={};return 0
    X=np.array([history[t] for t in ids]);r=np.exp2(-np.arange(len(ids)-1,-1,-1)/63.)
    mu=(X*r[:,None]).sum(axis=0)/r.sum();scale=np.maximum(np.sqrt(((X-mu)**2*r[:,None]).sum(axis=0)/r.sum()),.001)
    d=np.square((X-np.array(raw))/scale).sum(axis=1);w=r*np.exp(-.5*np.minimum(d,1400))
    rng=np.random.default_rng(828+at.tz_convert(core.ET).date().toordinal());count=[]
    for _ in range(199):
        starts=rng.integers(0,len(ids),size=math.ceil(len(ids)/10));counts=np.zeros(len(ids))
        samples=[(int(s)+k)%len(ids) for s in starts for k in range(10)][:len(ids)]
        for i in samples:counts[i]+=1
        count.append(counts)
    W=np.vstack([w,np.array(count)*w]);pred=[];expired=False;checks=0
    for symbol in core.ETFS:
        obs=[reference_observation(paths[(t,symbol)],cutoff) for t in ids]
        saved=out['observations'][symbol];assert len(obs)==len(saved)
        for want,have in zip(obs,saved):
            for k,v in want.items():
                if k=='price_log_return' and v is not None:near(v,have[k])
                else:assert v==have[k],(at,symbol,k,v,have[k])
        expired |= any(o['expired_missing'] for o in obs)
        mask=np.array([bool(o['event']) if terminal else True for o in obs]);weights=W*mask
        cif,tail,val=independent_aj(obs,weights);risk=out['risks'][symbol]
        np.testing.assert_allclose(cif[0],risk['cif'],atol=1e-12);near(tail[0],risk['unidentified_mass'])
        effective=weights[0].sum()**2/max(np.square(weights[0]).sum(),1e-300);near(effective,risk['effective_dates'])
        support=(tail<=1e-10)&(weights.sum(axis=1)>1e-20)
        assert risk['bootstrap_supported']==int(support[1:].sum())
        assert risk['known_terminals']==sum(o['event']>0 for o in obs) and risk['right_censored']==sum(o['event']==0 for o in obs)
        checks+=len(val)
        if effective<30 or not support.all():assert not out['ready'] and out['reason']=='insufficient_local_or_horizon_support';return checks
        pred.append(val)
    val=np.array(pred).T;deviations=np.sort(np.max(val[1:]-val[0],axis=1));margin=max(0.,deviations[179])
    np.testing.assert_allclose(out['point'],val[0],atol=2e-12,rtol=2e-10)
    np.testing.assert_allclose(out['bootstrap_values'],val[1:],atol=2e-12,rtol=2e-10)
    near(out['margin'],margin);np.testing.assert_allclose(out['lower'],val[0]-margin,atol=2e-12)
    np.testing.assert_allclose(out['weights'],w,atol=1e-12);np.testing.assert_allclose(out['kernel_scale'],scale,atol=1e-12)
    assert out['ready']==(not expired);return checks


def verify(run):
    run=Path(run);amendment=read(Path(__file__).parent/'evidence/implementation_amendment.json')
    assert sha(Path(__file__))==amendment['corrected_sources_sha256'][str(Path(__file__).relative_to(core.ROOT))]
    assert sha(run/'predictions.json')==amendment['original_predictions_sha256'] or read(run/'registration.json')['phase']=='cases'
    reg=read(run/'registration.json');manifest=read(run/'input_manifest.json');p=reg['preregistration']['protocol']
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    assert reg['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')
    market=core.load_market();assert market.manifest==manifest['full_market']
    market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    features=pd.read_parquet(run/'features.parquet');panel=pd.read_parquet(run/'panel.parquet');base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json')
    source=Path(manifest['source_run'])
    for f,h in manifest['source_files_sha256'].items():assert sha(source/f)==h
    if reg['phase']=='development':
        pd.testing.assert_frame_equal(features,pd.read_parquet(source/'features.parquet'))
        old=pd.read_parquet(source/'panel.parquet');pd.testing.assert_frame_equal(panel[old.columns],old)
        from verify_clock_reliability_v2 import compare_reencoded_labels
        for file in ['baseline_labels.json','stress_labels.json']:compare_reencoded_labels(read(run/file),read(source/file))
    paths,exit_checks=independent_paths(market,features,base)
    predictions=read(run/'predictions.json');histories={c:{} for c in CLOCKS};equations=actions=0
    oldroot=core.ROOT/p['control_source'];delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v27/evidence/delivery_manifest.json')
    for file in ['predictions.json','score_sequential_equity.csv','score_sequential_trades.csv']:assert sha(oldroot/file)==delivery['cache_outputs_sha256'][file]
    oldpred={pd.Timestamp(e['at']):e for e in read(oldroot/'predictions.json')['score_sequential']}
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            rec=read(run/'training'/(tag(at)+'.json'));raw=rec['raw'];c=row.clock
            assert pd.Timestamp(rec['at'])==at and rec['clock']==c and pd.Timestamp(rec['cutoff'])==day_cutoff(at)
            if row.date<pd.Timestamp('2026-05-01'):
                src=source/'training'/(tag(at)+'.json');assert sha(src)==manifest['source_training_sha256'][src.name]==rec['origin']['sha256']
                want=read(src)['variants']['context']['forecasts']['early']['event']['raw']
                assert raw==([want[s] for s in core.ETFS] if want else None)
            else:
                same=panel[panel.clock==c];dates,_=training_dates(same,day_cutoff(at));info=rec['origin']['fit'];mp=run/'models'/(tag(at)+'.pkl.gz')
                model=pickle.load(gzip.open(mp,'rb')) if info['ready'] else None
                want,n=model_equations(same,dates['early'],at,model,info,columns(c,'context'));equations+=n
                if want:np.testing.assert_allclose(raw,[want[s] for s in core.ETFS],atol=1e-12)
                else:assert raw is None
            for k in ['censored','terminal']:equations+=check_estimate(histories[c],paths,at,raw,k=='terminal',rec['estimates'][k])
            for name,items in predictions.items():
                event=items[i];assert rec['events'][name]==event and pd.Timestamp(event['at'])==at
                if name=='source_v27' and at in oldpred:assert event==oldpred[at]
                elif name=='source_v27':
                    from clock_reliability import issue
                    reference=issue(panel,histories[c],at,np.array(raw) if raw else None)['band']
                    for field,v in reference.items():assert event[field]==v
                elif name=='cash' or (name=='premarket_only' and c!=CLOCKS[0]):assert event['action']=='CASH' and event['allocation']==0
                else:
                    info=rec['estimates']['terminal' if name=='terminal_only_band' else 'censored'];assert event['ready']==info['ready']
                    if info['ready']:
                        values=info['point' if name=='censored_point' else 'lower'];j=0 if values[0]>=values[1] else 1
                        action=core.ETFS[j] if values[j]>1e-12 else 'CASH';assert event['values']==values
                    else:action='CASH';assert event['values'] is None
                    assert event['action']==action and event['allocation']==float(action!='CASH')
                actions+=1
            histories[c][at]=raw
            if (i+1)%200==0:print('independent AJ checked through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'));books=[]
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected and result['coverage']['complete']
    for name,s in result['routes'].items():
        wanted=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert frozen['routes'][name]==wanted
        books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,wanted,2 if name=='double_cost' else 1,s,months))
    if reg['phase']=='development':
        for suffix in ['equity','trades']:
            old=pd.read_csv(oldroot/('score_sequential_'+suffix+'.csv'));new=pd.read_csv(run/('source_v27_'+suffix+'.csv'))
            pd.testing.assert_frame_equal(new[old.columns],old,atol=1e-12,rtol=1e-12)
    else:
        dev=Path(reg['authorization']['run']);prior=read(dev/'predictions.json')
        for name,items in prior.items():assert predictions[name][:len(items)]==items
        for f in (dev/'training').glob('*.json'):assert read(f)==read(run/'training'/f.name)
    receipt={'passed':True,'clock_ticks':len(features),'independent_price_terminal_checks':exit_checks,'independent_estimator_checks':equations,
        'action_checks':actions,'capital_books':books,'results_sha256':sha(run/'results.json'),'source_v27_exact':reg['phase']=='development',
        'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
