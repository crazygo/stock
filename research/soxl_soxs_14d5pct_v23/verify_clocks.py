"""Audit clock separation, matured membership, equations, sequential actions and capital."""
import argparse
import gzip
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from clock_training import *
from run_clocks import sha,tag
from verify_continuation import read,labels,key,near,raw_exit
from verify_early import risk_equations
from verify_feedback_v2 import terminal_audit,verify_asof_replays
from audit_books import book_check
from audit_models import model_equations
from audit_inputs import input_check


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    source=Path(manifest['source_run'])
    for f,h in manifest['source_files_sha256'].items():assert sha(source/f)==h
    original=core.load_market();assert original.manifest==manifest['full_market']
    cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive']);market=restrict_market(original,cutoff)
    leaders=load_panel(original.calendar);assert leaders[2]==manifest['panel']
    features=pd.read_parquet(run/'features.parquet');panel=pd.read_parquet(run/'panel.parquet')
    old=pd.read_parquet(source/'features.parquet');pd.testing.assert_frame_equal(features,old[old.index<cutoff])
    input_audit=input_check(run,market,leaders[0],features,panel,cutoff)
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json')
    terminals={};releases={};receipts={pd.Timestamp(r['at']):r for r in read(run/'terminal_receipts.json')}
    for at,row in panel.iterrows():
        pair=[base[key(at,s)] for s in core.ETFS];l,s=np.log1p([b.get('net_return',np.nan) for b in pair])
        np.testing.assert_allclose(row[['C','D','M']].to_numpy(float),[(l+s)/2,(l-s)/2,abs(l-s)/2],equal_nan=True)
        if all(b['complete_window'] and b.get('label_available_at') is not None and np.isfinite(b.get('net_return',np.nan)) for b in pair):
            t=max(b['label_available_at'] for b in pair);assert row.release==t
            if row.feature_ready:releases[at]=t
        else:assert pd.isna(row.release)
        if not row.feature_ready:continue
        ts=[terminal_audit(market,b) for b in pair]
        if all(t is not None for t in ts):
            terminals[at]=max(ts);assert row.terminal_release==max(ts)==pd.Timestamp(receipts[at]['terminal_release'])
        else:assert pd.isna(row.terminal_release) and receipts[at]['terminal_release'] is None
    anchors=read(run/'anchors.json');vectors={};returns={};riskreleases={}
    for c in CLOCKS:
        for v in ['base','context']:
            coords={};outcomes={};known={};b=pd.read_parquet(run/('bank_'+c[:5].replace(':','')+'_'+v+'.parquet'))
            for e in anchors[c][v]:
                if not e['ready']:continue
                at=pd.Timestamp(e['at']);r=panel.loc[at]
                if at in releases:known[at]=releases[at]
                for s,sign in [('SOXL',1),('SOXS',-1)]:
                    vals=np.array([e['adjusted'][s],sign*e['adjusted']['D'],sign*r.relative_5,sign*r.breadth_5,
                                   sign*r.volume_direction_diffusion,r.log_sector_volatility])
                    coords[at,s]=vals;outcomes[at,s]=base[key(at,s)].get('net_return',np.nan)
                    np.testing.assert_allclose(b.loc[(at,s),['score','advantage','relative_5','breadth_5',
                        'volume_direction_diffusion','log_sector_volatility']].to_numpy(float),vals,atol=1e-12)
                    np.testing.assert_allclose(b.loc[(at,s),'net_return'],outcomes[at,s],atol=1e-12,equal_nan=True)
            assert len(b)==len(coords);vectors[c,v]=coords;returns[c,v]=outcomes;riskreleases[c,v]=known
    predictions=read(run/'predictions.json');history={(c,v,m):[] for c in CLOCKS for v in ['base','context'] for m in ['early','full']}
    membership=equations=calibrations=entropy=0;old21=core.ROOT/p['premarket_control'][reg['phase']]
    old_events={pd.Timestamp(e['at']):e for e in read(old21/'predictions.json')['early_base_cal']};control=0
    for i,(at,row) in enumerate(features.iterrows()):
        c=row.clock;sp=panel.loc[features[features.clock==c].index];dc=day_cutoff(at)
        rec=read(run/'training'/(tag(at)+'.json'));assert pd.Timestamp(rec['cutoff'])==dc and rec['clock']==c
        past=list(sp[sp.feature_ready&(sp.date<row.date)&(sp.index<dc)].index);early=[]
        for t in past:
            if t not in terminals or terminals[t]>=dc:break
            early.append(t)
        full=[t for t in past if t in releases and releases[t]<dc]
        ids={'early':early,'full':full};expired=len(early)<len(past) and sp.loc[past[len(early)],'date']+pd.Timedelta(days=14)<=row.date
        with gzip.open(run/'models'/(tag(at)+'.pkl.gz'),'rb') as f:models=pickle.load(f)
        for v in ['base','context']:
            r=rec['variants'][v];fb=r['feedback']
            assert fb['prior_dates']==len(past) and fb['selected_early_dates']==len(early) and fb['selected_full_dates']==len(full)
            assert fb['expired_unresolved_prefix']==bool(expired)
            assert fb['known_but_waiting']==sum(t in terminals and terminals[t]<dc for t in past[len(early)+1:])
            for m in ['early','full']:
                info=r['fits'][m];assert list(pd.to_datetime(info['members'],utc=True))==ids[m]
                assert list(pd.to_datetime(info['dates']))==list(sp.loc[ids[m],'date']) and info['columns']==columns(c,v)
                raw,n=model_equations(sp,ids[m],at,models[v][m],info,columns(c,v));equations+=n
                saved=r['forecasts'][m];cal=saved['calibration'];event=saved['event']
                assert set(raw)==set(event['raw'])
                for k in raw:near(raw[k],event['raw'][k])
                eligible=set(ids[m]);old=[h for h in history[c,v,m] if h['at'] in eligible and h['raw']][-120:]
                assert list(pd.to_datetime(cal['members'],utc=True))==[h['at'] for h in old]
                assert list(pd.to_datetime(cal['dates']))==list(sp.loc[[h['at'] for h in old],'date'])
                assert cal['ready']==(len(old)>=60);values={}
                for target in ['C','D']:
                    if not old:continue
                    x=np.array([h['raw'][target] for h in old]);y=sp.loc[[h['at'] for h in old],target].to_numpy(float)
                    center=x.mean();scale=x.std() if x.std()>=1e-12 else 1.;z=(x-center)/scale;intercept=y.mean();slope=z@(y-intercept)/(z@z+20.)
                    for k,val in {'center':center,'scale':scale,'intercept':intercept,'coefficient':slope,'score_min':x.min(),'score_max':x.max()}.items():near(cal['maps'][target][k],val)
                    if raw:values[target]=intercept+slope*(raw[target]-center)/scale
                ready=bool(raw) and cal['ready'] and not(m=='early' and expired)
                for method,ev in [('paired',event)]+([('raw',r['forecasts']['raw']['event'])] if m=='early' else []):
                    assert ev['ready']==ready
                    if ready:
                        a,d=[raw[k] if method=='raw' else values[k] for k in ['C','D']];top='SOXL' if d>=0 else 'SOXS'
                        assert ev['top_symbol']==top and ev['action']==(top if a+abs(d)>0 else 'CASH')
                        for k,val in {'C':a,'D':d,'SOXL':a+d,'SOXS':a-d}.items():near(ev['adjusted'][k],val)
                    else:assert ev['action']=='CASH' and ev['adjusted'] is None
                history[c,v,m].append({'at':at,'raw':event['raw']});calibrations+=1
                if m=='full':
                    anchor=next(e for e in anchors[c][v] if pd.Timestamp(e['at'])==at)
                    for k in event:assert anchor[k]==event[k]
            for m in ['early','full','raw']:
                entropy+=risk_equations(r['capital'][m],r['forecasts'][m]['event'],at,dc,vectors[c,v],returns[c,v],riskreleases[c,v])
            membership+=1
        # Route mapping is checked independently of production route_events.
        for n in ROUTES:
            v='base' if n=='base_sequential' else 'context';m='raw' if n=='context_raw' else 'full' if n=='context_full_window' else 'early'
            active=n!='cash' and not(n=='context_after_open' and c==CLOCKS[0]) and not(n=='premarket_only' and c!=CLOCKS[0])
            if n in ['context_1005','context_1035','context_1305']:active=c=={'context_1005':CLOCKS[1],'context_1035':CLOCKS[2],'context_1305':CLOCKS[3]}[n]
            e=predictions[n][i];assert pd.Timestamp(e['at'])==at
            if n=='context_no_risk':
                f=rec['variants']['context']['forecasts']['early']['event'];take=f['ready'] and f['action']!='CASH'
                assert e['action']==(f['top_symbol'] if take else 'CASH') and e['allocation']==float(take)
            elif active:assert e==rec['variants'][v]['capital'][m]['event']
            else:assert e['action']=='CASH' and e['allocation']==0.
        if c==CLOCKS[0]:
            assert predictions['premarket_only'][i]==old_events[at];control+=1
        if (i+1)%200==0:print('verified through',at,flush=True)
    exits=0
    for lab,multiple in [(base,1),(stress,2)]:
        for label in lab.values():
            if 'entry_fill_economic' in label:raw_exit(market,label,None,multiple);exits+=1
    asof=verify_asof_replays(market,base,terminals)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];months=settings.get('months',settings.get('regression_months'));books=[]
    for n,stats in result['routes'].items():
        wanted=[e for e in predictions[n] if settings['evaluation_start']<=e['date'][:10]<=settings['evaluation_end']]
        assert frozen['routes'][n]==wanted
        books.append(book_check(run,n,market,features,stress if n=='double_cost' else base,wanted,2 if n=='double_cost' else 1,stats,months))
    for suffix in ['equity','trades']:
        a=pd.read_csv(run/('premarket_only_'+suffix+'.csv'));b=pd.read_csv(old21/('early_base_cal_'+suffix+'.csv'))
        pd.testing.assert_frame_equal(a[b.columns],b,atol=1e-12,rtol=1e-12)
    if reg['phase']=='cases':
        dev=Path(reg['authorization']['run']);oldp=read(dev/'predictions.json')
        for n in ROUTES:assert [e for e in predictions[n] if e['date'][:10]<'2026-05-01']==oldp[n]
    result={'passed':True,'phase':reg['phase'],'input_audit':input_audit,'clock_variant_membership':membership,
        'independent_model_equations':equations,'own_clock_calibrations':calibrations,'entropy_roots':entropy,
        'raw_exit_checks':exits,'joint_terminal_receipts':len(terminals),'truncated_asof_outcome_replays':asof,
        'premarket_predictions_exact_v21':control,'premarket_ledger_exact_v21':True,'capital_books':books,
        'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    core.dump(run/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);verify(parser.parse_args().run)
