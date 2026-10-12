"""Independent transition timing, Bellman ridge equations and cash books."""
import argparse,gzip,pickle
from pathlib import Path
from threadpoolctl import threadpool_limits
from planning import core,COLUMNS,restrict_market
from run_planning import sha,read_bank,dump
from verify_continuation import read,labels,key,near
from verify_feedback_v2 import terminal_audit
from audit_books import book_check
import numpy as np
import pandas as pd


def close(a,b):
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-8,equal_nan=True)


def decay(n,half):
    w=np.array([2.**((j+1-n)/half) for j in range(n)])
    return w*n/w.sum() if n else w


def transitions(market,panel,base,saved):
    expected={};by_date={r.date:t for t,r in panel.iterrows()};terminal_count=0
    for t,row in panel[panel.feature_ready].iterrows():
        pair=[base[key(t,s)] for s in core.ETFS];term=[terminal_audit(market,x) for x in pair]
        terminal=max(term) if all(x is not None for x in term) else None
        assert (pd.isna(row.terminal_release) and terminal is None) or row.terminal_release==terminal
        terminal_count+=terminal is not None
        upcoming=[d for d in market.sessions if d>row.date]
        dates=[upcoming[0] if upcoming else None];rewards=[0.]
        for label in pair:
            net=label.get('net_return');valid=net is not None and np.isfinite(net) and label['exit_at'] is not None
            rewards.append(float(np.log1p(net)) if valid else np.nan)
            if not valid:dates.append(None);continue
            exitday=pd.Timestamp(label['exit_at'].tz_convert(core.ET).date())
            settled=[d for d in market.sessions if d>exitday]
            if not settled:dates.append(None);continue
            deadline=max([settled[0]]+[d['pay'] for d in label['dividends']])
            candidates=[d for d in market.sessions if d>=deadline]
            dates.append(candidates[0] if candidates else None)
        nxt=[by_date.get(d) for d in dates]
        nxt=[n if n is not None and panel.loc[n,'feature_ready'] else None for n in nxt]
        durations=[sum(row.date<d<=end for d in market.sessions) if end is not None else None for end in dates]
        ready=max([terminal]+nxt) if terminal is not None and all(n is not None for n in nxt) and np.isfinite(rewards).all() else None
        expiry=pair[0].get('expiry_date',row.date+pd.Timedelta(days=13))
        bounds=[d for d in market.sessions if d>expiry];bound=bounds[0] if bounds else None
        if bound is not None:
            for events in [market.actions[s] for s in core.ETFS]:
                for e in events:
                    if e['type']=='CashDividends' and row.date<pd.Timestamp(e['date'])<=expiry:
                        bound=max(bound,pd.Timestamp(e['payment_date']))
            future=[d for d in market.sessions if d>=bound];bound=future[0] if future else None
        r={'at':t,'date':row.date,'next_date':dates,'next_at':nxt,'duration':durations,
           'reward':rewards,'ready_at':ready,'expired_after':bound}
        expected[t]=r
    assert len(saved)==len(expected)
    for r in saved:
        wanted=expected[r['at']]
        for k in wanted:
            if k=='reward':close(r[k],wanted[k])
            else:assert r[k]==wanted[k],(r['at'],k,r[k],wanted[k])
    return expected,terminal_count


def equations(panel,bank,ids,at,model,info):
    assert info['ready']==(len(ids)>=120) and info['columns']==COLUMNS
    assert list(pd.to_datetime(info['members'],utc=True))==ids
    assert list(pd.to_datetime(info['dates']))==list(panel.loc[ids,'date'])
    w=decay(len(ids),126);close(info['weights'],w)
    if len(ids)<120:
        assert model is None and info['stages']==[];return None,0
    X=panel.loc[ids,COLUMNS].to_numpy(float);mu=np.average(X,axis=0,weights=w)
    sd=np.sqrt(np.average((X-mu)**2,axis=0,weights=w));sd[sd<1e-12]=1.
    close(model['scaler'].mean_,mu);close(model['scaler'].scale_,sd);close(info['center'],mu);close(info['scale'],sd)
    Z=(X-mu)/sd;zm=np.average(Z,axis=0,weights=w);A=Z-zm
    matrix=A.T@(w[:,None]*A)+100*np.eye(len(COLUMNS))
    next_x=np.array([[panel.loc[n,COLUMNS].to_numpy(float) for n in bank[t]['next_at']] for t in ids])
    N=(next_x.reshape(-1,len(COLUMNS))-mu)/sd
    R=np.array([bank[t]['reward'] for t in ids]);D=np.array([bank[t]['duration'] for t in ids])
    assert np.isfinite(R).all() and (D>=1).all()
    V=np.zeros_like(R);raw={}
    assert len(model['stages'])==len(info['stages'])==20
    for j,(m,stored) in enumerate(zip(model['stages'],info['stages'])):
        target=.99**D*(R+V);mean=np.average(target,axis=0,weights=w)
        b=np.linalg.solve(matrix,A.T@(w[:,None]*(target-mean)));a=mean-zm@b
        close(m.coef_,b.T);close(m.intercept_,a);close(stored['coefficients'],b.T);close(stored['intercept'],a)
        assert stored['iteration']==j+1;near(stored['target_min'],target.min());near(stored['target_max'],target.max())
        V=np.maximum(0.,(N@b+a).max(axis=1)).reshape(len(ids),3)
        if j in [0,19] and panel.loc[at,'feature_ready']:
            raw['myopic' if j==0 else 'planning']=((panel.loc[at,COLUMNS].to_numpy(float)-mu)/sd@b+a).tolist()
    return raw if raw else None,60


def verify(run):
    run=Path(run);reg=read(run/'registration.json');manifest=read(run/'input_manifest.json');p=reg['preregistration']['protocol']
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    source=Path(manifest['source_run']);label_source=Path(manifest['label_source'])
    for f,h in manifest['source_files_sha256'].items():assert sha(source/f)==h
    for f,h in manifest['label_files_sha256'].items():assert sha(label_source/f)==h
    assert reg['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')
    market=core.load_market();assert market.manifest==manifest['full_market']
    market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    panel=pd.read_parquet(run/'panel.parquet');features=pd.read_parquet(run/'features.parquet')
    pd.testing.assert_frame_equal(panel,pd.read_parquet(source/'panel.parquet'))
    pd.testing.assert_frame_equal(features,pd.read_parquet(source/'features.parquet'))
    base=labels(label_source/'baseline_labels.json');stress=labels(label_source/'stress_labels.json')
    bank,terminal_count=transitions(market,panel,base,read_bank(run/'transitions.json'))
    predictions=read(run/'predictions.json');old=read(source/'predictions.json')['early_base_cal']
    assert predictions['source_early']==predictions['source_full_cash']==old
    assert predictions['same_action_regular_cash']==predictions['double_cost']==predictions[p['primary']]
    history={};checks=0;calchecks=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            cutoff=row.date.tz_localize(core.ET).tz_convert('UTC')
            rec=read(run/'training'/(str(row.date.date())+'.json'))
            assert pd.Timestamp(rec['at'])==at and pd.Timestamp(rec['cutoff'])==cutoff
            prior=[t for t,r in bank.items() if t<cutoff and r['date']<row.date];ids=[]
            for t in prior:
                if bank[t]['ready_at'] is None or bank[t]['ready_at']>=cutoff:break
                assert all(n<cutoff for n in bank[t]['next_at']);ids.append(t)
            pending=bank[prior[len(ids)]] if len(ids)<len(prior) else None
            expired=bool(pending and pending['expired_after'] is not None and pending['expired_after']<row.date)
            meta=rec['feedback'];assert meta['prior_dates']==len(prior) and meta['admitted_dates']==len(ids)
            assert meta['first_pending']==(str(pending['date']) if pending else None) and meta['expired_unresolved']==expired
            assert meta['known_behind_pending']==sum(bank[t]['ready_at'] is not None and bank[t]['ready_at']<cutoff for t in prior[len(ids)+1:])
            with gzip.open(run/'models'/(str(row.date.date())+'.pkl.gz'),'rb') as f:model=pickle.load(f)
            raw,count=equations(panel,bank,ids,at,model,rec['fit']);checks+=count
            if raw is None:assert rec['raw'] is None
            else:
                for mode in raw:close(rec['raw'][mode],raw[mode])
            # Use the saved as-issued values, never the independently re-fitted
            # approximation or a new historical forecast, for calibration.
            raw=rec['raw']
            for mode in ['planning','myopic']:
                members=[t for t in history if t in set(ids) and history[t] is not None and
                    (mode=='myopic' or all(n in history and history[n] is not None for n in bank[t]['next_at']))][-120:]
                cal=rec['calibration'][mode];assert list(pd.to_datetime(cal['members'],utc=True))==members
                assert cal['ready']==(len(members)>=60);w=decay(len(members),63);close(cal['weights'],w)
                errors=[]
                for t in members:
                    r=bank[t];v=[max([0.]+history[n][mode]) for n in r['next_at']] if mode=='planning' else [0.]*3
                    errors.append(.99**np.array(r['duration'])*(np.array(r['reward'])+v)-history[t][mode])
                correction=np.average(errors,axis=0,weights=w) if members else None
                if correction is None:assert cal['correction'] is None
                else:close(cal['correction'],correction)
                expected_next=[[str(n) for n in bank[t]['next_at']] for t in members] if mode=='planning' else []
                assert cal['next_forecast_members']==expected_next
                for raw_mode in [False,True]:
                    name=mode+('_raw' if raw_mode else '_calibrated');event=predictions[name][i]
                    assert rec['events'][name]==event and event['raw']==(raw[mode] if raw else None)
                    ready=raw is not None and len(members)>=60 and not expired
                    assert event['ready']==ready
                    if ready:
                        q=np.array(raw[mode])+(0 if raw_mode else correction);wait=max(0.,q[0]);j=1 if q[1]>=q[2] else 2
                        chosen=['CASH','SOXL','SOXS'][j] if q[j]>wait+1e-12 else 'CASH'
                        close(event['adjusted'],q);near(event['waiting_value'],wait);near(event['buy_advantage'],q[j]-wait)
                    else:chosen='CASH';assert event['adjusted'] is None
                    assert event['action']==chosen and event['allocation']==float(chosen!='CASH');calchecks+=1
            assert predictions['cash'][i]['action']=='CASH' and predictions['cash'][i]['allocation']==0
            history[at]=raw
            if (i+1)%75==0:print('verified through',row.date.date(),flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');settings=p[reg['phase']]
    assert frozen['predictions_sha256']==sha(run/'predictions.json') and not frozen['new_v25_capital_outcomes_seen']
    months=settings.get('months',settings.get('regression_months'));books=[]
    for name,stats in result['routes'].items():
        wanted=[e for e in predictions[name] if settings['evaluation_start']<=e['date'][:10]<=settings['evaluation_end']]
        assert frozen['routes'][name]==wanted
        books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,wanted,
            2 if name=='double_cost' else 1,stats,months,require_full_cash=name not in ['source_early','same_action_regular_cash']))
    for suffix in ['equity','trades']:
        pd.testing.assert_frame_equal(pd.read_csv(run/('source_early_'+suffix+'.csv')),
            pd.read_csv(source/('early_base_cal_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    prefix=None
    if reg['phase']=='cases':
        dev=Path(reg['authorization']['run']);past=read(dev/'predictions.json')
        for name,events in past.items():
            assert predictions[name][:len(events)]==events
        for path in sorted((dev/'training').glob('*.json')):
            assert read(path)==read(run/'training'/path.name)
        prefix={'exact_preMay_forecasts':len(next(iter(past.values()))),'all_training_records_exact':True}
    rates=[m['return'] for m in result['primary']['monthly']]
    complete=result['coverage']['complete'] and all(m['mark_complete'] for m in result['primary']['monthly'])
    if reg['phase']=='development':
        passed=bool(complete and len(rates)==4 and np.mean(rates)>=.1 and min(rates)>=-1e-12 and result['primary']['completed_trades']>=10)
        assert result['development_gate']['passed']==passed and result['case_regression']['eligible_after_verification']==passed
    receipt={'passed':True,'phase':reg['phase'],'independent_transitions':len(bank),'joint_terminal_receipts':terminal_count,
        'daily_membership_checks':len(panel),'independent_ridge_head_equations':checks,'calibration_and_action_checks':calchecks,
        'capital_books':books,'source_control_exact':True,'case_prefix':prefix,'results_sha256':sha(run/'results.json'),
        'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path)
    verify(parser.parse_args().run)
