"""Independent mean-constrained optimization and complete capital audit."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp
from anchored_allocation import core,restrict_market,COLUMNS,ROUTES
from verify_allocation import book_check
from verify_continuation import read,labels,key,near,raw_exit
from risk_ledger import replay as constant_replay


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json');source=Path(manifest['source_run'])
    for f,digest in reg['source_sha256'].items():assert hashlib.sha256((core.ROOT/f).read_bytes()).hexdigest()==digest
    for f,digest in manifest['source_files_sha256'].items():assert hashlib.sha256((source/f).read_bytes()).hexdigest()==digest
    for f,field in [('distribution_bank.parquet','bank_sha256'),('features.parquet','features_sha256')]:assert hashlib.sha256((run/f).read_bytes()).hexdigest()==manifest[field]
    cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive']);original=core.load_market();assert original.manifest==manifest['full_market'];market=restrict_market(original,cutoff)
    assert all((f.end_at<cutoff).all() and (f.available_at<cutoff).all() for f in market.bars.values())
    features=pd.read_parquet(run/'features.parquet');panel=pd.read_parquet(source/'panel.parquet');bank=pd.read_parquet(run/'distribution_bank.parquet')
    events=read(source/'predictions.json')['regret_regime'];base=labels(source/'baseline_labels.json');stress=labels(source/'stress_labels.json')
    source_events={pd.Timestamp(e['at']):e for e in events};vectors={};outcomes={};releases={};bankchecks=0
    for e in events:
        if not e['ready']:continue
        at=pd.Timestamp(e['at']);r=panel.loc[at];pair=[base[key(at,s)] for s in core.ETFS]
        if all(x['complete_window'] and x.get('label_available_at') is not None and np.isfinite(x.get('net_return',np.nan)) for x in pair):
            assert all(x['label_available_at']>=x['expiry_at']+pd.Timedelta(seconds=1) for x in pair)
            releases[at]=max(x['label_available_at'] for x in pair)
        for s,sign in [('SOXL',1.),('SOXS',-1.)]:
            vals=[e['adjusted'][s],sign*e['adjusted']['D'],sign*r.relative_5,sign*r.breadth_5,sign*r.volume_direction_diffusion,r.log_sector_volatility]
            row=bank.loc[(at,s)];np.testing.assert_allclose(row[COLUMNS].astype(float),vals,atol=1e-12,rtol=1e-12)
            ret=base[key(at,s)].get('net_return',np.nan);np.testing.assert_allclose(row.net_return,ret,atol=1e-12,rtol=1e-12,equal_nan=True)
            assert row.date==pd.Timestamp(e['date']);vectors[at,s]=np.array(vals);outcomes[at,s]=ret;bankchecks+=1
    predictions=read(run/'predictions.json');decisions=read(run/'decisions_frozen.json');checks=weightchecks=anchorchecks=0;errors=[]
    for i,e in enumerate(events):
        at=pd.Timestamp(e['at']);date=pd.Timestamp(e['date']);daycut=date.tz_localize(core.ET).tz_convert('UTC')
        record=read(run/'training'/(str(date.date())+'.json'));assert pd.Timestamp(record['cutoff'])==daycut
        ids=[t for t in sorted(releases) if t<daycut and releases[t]<daycut][-120:] if e['ready'] else []
        for name,(mode,gamma,fractions,anchor,source_gate) in ROUTES.items():
            fit=record['routes'][name]['fit'];event=predictions[name][i];assert record['routes'][name]['event']==event
            assert list(pd.to_datetime(fit['members'],utc=True))==ids
            assert list(pd.to_datetime(fit['dates']))==[pd.Timestamp(source_events[t]['date']) for t in ids]
            checks+=1
            if len(ids)<60:
                assert not fit['ready'] and event['action']=='CASH' and event['allocation']==0.;continue
            s=e['top_symbol'];b=2.**(-np.arange(len(ids))[::-1]/63.);b/=b.sum();w=b.copy();info=fit['distribution']
            np.testing.assert_allclose(info['base_weights'],b,atol=1e-12,rtol=1e-12)
            cols=6 if mode=='local' else 2 if mode=='score' else 0
            if cols:
                X=np.array([vectors[t,s][:cols] for t in ids]);Q=vectors[at,s][:cols]
                center=np.average(X,axis=0,weights=b);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=b));scale[scale<1e-12]=1.
                distance=np.sum(((X-Q)/scale)**2,axis=1)/cols;bandwidth=max(sorted(distance)[39],1e-12)
                w*=np.exp(-distance/(2*bandwidth))
                np.testing.assert_allclose(info['center'],center,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(info['scale'],scale,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(info['squared_distances'],distance,atol=1e-10,rtol=1e-10);near(info['bandwidth_squared'],bandwidth)
            w/=w.sum();prior=w.copy();r=np.array([outcomes[t,s] for t in ids]);x=np.log1p(r)
            np.testing.assert_allclose(info['prior_weights'],prior,atol=1e-12,rtol=1e-12);near(info['prior_effective_dates'],1/(prior@prior))
            if info['prior_effective_dates']<20:
                assert not fit['ready'] and fit['reason']=='insufficient_prior_effective_dates';continue
            if anchor:
                target=vectors[at,s][0];positive=prior>0;low=min(x[positive]);high=max(x[positive]);a=fit['anchor']
                near(a['target_log_mean'],target);near(a['prior_log_mean'],prior@x);near(a['support_min'],low);near(a['support_max'],high)
                valid=(low+1e-12<target<high-1e-12) if high-low>1e-12 else abs(target-prior@x)<=1e-12
                assert a['valid']==valid
                if not valid:
                    assert not fit['ready'] and event['allocation']==0. and fit['reason']=='mean_outside_empirical_support';continue
                logprior=np.log(prior[positive]);xs=x[positive]
                def posterior(value):
                    z=logprior+value*xs
                    return np.exp(z-logsumexp(z))
                if high-low>1e-12:
                    solution=brentq(lambda v:float(posterior(v)@xs-target),-1e6,1e6,xtol=1e-12)
                    near(a['lambda'],solution);w=np.zeros_like(prior);w[positive]=posterior(solution)
                else:solution=0.;w=prior.copy()
                mask=w>0;kl=float(w[mask]@np.log(w[mask]/prior[mask]));near(a['kl_divergence'],max(0.,kl))
                near(w@x,target);errors.append(abs(w@x-target))
                # Convex KKT stationarity certifies the unique feasible KL minimum.
                intercept=np.log(w[mask]/prior[mask])-solution*x[mask]
                np.testing.assert_allclose(intercept,np.full(len(intercept),intercept[0]),atol=1e-10,rtol=1e-10)
                anchorchecks+=1
            else:assert fit['anchor'] is None
            np.testing.assert_allclose(info['returns'],r,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(info['weights'],w,atol=1e-11,rtol=1e-10);effective=1/(w@w);near(info['effective_dates'],effective)
            for field,value in [('mean_return',w@r),('mean_log_return',w@x),('second_moment',w@(r*r)),('loss_frequency',w@(r<0))]:near(info[field],value)
            mass=.05;tail=0.
            for j in np.argsort(r,kind='stable'):
                take=min(mass,w[j]);tail+=take*r[j];mass-=take
                if mass<=0:break
            near(info['lower_5pct_mean_return'],tail/.05);weightchecks+=1
            if effective<20:
                assert not fit['ready'] and event['allocation']==0. and fit['reason']=='insufficient_posterior_effective_dates';continue
            scores=[]
            for fraction in fractions:
                wealth=1+fraction*r
                values=np.log(wealth) if gamma==1 else (wealth**(1-gamma)-1)/(1-gamma)
                scores.append(float(w@values))
            np.testing.assert_allclose(fit['utility']['scores'],scores,atol=1e-12,rtol=1e-10)
            best=next(j for j,v in enumerate(scores) if v>=max(scores)-1e-12);fraction=float(fractions[best])
            assert fit['ready'] and fit['utility']['selected_index_before_source_gate']==best
            if source_gate and vectors[at,s][0]<=0:fraction=0.
            assert event['allocation']==fraction and event['action']==(s if fraction else 'CASH')
        for n,f,direction in [('source_full',1.,False),('source_half',.5,False),('source_direction',1.,True)]:
            buy=e['ready'] and (direction or e['action']!='CASH');current=predictions[n][i]
            assert current['allocation']==(f if buy else 0.) and current['action']==(e['top_symbol'] if buy else 'CASH')
        assert predictions['cash'][i]['allocation']==0. and predictions['cash'][i]['action']=='CASH'
        assert predictions['double_cost'][i]==predictions[p['primary']][i]
    results=read(run/'results.json');settings=p[reg['phase']];months=settings.get('months',settings.get('regression_months'));books=[]
    for n,stats in results['routes'].items():
        wanted=[e for e in predictions[n] if settings['evaluation_start']<=e['date'][:10]<=settings['evaluation_end']]
        assert decisions['routes'][n]==wanted
        books.append(book_check(run,n,market,features,stress if n=='double_cost' else base,wanted,2 if n=='double_cost' else 1,stats,months))
    for n,old in [('source_full','regret_regime'),('source_direction','same_primary_direction')]:
        pd.testing.assert_frame_equal(pd.read_csv(run/(n+'_equity.csv')),pd.read_csv(source/(old+'_equity.csv')),atol=1e-12,rtol=1e-12)
        a=pd.read_csv(run/(n+'_trades.csv'));b=pd.read_csv(source/(old+'_trades.csv'));pd.testing.assert_frame_equal(a[b.columns],b,atol=1e-12,rtol=1e-12)
    test=features[(features.date>=settings['evaluation_start'])&(features.date<=settings['evaluation_end'])]
    half_actions={pd.Timestamp(e['at']):e['action'] for e in decisions['routes']['source_half']}
    old_half=constant_replay(market,test,list(base.values()),half_actions,settings['evaluation_start'],settings['evaluation_end'],allocation=.5)
    pd.testing.assert_frame_equal(pd.read_csv(run/'source_half_equity.csv',parse_dates=['date']).set_index('date'),old_half[0],atol=1e-12,rtol=1e-12)
    if reg['phase']=='development':
        previous=core.ROOT/'.cache/soxl_soxs_14d5pct_v15/runs/allocation_20261012'
        for ending in ['equity','trades','decisions','orders']:
            pd.testing.assert_frame_equal(pd.read_csv(run/('unanchored_local_power_'+ending+'.csv')),pd.read_csv(previous/('local_power_'+ending+'.csv')),atol=1e-12,rtol=1e-12)
    exits=0
    for lab,multiple in [(base,1),(stress,2)]:
        for label in lab.values():
            if 'entry_fill_economic' in label:raw_exit(market,label,None,multiple);exits+=1
    result={'passed':True,'phase':reg['phase'],'bank_vectors_checked':bankchecks,'daily_membership_decision_checks':checks,
        'distribution_weight_and_tail_checks':weightchecks,'independent_root_and_KKT_checks':anchorchecks,'maximum_independent_mean_error':max(errors,default=0.),
        'raw_exit_checks':exits,'capital_books':books,'unchanged_v14_controls_and_v3_half_exact':True,
        'v15_unanchored_exact':reg['phase']=='development','results_sha256':hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),
        'engineering_checks_are_not_return_acceptance':True}
    core.dump(run/'verification.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
