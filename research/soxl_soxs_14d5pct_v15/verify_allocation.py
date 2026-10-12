"""Independent distribution arithmetic, decisions and fractional capital reconstruction."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from allocation import core,restrict_market,COLUMNS,ROUTES
from verify_continuation import read,labels,key,near,raw_exit
from risk_ledger import replay as constant_replay


def book_check(run,name,market,features,lab,events,multiple,stats,months):
    eq=pd.read_csv(run/(name+'_equity.csv'),parse_dates=['date']).set_index('date')
    try:trades=pd.read_csv(run/(name+'_trades.csv'))
    except pd.errors.EmptyDataError:trades=pd.DataFrame()
    actual={pd.Timestamp(r.decision_at):r for r in trades.itertuples()};actions={pd.Timestamp(e['at']):e for e in events}
    cash=1.;unsettled=[];receivables=[];held=None;units=0.;count=0
    for day,mark in eq.iterrows():
        cash+=sum(v for d,v in unsettled if d<=day);unsettled=[(d,v) for d,v in unsettled if d>day]
        cash+=sum(v for d,v in receivables if d<=day);receivables=[(d,v) for d,v in receivables if d>day]
        if held is not None:
            for d in held['dividends']:
                if d['ex']==day:
                    value=units*d['economic_amount']
                    if d['pay']<=day:cash+=value
                    else:receivables.append((d['pay'],value))
        at=features[features.date==day].index[0];e=actions[at]
        assert 0<=e['allocation']<=1 and (e['action'] in core.ETFS)==(e['allocation']>0)
        if held is None and cash>1e-15 and e['allocation']>0:
            chosen=lab[key(at,e['action'])]
            if 'entry_fill_economic' in chosen:
                assert at in actual and actual[at].symbol==e['action']
                capital=cash*e['allocation'];units=capital/(chosen['entry_fill_economic']*(1+.0005*multiple))
                near(actual[at].capital,capital);near(actual[at].allocation,e['allocation']);near(actual[at].units_economic,units)
                assert actual[at].reason==chosen['reason']
                for c in ['entry_fill_economic','exit_fill_economic','net_return','dividend_per_economic_unit']:
                    if np.isfinite(chosen.get(c,np.nan)):near(getattr(actual[at],c),chosen[c])
                if chosen['exit_at'] is not None:assert pd.Timestamp(actual[at].exit_at)==chosen['exit_at']
                held=chosen;cash-=capital;count+=1
        else:assert at not in actual
        if held is not None and held['exit_at'] is not None and held['exit_at'].tz_convert(core.ET).date()==day.date():
            unsettled.append((market.sessions[market.sessions>day][0],units*held['exit_fill_economic']*(1-.0005*multiple)));held=None
        asset=units*market.daily[held['symbol']].loc[day,'close'] if held else 0.
        pending=sum(v for _,v in unsettled);owed=sum(v for _,v in receivables)
        for field,value in [('cash_settled',cash),('cash_unsettled',pending),('receivables',owed),('position_value',asset),('equity',cash+pending+owed+asset)]:near(mark[field],value)
        assert mark.symbol==(held['symbol'] if held else 'CASH') and mark.mark_complete
    assert count==len(trades)
    prior=1.;rates=[]
    for month,g in eq.groupby(eq.index.to_period('M')):
        rate=g.equity.iloc[-1]/prior-1;prior=g.equity.iloc[-1];rates.append((str(month),rate))
        near(rate,next(m['return'] for m in stats['monthly'] if m['month']==str(month)))
    avg=np.mean([v for m,v in rates if m in months]);near(stats['primary_average'],avg);near(stats['continuous_return'],eq.equity.iloc[-1]-1)
    assert stats['primary_pass']==(avg>=.1 and all(v>=-1e-12 for _,v in rates))
    return {'name':name,'actual_trades':count,'daily_marks':len(eq),'capital_reconciled':True}


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
    predictions=read(run/'predictions.json');decisions=read(run/'decisions_frozen.json');checks=weightchecks=0
    for i,e in enumerate(events):
        at=pd.Timestamp(e['at']);date=pd.Timestamp(e['date']);daycut=date.tz_localize(core.ET).tz_convert('UTC')
        record=read(run/'training'/(str(date.date())+'.json'));assert pd.Timestamp(record['cutoff'])==daycut
        ids=[t for t in sorted(releases) if t<daycut and releases[t]<daycut][-120:] if e['ready'] else []
        for name,(mode,gamma,fractions) in ROUTES.items():
            fit=record['routes'][name]['fit'];event=predictions[name][i];assert record['routes'][name]['event']==event
            assert list(pd.to_datetime(fit['members'],utc=True))==ids
            assert list(pd.to_datetime(fit['dates']))==[pd.Timestamp(source_events[t]['date']) for t in ids]
            if len(ids)<60:
                assert not fit['ready'] and event['action']=='CASH' and event['allocation']==0.
            else:
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
                w/=w.sum();r=np.array([outcomes[t,s] for t in ids]);np.testing.assert_allclose(info['returns'],r,atol=1e-12,rtol=1e-12)
                np.testing.assert_allclose(info['weights'],w,atol=1e-12,rtol=1e-12);effective=1/sum(w*w);near(info['effective_dates'],effective)
                near(info['mean_return'],w@r);near(info['second_moment'],w@(r*r));near(info['loss_frequency'],w@(r<0))
                mass=.05;tail=0.
                for j in np.argsort(r,kind='stable'):
                    take=min(mass,w[j]);tail+=take*r[j];mass-=take
                    if mass<=0:break
                near(info['lower_5pct_mean_return'],tail/.05);weightchecks+=1
                if effective<20:assert not fit['ready'] and event['allocation']==0.
                else:
                    scores=[]
                    for fraction in fractions:
                        wealth=1+fraction*r
                        values=np.log(wealth) if gamma==1 else (wealth**(1-gamma)-1)/(1-gamma)
                        scores.append(float(w@values))
                    np.testing.assert_allclose(fit['utility']['scores'],scores,atol=1e-12,rtol=1e-10)
                    best=next(j for j,v in enumerate(scores) if v>=max(scores)-1e-12);fraction=float(fractions[best])
                    assert fit['ready'] and fit['utility']['selected_index']==best
                    assert event['allocation']==fraction and event['action']==(s if fraction else 'CASH')
            checks+=1
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
        a=pd.read_csv(run/(n+'_trades.csv'));b=pd.read_csv(source/(old+'_trades.csv'))
        pd.testing.assert_frame_equal(a[b.columns],b,atol=1e-12,rtol=1e-12)
    test=features[(features.date>=settings['evaluation_start'])&(features.date<=settings['evaluation_end'])]
    half_actions={pd.Timestamp(e['at']):e['action'] for e in decisions['routes']['source_half']}
    old_half=constant_replay(market,test,list(base.values()),half_actions,settings['evaluation_start'],settings['evaluation_end'],allocation=.5)
    saved_half=pd.read_csv(run/'source_half_equity.csv',parse_dates=['date']).set_index('date')
    pd.testing.assert_frame_equal(saved_half,old_half[0],atol=1e-12,rtol=1e-12)
    saved_trades=pd.read_csv(run/'source_half_trades.csv')
    for c in ['capital','units_economic','net_return']:
        np.testing.assert_allclose(saved_trades[c],old_half[1][c],atol=1e-12,rtol=1e-12)
    exits=0
    for lab,multiple in [(base,1),(stress,2)]:
        for label in lab.values():
            if 'entry_fill_economic' in label:raw_exit(market,label,None,multiple);exits+=1
    result={'passed':True,'phase':reg['phase'],'bank_vectors_checked':bankchecks,'daily_membership_decision_checks':checks,
        'distribution_weight_and_tail_checks':weightchecks,'raw_exit_checks':exits,'capital_books':books,
        'v14_full_and_direction_ledgers_exactly_reproduced':True,'v3_constant_half_ledger_exactly_reproduced':True,
        'results_sha256':hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),
        'engineering_checks_are_not_return_acceptance':True}
    core.dump(run/'verification.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
