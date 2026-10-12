"""Independent pre-case boundary, parent calibration, model and capital audit."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from uncertainty import core,MANAGER_COLUMNS,restrict_market
from verify_micro import verify_states
from verify_continuation import read,labels,key,near,raw_exit


def book_check(run,name,market,features,lab,actions,multiple,stats,months):
    equity=pd.read_csv(run/(name+'_equity.csv'),parse_dates=['date']).set_index('date')
    try:trades=pd.read_csv(run/(name+'_trades.csv'))
    except pd.errors.EmptyDataError:trades=pd.DataFrame()
    actual={pd.Timestamp(r.decision_at):r for r in trades.itertuples()}
    cash=1.;unsettled=[];receivables=[];held=None;units=0.;count=0
    for day,mark in equity.iterrows():
        cash+=sum(v for d,v in unsettled if d<=day);unsettled=[(d,v) for d,v in unsettled if d>day]
        cash+=sum(v for d,v in receivables if d<=day);receivables=[(d,v) for d,v in receivables if d>day]
        if held is not None:
            for d in held['dividends']:
                if d['ex']==day:
                    value=units*d['economic_amount']
                    if d['pay']<=day:cash+=value
                    else:receivables.append((d['pay'],value))
        at=features[features.date==day].index[0]
        if held is None and cash>1e-15 and actions[at] in core.ETFS:
            chosen=lab[key(at,actions[at])]
            if 'entry_fill_economic' in chosen:
                assert at in actual and actual[at].symbol==actions[at]
                near(actual[at].capital,cash);units=cash/(chosen['entry_fill_economic']*(1+.0005*multiple))
                near(actual[at].units_economic,units);assert actual[at].reason==chosen['reason']
                for c in ['entry_fill_economic','exit_fill_economic','net_return','dividend_per_economic_unit']:
                    if np.isfinite(chosen.get(c,np.nan)):near(getattr(actual[at],c),chosen[c])
                if chosen['exit_at'] is not None:assert pd.Timestamp(actual[at].exit_at)==chosen['exit_at']
                held=chosen;cash=0.;count+=1
        else:assert at not in actual
        if held is not None and held['exit_at'] is not None and held['exit_at'].tz_convert(core.ET).date()==day.date():
            due=market.sessions[market.sessions>day][0]
            unsettled.append((due,units*held['exit_fill_economic']*(1-.0005*multiple)));held=None
        asset=units*market.daily[held['symbol']].loc[day,'close'] if held else 0.
        pending=sum(v for _,v in unsettled);owed=sum(v for _,v in receivables)
        for field,value in [('cash_settled',cash),('cash_unsettled',pending),('receivables',owed),('position_value',asset),('equity',cash+pending+owed+asset)]:near(mark[field],value)
        assert mark.symbol==(held['symbol'] if held else 'CASH') and mark.mark_complete
    assert count==len(trades)
    prior=1.;rates=[]
    for month,g in equity.groupby(equity.index.to_period('M')):
        rate=g.equity.iloc[-1]/prior-1;prior=g.equity.iloc[-1];rates.append((str(month),rate))
        near(rate,next(m['return'] for m in stats['monthly'] if m['month']==str(month)))
    avg=np.mean([v for m,v in rates if m in months]);near(stats['primary_average'],avg)
    near(stats['continuous_return'],equity.equity.iloc[-1]-1)
    assert stats['primary_pass']==(avg>=.1 and all(v>=-1e-12 for _,v in rates))
    return {'name':name,'actual_trades':count,'daily_marks':len(equity),'capital_reconciled':True}


def verify(run):
    run=Path(run);registration=read(run/'registration.json');protocol=registration['preregistration']['protocol']
    phase=registration['phase'];manifest=read(run/'input_manifest.json');source=Path(manifest['source_run'])
    cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive']);cutday=cutoff.tz_convert(core.ET).tz_localize(None).normalize()
    if phase=='development':assert cutoff==pd.Timestamp('2026-05-01',tz=core.ET)
    for name,value in registration['source_sha256'].items():assert hashlib.sha256((core.ROOT/name).read_bytes()).hexdigest()==value
    for name,value in manifest['source_files_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==value
    for name,value in manifest['source_model_sha256'].items():
        assert pd.Timestamp(name[:10]).tz_localize(core.ET)<cutoff
        assert hashlib.sha256((source/'models'/name).read_bytes()).hexdigest()==value
    original=core.load_market();assert original.manifest==manifest['full_source_market'];market=restrict_market(original,cutoff)
    for s,f in market.bars.items():
        assert (f.end_at<cutoff).all() and (f.available_at<cutoff).all()
        assert market.daily[s].index.max()<cutday and (market.daily[s].available_at<cutoff).all()
        assert all(pd.Timestamp(a['date'])<cutday for a in market.actions[s])
    features=pd.read_parquet(run/'features.parquet');assert (features.index<cutoff).all()
    baseline=labels(run/'baseline_labels.json');stress_base=labels(run/'stress_baseline_labels.json')
    states={};state_checks=0
    for name,lab,file,multiple in [('base',baseline,'review_forecasts.json',1),('stress',stress_base,'stress_review_forecasts.json',2)]:
        states[name],n,_=verify_states(market,features,lab,read(run/file),multiple);state_checks+=n
        assert all(r['at']<cutoff for r in states[name].values())
    releases={}
    for at,row in features.iterrows():
        pair=[baseline[key(at,s)] for s in core.ETFS]
        if all(x.get('complete_window') and x.get('label_available_at') is not None and np.isfinite(x.get('net_return',np.nan)) for x in pair):
            for x in pair:assert x['label_available_at']>=x['expiry_at']+pd.Timedelta(seconds=1)
            releases[row.date]=max(x['label_available_at'] for x in pair)
    by_day={name:{} for name in states}
    for name,rows in states.items():
        for r in rows.values():by_day[name].setdefault(r['at'].tz_convert(core.ET).tz_localize(None).normalize(),[]).append(r)
    bank=pd.read_parquet(run/'calibration_bank.parquet');cols=protocol['local_columns']
    assert list(bank.index)==list(states['base'])
    for i,r in states['base'].items():
        for c in cols:
            value=r['raw'] if c=='raw' else r['state'][c]
            if value is not None:near(bank.loc[i,c],value)
        if np.isfinite(r['target']):near(bank.loc[i,'target'],r['target'])
    parent_groups={d:g for d,g in bank.groupby('entry_date',sort=True)}
    policies=[protocol['primary'],protocol['diagnostic'],'double_cost'];closed={n:set() for n in policies};intents={n:{} for n in policies}
    source_fit_checks=0;raw_checks=0;bound_checks=0;local_checks=0;refits=0;all_events=[]
    with threadpool_limits(limits=1):
        for at,row in features.iterrows():
            day=row.date;day_cutoff=day.tz_localize(core.ET).tz_convert('UTC')
            saved=read(run/'training'/(str(day.date())+'.json'));pool=saved['pool']
            info=read(source/'training'/(str(day.date())+'.json'));assert pd.Timestamp(info['cutoff'])==day_cutoff
            fit_dates=[d for d,r in sorted(releases.items()) if r<day_cutoff and d.tz_localize(core.ET)<day_cutoff][-252:]
            train=[r for r in states['base'].values() if r['entry_date'] in set(fit_dates) and np.isfinite(r['target']) and r['state'] is not None]
            assert info['fits']['manager']['members']==[r['id'] for r in train];source_fit_checks+=1
            with gzip.open(source/'models'/(str(day.date())+'.pkl.gz'),'rb') as f:model=pickle.load(f)['manager']
            for name in states:
                records=by_day[name].get(day,[])
                if model is None:assert all(r['raw'] is None for r in records)
                elif records:
                    X=np.array([[r['state'][c] for c in MANAGER_COLUMNS] for r in records])
                    np.testing.assert_allclose(model.predict(X),[r['raw'] for r in records],atol=1e-13,rtol=1e-11);raw_checks+=len(records)
            if model is not None and str(day.date()) in ['2025-09-02','2025-12-01','2026-03-02']:
                X=np.array([[r['state'][c] for c in MANAGER_COLUMNS] for r in train]);y=np.array([r['target'] for r in train])
                counts=pd.Series([r['entry_date'] for r in train]).value_counts();weights=[1/counts[r['entry_date']] for r in train]
                fitted=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,learning_rate=.05,
                    l2_regularization=5,early_stopping=False,random_state=814).fit(X,y,sample_weight=weights)
                np.testing.assert_allclose(fitted.predict(X),model.predict(X),atol=1e-13,rtol=1e-11);refits+=1
            eligible=[]
            for d,g in parent_groups.items():
                if d not in releases or releases[d]>=day_cutoff or d.tz_localize(core.ET)>=day_cutoff:continue
                if not (g['at']<day_cutoff).all() or not np.isfinite(g[[*cols,'target']].to_numpy(dtype=float)).all():continue
                eligible.append(d)
            eligible=eligible[-126:];recent=eligible[-60:]
            assert list(pd.to_datetime(pool['eligible_parents']))==eligible
            assert list(pd.to_datetime(pool['quantile_parents']))==recent
            maxima=[float((parent_groups[d].target-parent_groups[d].raw).max()) for d in recent]
            np.testing.assert_allclose(pool['parent_maxima'],maxima,rtol=1e-12,atol=1e-13)
            ready=len(recent)>=40;q=None
            assert pool['ready']==ready
            if ready:
                rank=min(len(recent),math.ceil((len(recent)+1)*.9));q=sorted(maxima)[rank-1]
                assert pool['quantile_rank_one_based']==rank;near(pool['residual_upper'],q)
            local_frame=bank[bank.entry_date.isin(eligible)];scale=None
            if len(eligible)>=40:
                means=np.array([parent_groups[d][cols].mean().to_numpy() for d in eligible])
                moments=np.array([(parent_groups[d][cols]**2).mean().to_numpy() for d in eligible])
                center=means.mean(axis=0);scale=np.sqrt(np.maximum(moments.mean(axis=0)-center**2,0));scale[scale<1e-12]=1.
                np.testing.assert_allclose(pool['local_scale'],scale,rtol=1e-10,atol=1e-12)
                np.testing.assert_allclose(pool['local_center'],center,rtol=1e-10,atol=1e-12)
            position=0
            for name in policies:
                rows=by_day['stress' if name=='double_cost' else 'base'].get(day,[])
                for r in sorted(rows,key=lambda x:(x['at'],x['key'])):
                    if r['key'] in closed[name]:continue
                    e=saved['events'][position];position+=1
                    assert e['policy']==name and e['id']==r['id']
                    if r['raw'] is None:
                        assert not e['ready'] and e['action']=='HOLD';continue
                    bound=None
                    if name!=protocol['diagnostic']:
                        assert e['ready']==ready
                        if ready:bound=r['raw']+q
                    else:
                        assert e['ready']==(scale is not None)
                        if scale is not None:
                            query=np.array([r['raw'] if c=='raw' else r['state'][c] for c in cols])
                            dist=np.sum(((local_frame[cols].to_numpy()-query)/scale)**2,axis=1)
                            temp=pd.DataFrame({'distance':dist,'parent':local_frame.entry_date.to_numpy(),'id':local_frame.index})
                            chosen=temp.sort_values(['distance','id'],kind='stable').drop_duplicates('parent').head(40)
                            assert chosen.id.tolist()==e['neighbors'] and chosen.parent.nunique()==40
                            np.testing.assert_allclose(chosen.distance,e['distances'],rtol=1e-9,atol=1e-10)
                            targets=bank.loc[chosen.id,'target'].to_numpy();mean=targets.mean();se=targets.std(ddof=1)/40**.5
                            near(e['mean'],mean);near(e['se'],se);bound=mean+1.645*se;local_checks+=1
                    if bound is not None:near(e['bound'],bound);bound_checks+=1
                    assert e['action']==('EXIT' if bound is not None and bound<0 else 'HOLD')
                    if e['action']=='EXIT' and r['sale'] is not None:
                        intents[name].setdefault(r['key'],r['at']);closed[name].add(r['key'])
            assert position==len(saved['events']);all_events.extend(saved['events'])
    assert all_events==read(run/'manager_predictions.json')
    label_sets={n:labels(run/(n+'_labels.json')) for n in policies+['original_v10_point']}
    old_intents={}
    for e in read(source/'manager_predictions.json'):
        if e['cost_path']=='base' and pd.Timestamp(e['at'])<cutoff and e['action']=='EXIT':old_intents.setdefault(e['key'],pd.Timestamp(e['at']))
    raw_exit_checks=0
    for name,lab,pending,multiple in [('baseline',baseline,{},1),('stress_baseline',stress_base,{},2),
        *[(n,label_sets[n],intents[n],2 if n=='double_cost' else 1) for n in policies],
        ('original_v10_point',label_sets['original_v10_point'],old_intents,1)]:
        for k,x in lab.items():
            if 'entry_fill_economic' not in x:continue
            row=dict(x)
            if row['reason'] in ['uncertainty_exit','intraday_value_exit']:row['reason']='continuation_exit'
            raw_exit(market,row,pending.get(k),multiple);raw_exit_checks+=1
            assert x['exit_at'] is None or x['exit_at']<cutoff
            base=(stress_base if multiple==2 else baseline)[k]
            assert x['label_available_at']==base['label_available_at']
    entry=read(run/'entry_predictions.json');actions={pd.Timestamp(e['at']):e['action'] for e in entry}
    assert all(at<cutoff for at in actions)
    results=read(run/'results.json');months=protocol[phase]['months' if phase=='development' else 'regression_months']
    definitions=[*[(n,label_sets[n],2 if n=='double_cost' else 1) for n in policies],
        ('fixed_contract',baseline,1),('original_v10_point',label_sets['original_v10_point'],1),('double_cost_fixed',stress_base,2)]
    books=[book_check(run,n,market,features,lab,actions,multiple,results['routes'][n],months) for n,lab,multiple in definitions]
    stats=results['primary'];passed=(stats['primary_average']>=.1 and all(m['return']>=-1e-12 and m['mark_complete'] for m in stats['monthly'])
        and results['coverage']['complete'] and stats['completed_trades']>=10)
    if phase=='development':
        assert results['development_gate']['passed']==passed and not results['case_regression']['executed']
        assert results['case_regression']['eligible_after_verification']==passed
    frozen=read(run/'decisions_frozen.json');assert frozen['manager_sha256']==hashlib.sha256((run/'manager_predictions.json').read_bytes()).hexdigest()
    out={'passed':True,'phase':phase,'source_daily_training_membership_checks':source_fit_checks,
        'raw_forecasts_checked':raw_checks,'independent_pre_case_model_refits':refits,'observed_state_checks':state_checks,
        'empirical_bounds_checked':bound_checks,'local_unique_parent_neighbor_checks':local_checks,
        'raw_counterfactual_exit_checks':raw_exit_checks,'capital_books':books,
        'observation_cutoff_exclusive':str(cutoff),'post_cutoff_prices_or_decisions_used':False,
        'development_gate_independently_recomputed':phase=='development','results_sha256':hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),
        'no_effect_pass_inferred':True}
    core.dump(run/'verification.json',out);print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)
