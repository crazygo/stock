"""Rebuild selected-trade labels, neighborhoods, saved-policy suffix and NAV."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sequential import core, META


def near(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)


def verify(run):
    run=Path(run);protocol=json.loads((run/'registration.json').read_text())['preregistration']['protocol']
    manifest=json.loads((run/'input_manifest.json').read_text());source=Path(manifest['source_run'])
    for name,value in manifest['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==value
    features=pd.read_parquet(source/'features.parquet')
    labels=pd.DataFrame(json.loads((source/'labels.json').read_text()))
    labels['decision_at']=pd.to_datetime(labels.decision_at,utc=True)
    labels['label_available_at']=pd.to_datetime(labels.label_available_at,utc=True)
    tables={s:labels[labels.symbol.eq(s)].set_index('decision_at') for s in core.ETFS}
    releases={}
    for date,g in features[features.regular_context_ready.eq(1.)].groupby('date'):
        part=labels[labels.decision_at.isin(g.index)]
        if len(part)==2*len(g) and part.complete_window.all() and part.label_available_at.notna().all() and np.isfinite(part.net_return).all():
            releases[date]=part.label_available_at.max()
    meta={};source_forecasts={}
    for variant in ['base','breadth']:
        books={a:pd.read_json(source/(variant+'__'+a+'_predictions.json'),orient='table') for a in ['ridge_log','boosted_log','ensemble_log']}
        source_forecasts[variant]=books['ensemble_log'];rows=[]
        for at,row in features.iterrows():
            e=books['ensemble_log'].loc[at]
            if not row.regular_context_ready or not e.ready:
                continue
            assert pd.Timestamp(e.training_cutoff)<=at
            scores=[e.SOXL_adjusted,e.SOXS_adjusted]
            if not np.isfinite(scores).all():continue
            j=0 if scores[0]>=scores[1] else 1;s=core.ETFS[j];sign=1. if j==0 else -1.
            x=[scores[j],scores[j]-scores[1-j],abs(books['ridge_log'].loc[at,s+'_adjusted']-books['boosted_log'].loc[at,s+'_adjusted']),
               sign,row.SOXX_vol20,sign*row.x_session_return,sign*row.x_vwap_distance,sign*row.q_recent15]
            if not np.isfinite(x).all():continue
            rows.append({'at':at,'date':row.date,'clock':row.clock,'symbol':s,'release':releases.get(row.date,pd.NaT),
                         'buy':np.log1p(tables[s].loc[at,'net_return']),**dict(zip(META,x))})
        meta[variant]=pd.DataFrame(rows).set_index('at')
    policies={n:pd.DataFrame(json.loads((run/(n+'_predictions.json')).read_text())) for n in protocol['policies']}
    for name,p in policies.items():
        p['at']=pd.to_datetime(p['at'],utc=True);policies[name]=p.set_index('at').sort_index()
    records={pd.Timestamp(p.stem):json.loads(p.read_text()) for p in sorted((run/'training').glob('*.json'))}
    checks=0;ready=0;suffixes=0;negative_suffixes=0;buy_count=0
    for name,settings in protocol['policies'].items():
        history=policies[name];states=meta[settings['input']].copy();wait={}
        for at,row in states.iterrows():
            future=history[(history.index>at)&(history.index.tz_convert(core.ET).date==row.date.date())]
            executed=future[future.action.isin(core.ETFS)]
            value=0.
            if len(executed):
                t=executed.index[0];s=executed.action.iloc[0]
                value=float(np.log1p(tables[s].loc[t,'net_return']))
            wait[at]=value;suffixes+=1;negative_suffixes+=int(value<0)
        states['wait']=pd.Series(wait)
        for clock in protocol['analysis_clocks_et'][1:]:
            saved=pd.read_json(run/(name+'_'+clock.replace(':','')+'_mature_states.json'),orient='table')
            expected=states[states.clock.eq(clock)]
            assert saved.index.equals(expected.index)
            np.testing.assert_allclose(saved[META],expected[META],atol=1e-13,rtol=1e-10)
            np.testing.assert_allclose(saved.buy_return,expected.buy,atol=1e-13,rtol=1e-10)
            np.testing.assert_allclose(saved.wait_return,expected.wait,atol=1e-13,rtol=1e-10)
            assert list(pd.to_datetime(saved.release,utc=True))==list(pd.to_datetime(expected.release,utc=True))
        for date,record in records.items():
            cutoff=date.tz_localize(core.ET).tz_convert('UTC')
            events=record['policies'][name]['events'];audits=record['policies'][name]['audits']
            assert pd.Timestamp(record['cutoff'])==cutoff
            times=[pd.Timestamp(e['at']) for e in events]
            for event,audit in zip(events,audits):
                at=pd.Timestamp(event['at']);clock=at.tz_convert(core.ET).strftime('%H:%M:%S')
                has_later=any(t>at for t in times);default='WAIT' if has_later else 'CASH'
                assert pd.Timestamp(event['cutoff'])==cutoff
                assert history.loc[at,'action']==event['action']
                if at not in states.index:
                    assert event['action']==default and not event['ready'];checks+=1;continue
                query=states.loc[at]
                past=states[(states.index<cutoff)&(states.release<cutoff)&states.clock.eq(clock)].tail(126)
                assert list(pd.to_datetime(audit['eligible'],utc=True))==list(past.index)
                if len(past)<80:
                    assert event['action']==default and not event['ready'];checks+=1;continue
                X=past[META].to_numpy(dtype=float);q=query[META].to_numpy(dtype=float)
                scale=np.std(X,axis=0,ddof=0);scale[scale==0]=1.
                d=np.sum(((X-q)/scale)**2,axis=1);idx=np.argsort(d,kind='stable')[:40]
                old=past.iloc[idx]
                assert list(pd.to_datetime(audit['neighbors'],utc=True))==list(old.index)
                np.testing.assert_allclose(audit['scale'],scale,atol=1e-13,rtol=1e-10)
                np.testing.assert_allclose(audit['center'],X.mean(axis=0),atol=1e-13,rtol=1e-10)
                mu=float(old.buy.mean());se=float(old.buy.std(ddof=1)/40**.5)
                w=float(old.wait.mean()) if settings['wait'] and has_later else 0.
                b=mu-settings['penalty']*se;v=max(0.,w)
                chosen=query.symbol if b>v else default
                assert event['action']==chosen and event['selected_symbol']==query.symbol and event['ready']
                for k,x in [('buy_mean',mu),('buy_se',se),('buy_value',b),('wait_mean',w),('wait_value',v)]:near(event[k],x)
                checks+=1;ready+=1;buy_count+=int(chosen in core.ETFS)
    primary=protocol['primary'];frozen=json.loads((run/'decisions_frozen.json').read_text())
    assert frozen['primary']==primary
    for e in frozen['decisions']:
        assert policies[primary].loc[pd.Timestamp(e['at']),'action']==e['action']
    market=core.load_market();results=json.loads((run/'results.json').read_text());trade_checks=0;books=[]
    names=[*protocol['policies'],'base__ensemble_log','breadth__ensemble_log','double_cost']
    for name in names:
        equity=pd.read_csv(run/(name+'_equity.csv'),parse_dates=['date']).set_index('date')
        try:trades=pd.read_csv(run/(name+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        if len(trades):
            trades['date']=pd.to_datetime(trades.entry_date).dt.normalize();assert not trades.date.duplicated().any()
        multiple=2 if name=='double_cost' else 1;fee=slip=.0005*multiple;capital=1.
        if name in protocol['policies']:actions=policies[name].action
        elif name=='double_cost':actions=policies[primary].action
        else:actions=source_forecasts[name.split('__')[0]].action
        for date,mark in equity.iterrows():
            row=trades[trades.date.eq(date)] if len(trades) else trades
            if len(row):
                t=row.iloc[0];at,start=pd.Timestamp(t.decision_at),pd.Timestamp(t.entry_at)
                assert actions.loc[at]==t.symbol
                earlier=actions[(actions.index.tz_convert(core.ET).date==date.date())&(actions.index<at)]
                assert not earlier.isin(core.ETFS).any(),'First eligible action must execute in this flat-close cash ledger'
                f=market.bars[t.symbol];candidate=f[f.date.eq(date)&f.session_type.eq('regular')&(f.start_at>at+pd.Timedelta(seconds=30))]
                first=candidate.iloc[0];assert first.start_at==start and start-at<=pd.Timedelta(minutes=6)
                entry=first.open*(1+slip)*first.share_factor
                target=math.ceil(first.open*(1+slip)*1.05*100-1e-10)/100*first.share_factor;stop=entry*.985
                close_at=pd.Timestamp(market.calendar[date]['close_at']);path=f[(f.start_at>=start)&(f.end_at<=close_at)&f.session_type.eq('regular')]
                found=None
                for bar in path.itertuples():
                    rs=math.floor(stop/bar.share_factor*100+1e-10)/100;rt=math.ceil(target/bar.share_factor*100-1e-10)/100
                    if bar.low<=rs:found=(bar.end_at,min(bar.open,rs)*bar.share_factor*(1-slip),'risk_stop');break
                    if bar.high>=rt:found=(bar.end_at,rt*bar.share_factor,'target');break
                if found is None:
                    last=path.iloc[-1];assert last.end_at==close_at
                    found=(close_at,last.econ_close*(1-slip),'session_close')
                assert pd.Timestamp(t.exit_at)==found[0] and t.reason==found[2]
                near(t.entry_fill_economic,entry);near(t.exit_fill_economic,found[1]);near(t.dividend_per_economic_unit,0.)
                net=found[1]*(1-fee)/(entry*(1+fee))-1;near(t.net_return,net);near(t.capital,capital)
                near(t.units_economic,capital/(entry*(1+fee)));capital*=1+net
                near(mark.cash_settled,0.);near(mark.cash_unsettled,capital);trade_checks+=1
            else:near(mark.cash_settled,capital);near(mark.cash_unsettled,0.)
            near(mark.equity,capital);near(mark.position_value,0.);near(mark.receivables,0.)
            assert mark.mark_complete and mark.symbol=='CASH'
        stats=results['double_cost'] if name=='double_cost' else results['policies'].get(name,results['controls'].get(name))
        prior=1.;monthly=[]
        for month,g in equity.groupby(equity.index.to_period('M')):
            rate=g.equity.iloc[-1]/prior-1;prior=g.equity.iloc[-1];monthly.append((str(month),rate))
            near(rate,next(x['return'] for x in stats['monthly'] if x['month']==str(month)))
        avg=np.mean([r for m,r in monthly if m in protocol['regression_months']])
        near(avg,stats['primary_average']);near(capital-1,stats['continuous_return'])
        assert stats['primary_pass']==(avg>=.1 and all(r>=-1e-12 for _,r in monthly))
        books.append({'name':name,'actual_trades':len(trades),'monthly_and_settlement_reconciled':True})
    out={'passed':True,'all_saved_policy_decisions':checks,'ready_neighborhood_decisions':ready,
         'counterfactual_clock_buys':buy_count,'mature_suffix_labels':suffixes,'negative_suffix_labels_retained':negative_suffixes,
         'raw_trade_checks':trade_checks,'capital_books':books,'frozen_primary_case_decisions':len(frozen['decisions']),
         'source_inputs_unchanged':True,'no_effect_pass_inferred':True}
    core.dump(run/'verification.json',out);print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)
