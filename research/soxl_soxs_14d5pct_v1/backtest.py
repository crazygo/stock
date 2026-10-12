"""Causal daily-decision SOXL/SOXS/cash research. No order-routing code.

Run from the project root; imported inputs and new outputs live under .cache/.
The original model and account rules are preserved in original/backtest.py.
"""
from __future__ import annotations
import argparse, hashlib, json, math, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
from paths import CACHE_ROOT, RAW_DIR, writable_results
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
ETFS = ['SOXL', 'SOXS']
SYMBOLS = ETFS + ['SOXX', 'QQQ', 'NVDA', 'AMD', 'TLT']
FEE = 0.0005
SLIP = 0.0005

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def load_inputs():
    data, metadata, events, audit = {}, {}, {}, []
    for s in SYMBOLS:
        p = RAW_DIR / f'{s}_ibkr_daily.json'
        d = json.loads(p.read_text()); metadata[s] = d
        f = pd.DataFrame({k:d[k] for k in ['time','open','high','low','close','volume']})
        f.index = pd.DatetimeIndex(pd.to_datetime(f.pop('time'), utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize())
        if f.index.has_duplicates or not f.index.is_monotonic_increasing:
            raise ValueError(f'{s}: duplicate/unsorted timestamps')
        if not ((f.high >= f[['open','low']].max(axis=1)) & (f.low <= f[['open','high']].min(axis=1)) & (f.low>0) & (f.close>0)).all():
            raise ValueError(f'{s}: invalid OHLC')
        close_outside=f[(f.close>f.high)|(f.close<f.low)]
        # IBKR official daily closes can differ from the reported intraday range.
        # Preserve both raw fields, record every exception, never clamp or invent bars.
        discrepancy=np.maximum((f.close-f.high).clip(lower=0),(f.low-f.close).clip(lower=0))/f.close
        if discrepancy.max()>.01:raise ValueError(f'{s}: close/range discrepancy > 1%')
        f['factor'] = 1.0
        actions = d.get('corp_actions', [])
        for a in actions:
            if a['type']=='Splits':
                f.loc[f.index < pd.Timestamp(a['date']), 'factor'] /= float(a['value'])
        divs = []
        for a in actions:
            if a['type']=='CashDividends':
                ex = pd.Timestamp(a['date']); pay = pd.Timestamp(a['payment_date'])
                factor = float(f.loc[ex,'factor']) if ex in f.index else 1.0
                divs.append({'ex':ex,'pay':pay,'amount':float(a['value'])*factor})
        events[s] = divs
        data[s] = f
        audit.append({'symbol':s,'rows':len(f),'first':str(f.index.min().date()),'last':str(f.index.max().date()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source':d.get('source'),'adjustment':'observed split-adjusted OHLC; cash dividends handled separately','close_outside_range':close_outside.reset_index().to_dict('records'),'max_close_range_discrepancy':float(discrepancy.max())})
    idx = data[ETFS[0]].index
    for s in SYMBOLS:
        if not data[s].index.equals(idx):
            raise ValueError(f'{s}: dates differ; no silent inner join or missing-bar fill')
    cal = json.loads((RAW_DIR/'calendar.json').read_text())
    future = pd.DatetimeIndex([x['session_date'] for x in cal['sessions']])
    sessions = idx.union(future).sort_values()
    expected = future[(future>=idx[0]) & (future<=idx[-1])]
    if not expected.isin(idx).all(): raise ValueError('Missing official market sessions')
    return data,events,sessions,audit

def features(data):
    out = pd.DataFrame(index=data['SOXL'].index)
    for s,f in data.items():
        r = f.close.pct_change(fill_method=None)
        for lag in [1,5,20]: out[f'{s}_ret{lag}'] = f.close.pct_change(lag, fill_method=None)
        for lag in [10,20]: out[f'{s}_vol{lag}'] = r.rolling(lag).std()
        out[f'{s}_dd20'] = f.close/f.close.rolling(20).max()-1
        out[f'{s}_range'] = (f.high-f.low)/f.close
        out[f'{s}_relative_volume'] = f.volume/f.volume.rolling(20).mean()
    out['sector_relative_5'] = out.SOXX_ret5-out.QQQ_ret5
    out['sector_relative_20'] = out.SOXX_ret20-out.QQQ_ret20
    out['pair_participation_5'] = ((out.NVDA_ret5>0).astype(int)+(out.AMD_ret5>0).astype(int))/2
    out['pair_dispersion_5'] = abs(out.NVDA_ret5-out.AMD_ret5)
    # A row indexed by entry date uses ONLY the prior session's completed data.
    return out.shift(1).replace([np.inf,-np.inf],np.nan)

def deadline(entry, sessions):
    target = pd.Timestamp(entry)+pd.Timedelta(days=13)
    if target>sessions[-1]:return pd.NaT
    available=sessions[sessions<=target]
    return available[-1]

def entry_values(f,day,fee=FEE,slip=SLIP):
    price=float(f.loc[day,'open'])*(1+slip)
    factor=float(f.loc[day,'factor'])
    # The nominal sell limit rounds upward to the original market's cent tick.
    limit=math.ceil((price/factor)*1.05*100-1e-10)/100*factor
    return price,limit

def label_one(s,entry,data,events,sessions,fee=FEE,slip=SLIP):
    f=data[s]; end=deadline(entry,sessions)
    price,limit=entry_values(f,entry,fee,slip)
    if pd.isna(end):
        return {'entry':entry,'symbol':s,'deadline':pd.NaT,'exit':pd.NaT,'entry_fill':price,'limit':limit,'exit_fill':np.nan,'target_hit':np.nan,'net_return':np.nan,'reason':'pending_calendar','complete_window':False,'dividend_per_unit':np.nan,'holding_sessions':np.nan}
    path=f.loc[entry:min(end,f.index[-1])]
    hits=path.index[path.high>=limit]
    complete=bool(end<=f.index[-1])
    if len(hits): exit_day=hits[0]; exit_price=limit; reason='target'
    elif complete: exit_day=end;exit_price=float(f.loc[end,'close'])*(1-slip);reason='expiry'
    else: exit_day=f.index[-1];exit_price=float(f.loc[exit_day,'close']);reason='pending'
    div=sum(a['amount'] for a in events[s] if entry<a['ex']<=exit_day)
    net=(exit_price*(1-fee)+div)/(price*(1+fee))-1 if reason!='pending' else np.nan
    return {'entry':entry,'symbol':s,'deadline':end,'exit':exit_day,'entry_fill':price,'limit':limit,'exit_fill':exit_price,'target_hit':int(reason=='target'),'net_return':net,'reason':reason,'complete_window':complete,'dividend_per_unit':div,'holding_sessions':len(f.loc[entry:exit_day])}

def make_labels(data,events,sessions):
    return {s:pd.DataFrame([label_one(s,d,data,events,sessions) for d in data[s].index]).set_index('entry') for s in ETFS}

def replay(data,events,sessions,actions,start,end,fee=FEE,slip=SLIP,allow_entries_until=None,skip_entry_high=False):
    """One capital ledger; no duplicate capital across labels or concurrent positions.

    Dividends accrue on ex-date only if held overnight; unpaid dividends remain
    receivables, and become spendable only on the payment date. Liquidation
    proceeds become reusable at the next session, not on the sale date.
    """
    idx=data['SOXL'].loc[start:end].index
    cash=1.0; pos=None; receivables=[]; trades=[]; marks=[]; decisions=[]
    for day in idx:
        paid=[x for x in receivables if x['pay']<=day]
        cash+=sum(x['amount'] for x in paid)
        receivables=[x for x in receivables if x['pay']>day]
        was_held=pos is not None
        if pos:
            for a in events[pos['symbol']]:
                if a['ex']==day:
                    amount=pos['units']*a['amount'];pos['dividends']+=amount
                    if a['pay']<=day:cash+=amount
                    else:receivables.append({'pay':a['pay'],'amount':amount})
        action=actions.get(day,'CASH')
        decision='hold' if pos else 'cash'
        if not pos and action in ETFS and (allow_entries_until is None or day<=pd.Timestamp(allow_entries_until)):
            if pd.isna(deadline(day,sessions)):raise ValueError('Future official calendar incomplete; cannot open a modeled trade')
            price,limit=entry_values(data[action],day,fee,slip)
            pos={'symbol':action,'entry':day,'deadline':deadline(day,sessions),'entry_fill':price,'limit':limit,'capital':cash,'units':cash/(price*(1+fee)),'dividends':0.0,'equity_at_entry':cash+sum(x['amount'] for x in receivables)}
            cash=0.0;decision='buy_'+action
        exit_reason=None
        if pos:
            f=data[pos['symbol']];row=f.loc[day]
            # One profit barrier and fixed expiry: no OHLC high/low path ordering guess.
            if row.high>=pos['limit'] and not(skip_entry_high and pos['entry']==day):
                exit_price=pos['limit'];exit_reason='target'
            elif day>=pos['deadline']:
                exit_price=float(row.close)*(1-slip);exit_reason='expiry'
            if exit_reason:
                proceeds=pos['units']*exit_price*(1-fee);cash+=proceeds
                trades.append({**pos,'exit':day,'exit_fill':exit_price,'reason':exit_reason,'net_return':(proceeds+pos['dividends'])/pos['capital']-1,'holding_sessions':len(f.loc[pos['entry']:day])})
                pos=None
        position_value=0 if pos is None else pos['units']*float(data[pos['symbol']].loc[day,'close'])
        recv=sum(x['amount'] for x in receivables)
        marks.append({'date':day,'equity':cash+position_value+recv,'cash':cash,'position_value':position_value,'receivables':recv,'symbol':pos['symbol'] if pos else 'CASH'})
        decisions.append({'date':day,'requested_action':action,'decision':decision,'exit_reason':exit_reason,'already_held_at_open':was_held})
    eq=pd.DataFrame(marks).set_index('date')
    if pos:
        trades.append({**pos,'exit':pd.NaT,'exit_fill':np.nan,'reason':'pending','net_return':np.nan,'holding_sessions':len(data[pos['symbol']].loc[pos['entry']:idx[-1]])})
    return eq,pd.DataFrame(trades),pd.DataFrame(decisions)

def statistics(eq,trades):
    if len(eq)==0:return {}
    done=trades[trades.reason!='pending'] if len(trades) else trades
    r=done.net_return if len(done) else pd.Series(dtype=float)
    curve=np.r_[1.0,eq.equity.to_numpy()]
    monthly=[];previous=1.0
    for month,g in eq.groupby(eq.index.to_period('M')):
        entry=trades[trades.entry.dt.to_period('M')==month] if len(trades) else trades
        closed=done[done.exit.dt.to_period('M')==month] if len(done) else done
        end_equity=float(g.equity.iloc[-1]);path=np.r_[previous,g.equity.to_numpy()]
        monthly.append({'month':str(month),'return':end_equity/previous-1,'month_max_drawdown_close':float((path/np.maximum.accumulate(path)-1).min()),'entries':len(entry),'closed':len(closed),'target_exits':int((closed.reason=='target').sum()) if len(closed) else 0,'expiry_losses':int(((closed.reason=='expiry') & (closed.net_return<0)).sum()) if len(closed) else 0,'expiry_gains':int(((closed.reason=='expiry') & (closed.net_return>=0)).sum()) if len(closed) else 0,'end_position':g.symbol.iloc[-1]})
        previous=end_equity
    return {'return':float(eq.equity.iloc[-1]-1),'max_drawdown_close':float((curve/np.maximum.accumulate(curve)-1).min()),'entries':len(trades),'closed':len(done),'pending':len(trades)-len(done),'target_hit_rate':float((done.reason=='target').mean()) if len(done) else None,'mean_trade_return':float(r.mean()) if len(r) else None,'worst_trade':float(r.min()) if len(r) else None,'mean_losing_trade':float(r[r<0].mean()) if (r<0).any() else None,'mean_holding_sessions':float(done.holding_sessions.mean()) if len(done) else None,'monthly':monthly}

def fit_models(X,labels,train_dates):
    models={}
    for s in ETFS:
        y=labels[s].loc[train_dates];x=X.loc[train_dates]
        classifier=make_pipeline(StandardScaler(),LogisticRegression(C=.2,max_iter=2000,random_state=814))
        classifier.fit(x,y.target_hit)
        def forest():return ExtraTreesRegressor(n_estimators=160,min_samples_leaf=16,max_depth=5,random_state=814,n_jobs=2,max_features=.8)
        payoff=forest().fit(x,y.net_return)
        miss=y.target_hit==0
        conditional=forest().fit(x.loc[miss],y.loc[miss,'net_return']) if miss.sum()>=32 else None
        models[s]={'p':classifier,'payoff':payoff,'miss':conditional,'miss_mean':float(y.loc[miss,'net_return'].mean()),'win_mean':float(y.loc[~miss,'net_return'].mean()),'train_base_rate':float(y.target_hit.mean())}
    return models

def predict(models,X):
    out={}
    for s,m in models.items():
        p=m['p'].predict_proba(X)[:,1]
        failure=m['miss'].predict(X) if m['miss'] is not None else np.full(len(X),m['miss_mean'])
        out[f'{s}_p']=p
        out[f'{s}_conditional_ev']=p*m['win_mean']+(1-p)*failure
        out[f'{s}_direct_ev']=m['payoff'].predict(X)
        out[f'{s}_miss_payoff']=failure
        out[f'{s}_base_p']=np.full(len(X),m['train_base_rate'])
    return pd.DataFrame(out,index=X.index)

def policy(pred,route,threshold):
    a=pred[f'SOXL_{route}_ev'];b=pred[f'SOXS_{route}_ev']
    ans=pd.Series(np.where(a>=b,'SOXL','SOXS'),index=pred.index)
    ans[np.maximum(a,b)<=threshold]='CASH'
    return ans.to_dict()

def run(output):
    out=writable_results(output);out.mkdir(parents=True,exist_ok=True)
    data,events,sessions,audit=load_inputs();X=features(data);labels=make_labels(data,events,sessions)
    valid=X.index[X.notna().all(axis=1)]
    X=X.loc[valid]
    dump(out/'data_audit.json',audit)
    X.to_csv(out/'features.csv',index_label='entry_date')
    for s in ETFS:labels[s].to_csv(out/f'labels_{s}.csv')
    actions={};preds=[];registrations=[];trials=[];frozen={}
    for month in pd.period_range('2026-05','2026-09',freq='M'):
        start=month.start_time
        eligible=valid[(labels['SOXL'].loc[valid,'deadline']<start)&(labels['SOXL'].loc[valid,'complete_window'])]
        selection_dates=eligible[-50:]
        train_dates=eligible[labels['SOXL'].loc[eligible,'deadline']<selection_dates[0]]
        if len(train_dates)<100:raise ValueError('insufficient training history')
        models=fit_models(X,labels,train_dates)
        valpred=predict(models,X.loc[selection_dates]);candidates=[]
        for route in ['conditional','direct']:
            for threshold in [0,.005,.01,.02]:
                act=policy(valpred,route,threshold)
                end=labels['SOXL'].loc[selection_dates,'deadline'].max()
                eq,ts,_=replay(data,events,sessions,act,selection_dates[0],end)
                st=statistics(eq,ts)
                item={'month':str(month),'route':route,'threshold':threshold,'selection_return':st['return'],'selection_closed':st['closed'],'selection_mdd':st['max_drawdown_close']}
                trials.append(item)
                if st['closed']>=3 and st['return']>0:candidates.append(item)
        chosen=sorted(candidates,key=lambda x:(x['selection_return'],-x['selection_closed'],x['threshold']),reverse=True)[0] if candidates else {'route':'cash','threshold':None,'selection_return':0}
        test_dates=valid[valid.to_period('M')==month]
        testpred=predict(models,X.loc[test_dates]);testpred['month']=str(month)
        act=policy(testpred,chosen['route'],chosen['threshold']) if chosen['route']!='cash' else dict.fromkeys(test_dates,'CASH')
        actions.update(act);testpred['action']=[act[d] for d in test_dates];preds.append(testpred)
        reg={'month':str(month),'train_start':train_dates[0],'train_last_entry':train_dates[-1],'train_last_deadline':labels['SOXL'].loc[train_dates,'deadline'].max(),'selection_start':selection_dates[0],'selection_last_entry':selection_dates[-1],'selection_last_deadline':labels['SOXL'].loc[selection_dates,'deadline'].max(),'n_train_each_etf':len(train_dates),'n_selection_dates':len(selection_dates),'chosen':chosen,'n_policies_tried':8,'features':list(X.columns)}
        assert reg['train_last_deadline']<reg['selection_start']
        assert reg['selection_last_deadline']<start
        registrations.append(reg);frozen[str(month)]=models
        print(str(month),chosen,'train',len(train_dates),flush=True)
    pred=pd.concat(preds);pred.to_csv(out/'predictions.csv',index_label='entry_date')
    dump(out/'model_registration.json',registrations);dump(out/'all_policy_trials.json',trials)
    with (out/'frozen_models.pkl').open('wb') as f:pickle.dump(frozen,f)
    variants={'model':actions,'cash':{},'always_SOXL':dict.fromkeys(pred.index,'SOXL'),'always_SOXS':dict.fromkeys(pred.index,'SOXS'),'trend20':dict(zip(pred.index,np.where(X.loc[pred.index,'SOXX_ret20']>0,'SOXL','SOXS')))}
    results={};all_eq={}
    for name,act in variants.items():
        eq,ts,dec=replay(data,events,sessions,act,'2026-05-01','2026-09-30')
        results[name]=statistics(eq,ts);all_eq[name]=eq
        eq.to_csv(out/f'{name}_equity.csv');ts.to_csv(out/f'{name}_trades.csv',index=False);dec.to_csv(out/f'{name}_decisions.csv',index=False)
    # After September no new entries: finish September positions where observed.
    eq,ts,_=replay(data,events,sessions,actions,'2026-05-01',data['SOXL'].index[-1],allow_entries_until='2026-09-30')
    ts.to_csv(out/'model_trades_followup.csv',index=False)
    results['followup']=statistics(eq,ts)
    # Independent monthly starts are secondary diagnostics, never silently replace continuity.
    results['month_reset_diagnostic']={}
    for month in ['2026-05','2026-07','2026-09']:
        dates=pred.index[pred.index.to_period('M')==pd.Period(month)]
        ma={d:actions[d] for d in dates}
        meq,mt,_=replay(data,events,sessions,ma,dates[0],dates[-1]);results['month_reset_diagnostic'][month]=statistics(meq,mt)
    results['double_cost'] = statistics(*replay(data,events,sessions,actions,'2026-05-01','2026-09-30',fee=FEE*2,slip=SLIP*2)[:2])
    results['ignore_entry_day_target_stress'] = statistics(*replay(data,events,sessions,actions,'2026-05-01','2026-09-30',skip_entry_high=True)[:2])
    calibration=[]
    for s in ETFS:
        y=labels[s].loc[pred.index];sel=y.complete_window
        yt=y.loc[sel,'target_hit'];pr=pred.loc[sel,f'{s}_p']
        calibration.append({'symbol':s,'candidate_days':len(yt),'brier':brier_score_loss(yt,pr),'baseline_brier':brier_score_loss(yt,pred.loc[sel,f'{s}_base_p']),'log_loss':log_loss(yt,pr),'realized_hit_rate':float(yt.mean()),'mean_predicted_hit_rate':float(pr.mean()),'warning':'overlapping horizons, not independent trades; classifier probabilities not production-calibrated'})
        buckets=pd.DataFrame({'p':pr,'hit':yt});buckets['bucket']=pd.cut(buckets.p,[0,.2,.4,.6,.8,1],include_lowest=True)
        buckets.groupby('bucket',observed=False).agg(n=('hit','size'),predicted=('p','mean'),observed=('hit','mean')).to_csv(out/f'calibration_{s}.csv')
    results['calibration']=calibration
    results['all_policy_requested_months']={}
    for name in variants:
        vv=[r['return'] for r in results[name]['monthly'] if r['month'] in ['2026-05','2026-07','2026-09']]
        results['all_policy_requested_months'][name]={'arithmetic_average':float(np.mean(vv)),'geometric_average':float(np.prod(1+np.array(vv))**(1/3)-1),'arithmetic_10pct_pass':bool(np.mean(vv)>=.1),'every_month_10pct_pass':bool(all(v>=.1 for v in vv)),'is_expected_return_established':False}
    selected=[r for r in results['model']['monthly'] if r['month'] in ['2026-05','2026-07','2026-09']]
    avg=float(np.mean([r['return'] for r in selected]));geom=float(np.prod([1+r['return'] for r in selected])**(1/3)-1)
    results['acceptance']={'numerical_each_month_pass':all(r['return']>=.1 for r in selected),'arithmetic_average':avg,'geometric_average':geom,'average_10pct_pass':avg>=.1,'full_requested_system_pass':False,'full_gate_blockers':['5-minute/news data not fitted','historical development, not untouched prospective proof'],'requested_months':selected}
    # Block-resampling is descriptive only: too few months for an expected-monthly-return claim.
    results['expected_monthly_return_status']='NOT_ESTABLISHED: three selected realized months do not establish a repeatable expectation'
    dump(out/'results.json',results)
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,5))
        for name,eq in all_eq.items():ax.plot(eq.index,eq.equity,label=name,linewidth=2 if name=='model' else 1)
        ax.axhline(1,color='#777',lw=.6);ax.set(ylabel='Equity / starting capital',title='SOXL / SOXS: one-position ledger, net of assumed costs');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'equity.png',dpi=170);plt.close(fig)
    except ImportError:pass
    print(json.dumps(results['acceptance'],indent=2),flush=True)
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, default=RAW_DIR)
    parser.add_argument('--output', type=Path, default=CACHE_ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args=parser.parse_args();RAW_DIR=args.raw_dir.resolve();run(args.output)
