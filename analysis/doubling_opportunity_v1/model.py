"""Purged monthly training, calibration and full retained forward predictions."""
from __future__ import annotations
import hashlib,json,platform
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from data import FEATURES,CACHE

ALGORITHMS={'logistic':'LogisticRegression','hist_gbdt':'HistGradientBoostingClassifier'}


def fit_month(d,month,h,cfg):
    first=pd.Timestamp(month+'-01')
    calstart=first-pd.Timedelta(days=120);calend=first-pd.Timedelta(days=60)
    calmid=calstart+pd.Timedelta(days=40)
    # Strictly mature before the next block; no current partial outcome enters fitting.
    train=d[d.eligible&(d.date<calstart)&(d[f'mature_at{h}']<calstart)&d[f'y{h}'].notna()]
    cal=d[d.eligible&(d.date>=calstart)&(d.date<calend)&(d[f'mature_at{h}']<first)&d[f'y{h}'].notna()]
    fitcal=cal[cal.date<calmid];choose=cal[cal.date>=calmid]
    audit={'month':month,'horizon':h,'train_rows':len(train),'train_positives':int(train[f'y{h}'].sum()),
           'calibration_rows':len(fitcal),'calibration_positives':int(fitcal[f'y{h}'].sum()),
           'selection_rows':len(choose),'selection_positives':int(choose[f'y{h}'].sum()),
           'calibration_start':str(calstart.date()),'calibration_fit_end_exclusive':str(calmid.date()),
           'selection_end_exclusive':str(calend.date()),'evaluation_start':str(first.date()),
           'train_latest_label_maturity':str(train[f'mature_at{h}'].max()) if len(train) else None,
           'cal_latest_label_maturity':str(cal[f'mature_at{h}'].max()) if len(cal) else None,
           'algorithms':{}}
    if len(train)<500 or train[f'y{h}'].nunique()!=2 or len(fitcal)<cfg['minimum_calibration_samples'] or fitcal[f'y{h}'].sum()<cfg['minimum_calibration_positives'] or len(choose)<200:
        audit['status']='insufficient_purged_training_or_calibration';return {},audit
    models={}
    x=train[FEATURES].to_numpy();y=train[f'y{h}'].astype(int).to_numpy()
    for name in ALGORITHMS:
        model=(make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=500,random_state=cfg['seed'])) if name=='logistic' else
               HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=2,random_state=cfg['seed']))
        model.fit(x,y)
        raw=model.predict_proba(fitcal[FEATURES].to_numpy())[:,1]
        iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(raw,fitcal[f'y{h}'])
        selected=iso.predict(model.predict_proba(choose[FEATURES].to_numpy())[:,1])
        brier=float(brier_score_loss(choose[f'y{h}'],selected))
        models[name]={'model':model,'calibrator':iso,'audit':audit}
        audit['algorithms'][name]={'algorithm':ALGORITHMS[name],'selection_brier':brier,
             'selection_baseline_brier':float(brier_score_loss(choose[f'y{h}'],np.full(len(choose),fitcal[f'y{h}'].mean()))),
             'calibration_base_rate':float(fitcal[f'y{h}'].mean()),'selection_base_rate':float(choose[f'y{h}'].mean()),
             'selection_p_gt80_count':int((selected>cfg['threshold']).sum())}
    audit['champion']=min(models,key=lambda n:audit['algorithms'][n]['selection_brier'])
    audit['status']='development_calibrated'
    for m in models.values():m['audit']=audit
    return models,audit


def predict(models,rows):
    return {n:m['calibrator'].predict(m['model'].predict_proba(rows[FEATURES].to_numpy())[:,1]) for n,m in models.items()}


def forward_backtest(d,cfg,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    results=[];audits=[]
    for month in cfg['evaluation_months']:
        target=d[d.eligible & (d.date.dt.strftime('%Y-%m')==month)].copy()
        for h in cfg['horizons']:
            print('fit',month,h,flush=True)
            models,audit=fit_month(d,month,h,cfg);audits.append(audit)
            if not models:continue
            predictions=predict(models,target)
            for n,p in predictions.items():
                q=target[['ticker','date','close',f'y{h}',f'mature_at{h}',f'first_touch{h}',f'end_close_double{h}',f'label_status{h}']].copy()
                q.columns=['ticker','date','reference_close','label','mature_at','first_touch','end_close_double','label_status']
                q['horizon']=h;q['algorithm']=n;q['probability']=p;q['month']=month;q['selected_algorithm']=n==audit['champion']
                results.append(q)
    (outdir/'split_audit.json').write_text(json.dumps(audits,indent=2))
    if not results:raise ValueError('No evaluable purged fold; inspect split_audit.json')
    allrows=pd.concat(results,ignore_index=True)
    allrows.to_parquet(CACHE/'forward_predictions.parquet',compression='zstd',compression_level=7,index=False)
    return allrows,audits


def wilson(k,n):
    if not n:return [None,None]
    z=1.95996398454;p=k/n;a=1+z*z/n
    mid=(p+z*z/(2*n))/a;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/a
    return [float(mid-half),float(mid+half)]


def cluster_ci(q,group,seed):
    known=q[q.label.notna()].copy()
    if len(known)==0:return [None,None]
    sums=known.groupby(group).label.agg(['sum','count']).to_numpy()
    if len(sums)<5:return [None,None]
    rng=np.random.default_rng(seed);draw=sums[rng.integers(0,len(sums),size=(1000,len(sums)))].sum(axis=1)
    rates=draw[:,0]/draw[:,1]
    return [float(x) for x in np.quantile(rates,[.025,.975])]


def signals(pred,cfg):
    accepted=[]
    for (h,n),q in pred.groupby(['horizon','algorithm']):
        last={}
        for r in q.sort_values(['date','probability','ticker'],ascending=[True,False,True]).itertuples():
            if r.probability<=cfg['threshold']:continue
            if r.ticker in last and (r.date-last[r.ticker]).days<cfg['signal_cooldown_calendar_days']:continue
            last[r.ticker]=r.date;accepted.append(r._asdict())
    return pd.DataFrame(accepted).drop(columns='Index',errors='ignore') if accepted else pd.DataFrame(columns=pred.columns)


def evaluate(pred,cfg,outdir):
    sig=signals(pred,cfg); sig.to_csv(outdir/'backtest_signals.csv',index=False)
    # Evaluate both fixed algorithms and the calibration-selected monthly policy separately.
    tables=[]
    for h in cfg['horizons']:
        pools=[(n,pred[(pred.horizon==h)&(pred.algorithm==n)],sig[(sig.horizon==h)&(sig.algorithm==n)]) for n in ALGORITHMS]
        policy=pred[(pred.horizon==h)&pred.selected_algorithm].copy();policy['algorithm']='monthly_selected'
        # Recompute cooldown across months even when the selected algorithm changes.
        policy_sig=signals(policy,cfg)
        pools.append(('monthly_selected',policy,policy_sig))
        for name,pool,q in pools:
            known=q[q.label.notna()].copy();n=len(known);tp=int(known.label.sum());fp=n-tp
            if len(q):q=q.copy();q['week']=q.date.dt.to_period('W-SUN').astype(str)
            stocks=sorted(q.ticker.unique()) if len(q) else []
            base=pool[pool.ticker.isin(stocks)&pool.label.notna()]
            base_by_stock=base.groupby('ticker').label.mean()
            baseline=float(known.ticker.map(base_by_stock).mean()) if n else None
            ci=wilson(tp,n)
            stock_ci=cluster_ci(q,'ticker',cfg['seed']) if len(q) else [None,None]
            week_ci=cluster_ci(q,'week',cfg['seed']) if len(q) else [None,None]
            scored=pool[pool.label.notna()]
            reliability=[]
            for lo,hi in [(0,.05),(.05,.1),(.1,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.000001)]:
                b=scored[(scored.probability>=lo)&(scored.probability<hi)]
                reliability.append({'low':lo,'high':hi,'rows':len(b),'predicted':float(b.probability.mean()) if len(b) else None,'observed':float(b.label.mean()) if len(b) else None})
            precision=tp/n if n else None
            passed=bool(n>=30 and len(stocks)>=10 and q.date.nunique()>=10 and ci[0]>.8 and stock_ci[0] is not None and stock_ci[0]>.8 and week_ci[0] is not None and week_ci[0]>.8 and baseline is not None and precision>baseline)
            tables.append({'horizon':h,'algorithm':name,'algorithm_name':ALGORITHMS.get(name,'monthly calibration-selected policy'),
                'signals':len(q),'mature_signals':n,'tp':tp,'fp':fp,'unknown':len(q)-n,'precision':precision,
                'conservative_lower_bound':tp/len(q) if len(q) else None,'wilson95':ci,'stock_cluster95':stock_ci,'week_cluster95':week_ci,
                'stock_count':len(stocks),'decision_dates':int(q.date.nunique()) if len(q) else 0,'same_stock_base_rate':baseline,
                'increment':precision-baseline if precision is not None and baseline is not None else None,
                'recall_of_daily_positive_cases':tp/int(scored.label.sum()) if scored.label.sum() else None,
                'daily_rows_not_independent':len(pool),'brier':float(brier_score_loss(scored.label,scored.probability)) if len(scored) else None,
                'reliability':reliability,'historical_numeric_gate':passed,'deployment_qualified':False,
                'blocking_evidence':['current_survivor_directory_not_PIT','no_new_independent_observation','corporate_action_vintages_not_reconciled','joint_quality_and_catalyst_not_backtested']})
            if len(q):q.to_csv(outdir/f'signals_{name}_{h}.csv',index=False)
    (outdir/'backtest_summary.json').write_text(json.dumps(tables,indent=2))
    return tables
