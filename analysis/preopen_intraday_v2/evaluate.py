"""Frozen time splits, three mechanisms, historical out-of-time diagnostics."""
from __future__ import annotations
import json, hashlib, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from scipy.special import logit

OUT=Path(__file__).resolve().parent
TARGET=.03
H=['h_range','h_mfe','h_ret','h_previous_ret','h_previous_range']
CONT=H+['pre_ret','pre_gap','pre_range','pre_eff','pre_fade','pre_early','pre_late','pre_rvol','pre_active','night_ret','night_gap','night_eff','night_fade','night_rvol','night_active','gap_scaled']
REPAIR=H+['pre_gap','pre_range','pre_rebound','pre_position','pre_trough','pre_early','pre_late','pre_rvol','pre_active','night_ret','night_rebound','night_position','night_trough','gap_scaled']
RELATIVE=H+['pre_ret','pre_gap','pre_range','night_ret','pre_rvol','relative_pre_ret','relative_pre_gap','relative_pre_late','relative_night_ret','qqq_pre_gap','qqq_pre_ret','qqq_night_ret','breadth','dispersion','available_pool']
FEATURES={'H':H,'continuation':CONT,'repair':REPAIR,'relative':RELATIVE}

def wilson(k,n):
    if not n:return [None,None]
    z=1.959964;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den
    width=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-width),float(center+width)]

def stats(g,selected):
    s=g[selected];n=len(s);k=int(s.y.sum());allhits=int(g.y.sum())
    rates=g.groupby('symbol').y.mean()
    matched=float(s.symbol.map(rates).mean()) if n else None
    tp=k;fp=n-k;fn=allhits-k;tn=len(g)-n-fn
    return dict(n=n,days=int(s.day.nunique()),months=int(s.day.str[:7].nunique()),tp=tp,fp=fp,fn=fn,tn=tn,
                precision=k/n if n else None,wilson=wilson(k,n),recall=k/allhits if allhits else None,
                fpr=fp/(fp+tn) if fp+tn else None,coverage=n/len(g) if len(g) else 0,
                all_base=float(g.y.mean()) if len(g) else None,matched_base=matched,
                lift=k/n-matched if n else None,
                mean_close=float(s.close_ret.mean()) if n else None,
                mean_proxy_net=float(s.cost_proxy.mean()) if n else None,
                mae_p10=float(s.mae.quantile(.1)) if n else None,mfe_median=float(s.mfe.median()) if n else None,
                delayed_precision=float((s.delayed_mfe>=TARGET).mean()) if n else None,
                targets={str(b):float((s.mfe>=b).mean()) for b in (.01,.02,.03,.05)} if n else {},
                close_up=float((s.close_ret>0).mean()) if n else None)

def block_ci(frame,values,denominator=None,seed=1003):
    if denominator is None:denominator=np.ones(len(frame))
    d=pd.DataFrame({'day':frame.day.to_numpy(),'v':values,'n':denominator}).groupby('day').sum().sort_index()
    if d.n.sum()<=0:return dict(estimate=None,ci=[None,None],p=1.,blocks=int(np.ceil(len(d)/5)),block_days=5)
    groups=[d.iloc[i:i+5].sum().to_numpy() for i in range(0,len(d),5)]
    a=np.array(groups);rng=np.random.default_rng(seed);res=[];null=[]
    observed=float(d.v.sum()/d.n.sum()) if d.n.sum()>0 else np.nan
    for _ in range(1000):
        v=a[rng.integers(0,len(a),len(a))].sum(axis=0)
        if v[1]>0:res.append(v[0]/v[1])
        null.append(float((a[:,0]*rng.choice([-1,1],len(a))).sum()/a[:,1].sum()))
    return dict(estimate=observed,ci=np.quantile(res,[.025,.975]).tolist() if res else [None,None],
                p=float((1+sum(v>=observed for v in null))/1001),blocks=len(a),block_days=5)

def predict(model,calibrator,x):
    raw=model.predict_proba(x)[:,1]
    return calibrator.predict_proba(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1))[:,1],raw

def main():
    frame=pd.read_parquet(OUT/'panel.parquet')
    # Common sample set for paired ablations. No label-driven exclusions.
    common=frame[frame.qqq_pre_gap.notna()&frame.qqq_night_ret.notna()].copy()
    common['y']=(common.mfe>=TARGET).astype(int)
    train=common[common.day<='2025-09-30'].copy()
    cal=common[(common.day>='2025-10-01')&(common.day<='2025-12-31')].copy()
    test=common[common.day>='2026-01-01'].copy()
    for f in (train,cal,test):
        assert (pd.to_datetime(f.max_source_available)<=pd.to_datetime(f.decision_at)).all()
        assert (pd.to_datetime(f.feature_end)<pd.to_datetime(f.day+' 09:30')).all()
    assert pd.to_datetime(train.label_end).max()<pd.Timestamp('2025-10-01')
    assert pd.to_datetime(cal.label_end).max()<pd.Timestamp('2026-01-01')
    if min(len(train),len(cal),len(test))<100:raise ValueError('Too few common time-split samples')
    report={'protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),
            'panel_sha256':hashlib.sha256((OUT/'panel.parquet').read_bytes()).hexdigest(),
            'counts':{'all_qualified':len(frame),'common':len(common),'qqq_missing':len(frame)-len(common)},
            'periods':{name:{'n':len(f),'first':f.day.min(),'last':f.day.max(),'symbols':f.symbol.nunique(),
                              'days':f.day.nunique(),'base':float(f.y.mean())} for name,f in [('train',train),('calibration',cal),('test',test)]},
            'routes':{},'status':'historical_development_not_independent'}
    rate=train.groupby('symbol').y.agg(['sum','count']);prior=float(train.y.mean())
    b0=((rate['sum']+prior*20)/(rate['count']+20)).to_dict()
    for f in (cal,test):f['score_B0']=f.symbol.map(b0).fillna(prior)
    modeldir=OUT/'models';modeldir.mkdir(exist_ok=True)
    for route,cols in FEATURES.items():
        model=LGBMClassifier(n_estimators=120,max_depth=3,num_leaves=7,min_child_samples=80,
                             learning_rate=.04,reg_lambda=15.,random_state=1003,n_jobs=1,verbosity=-1)
        model.fit(train[cols],train.y)
        for f in (cal,test):f['raw_'+route]=model.predict_proba(f[cols])[:,1]
        # Sigmoid and threshold selection use separate chronological blocks.
        fitcal=cal[cal.day<'2025-11-15'];selectcal=cal[cal.day>='2025-11-15'].copy()
        if fitcal.y.nunique()!=2:raise ValueError('Calibration split has one class')
        calibrator=LogisticRegression(C=1.,random_state=1003).fit(logit(np.clip(fitcal['raw_'+route],1e-5,1-1e-5)).to_numpy().reshape(-1,1),fitcal.y)
        for f in (cal,test):f['score_'+route],_=predict(model,calibrator,f[cols])
        amplitude=[]
        for quantile in (.2,.5,.8):
            reg=LGBMRegressor(objective='quantile',alpha=quantile,n_estimators=120,max_depth=3,num_leaves=7,
                min_child_samples=80,learning_rate=.04,reg_lambda=15.,random_state=1003,n_jobs=1,verbosity=-1)
            reg.fit(train[cols],train.mfe)
            for f in (cal,test):f[f'mfe_{route}_{int(quantile*100)}']=np.maximum(0,reg.predict(f[cols]))
            amplitude.append(reg)
        selectcal=cal[cal.day>='2025-11-15'].copy()
        candidates=[]
        for t in (.60,.65,.70,.75,.80):
            m=stats(selectcal,selectcal['score_'+route]>=t)
            m['threshold']=t;m['pass']=m['n']>=100 and m['days']>=30 and m['precision'] is not None and m['precision']>=.70 and m['lift']>=.03
            candidates.append(m)
        eligible=[m for m in candidates if m['pass']]
        threshold=max(eligible,key=lambda m:m['n'])['threshold'] if eligible else .70
        enrolled=[]
        for symbol,g in selectcal.groupby('symbol'):
            m=stats(g,g['score_'+route]>=threshold)
            if m['n']>=20 and m['precision']>=.70 and m['lift']>=.03:enrolled.append(symbol)
        selected=test['score_'+route]>=threshold
        summary=stats(test,selected)
        matched=test.symbol.map(test.groupby('symbol').y.mean()).to_numpy()
        liftci=block_ci(test,selected.to_numpy()*(test.y.to_numpy()-matched),selected.astype(int).to_numpy())
        brier=brier_score_loss(test.y,test['score_'+route]);h_brier=brier_score_loss(test.y,test['score_H']) if route!='H' else brier
        brierci=block_ci(test,(test.y-test['score_H']).to_numpy()**2-(test.y-test['score_'+route]).to_numpy()**2)
        symbols=[]
        for symbol,g in test.groupby('symbol'):
            m=stats(g,g['score_'+route]>=threshold)
            m.update(symbol=symbol,enrolled_in_calibration=symbol in enrolled)
            m['pass']=symbol in enrolled and m['n']>=30 and m['months']>=4 and m['precision']>=.70 and m['lift']>=.03
            symbols.append(m)
        months=[]
        for month,g in test.groupby(test.day.str[:7]):
            m=stats(g,g['score_'+route]>=threshold);m['month']=month;months.append(m)
        bins=[]
        for lo,hi in zip([0,.2,.4,.6,.7,.8,.9],[.2,.4,.6,.7,.8,.9,1.00001]):
            g=test[(test['score_'+route]>=lo)&(test['score_'+route]<hi)]
            bins.append({'lo':lo,'hi':min(hi,1),'n':len(g),'prediction':float(g['score_'+route].mean()) if len(g) else None,'observed':float(g.y.mean()) if len(g) else None})
        imp=sorted(zip(cols,model.feature_importances_.tolist()),key=lambda x:-x[1])
        report['routes'][route]=dict(features=cols,importance=imp,threshold=threshold,threshold_candidates=candidates,
            calibration_gate=bool(eligible),enrolled=enrolled,summary=summary,lift_block=liftci,brier=brier,h_brier=h_brier,
            brier_improvement=brierci,auc=float(roc_auc_score(test.y,test['score_'+route])),
            log_loss=float(log_loss(test.y,test['score_'+route])),symbols=symbols,months=months,reliability=bins,
            pass_pre_holm=bool(eligible) and summary['n']>=300 and summary['days']>=60 and summary['precision']>=.70 and liftci['ci'][0]>0 and brierci['ci'][0]>0)
        report['routes'][route]['amplitude']={'mae_median':float(np.abs(test[f'mfe_{route}_50']-test.mfe).mean()),
            'q20_q80_coverage':float(((test.mfe>=test[f'mfe_{route}_20'])&(test.mfe<=test[f'mfe_{route}_80'])).mean()),
            'quantile_crossing':int((test[f'mfe_{route}_20']>test[f'mfe_{route}_80']).sum()),
            'forecast_scope':'same-day MFE distribution, not executable exit price'}
        with (modeldir/(route+'.pkl')).open('wb') as f:pickle.dump({'model':model,'calibrator':calibrator,'amplitude_models':amplitude,'features':cols},f)
        print(json.dumps({'route':route,'threshold':threshold,'cal_gate':bool(eligible),'precision':summary['precision'],'n':summary['n'],
                          'lift':summary['lift'],'brier_improvement':brierci,'passed_stocks':[s['symbol'] for s in symbols if s['pass']]}),flush=True)
    routes=['continuation','repair','relative']
    pvals=[max(report['routes'][r]['brier_improvement']['p'],report['routes'][r]['lift_block']['p']) for r in routes]
    order=np.argsort(pvals);adjusted=np.empty(3);running=0.
    for i,j in enumerate(order):
        running=max(running,(3-i)*pvals[j]);adjusted[j]=min(1.,running)
    reject=adjusted<.05
    universe=json.loads((OUT/'universe.json').read_text());favorites={s for s,m in universe['members'].items() if '特别关注' in m['groups'] and m.get('stock_type')=='STOCK'}
    for r,p,rej in zip(routes,adjusted,reject):
        result=report['routes'][r];fg=test[test.symbol.isin(favorites)];result['favorite_summary']=stats(fg,fg['score_'+r]>=result['threshold']);result['holm_p']=float(p);result['pass']=result['pass_pre_holm'] and bool(rej)
        result['favorite_passed']=[s['symbol'] for s in result['symbols'] if s['pass'] and s['symbol'] in favorites]
        result['favorite_5_gate']=len(result['favorite_passed'])>=5
    test['signal_pre_up']=test.pre_ret>0
    report['pre_up_baseline']=stats(test,test.signal_pre_up)
    report['b0_brier']=float(brier_score_loss(test.y,test.score_B0))
    report['b0_within_auc']=None
    # Same-stock AUC prevents pooled identity/volatility sorting from passing as timing insight.
    for route in FEATURES:
        aucs=[];weights=[]
        for s,g in test.groupby('symbol'):
            if len(g)>=30 and g.y.nunique()==2:aucs.append(roc_auc_score(g.y,g['score_'+route]));weights.append(len(g))
        report['routes'][route]['within_stock_auc']=float(np.average(aucs,weights=weights)) if aucs else None
    pred=pd.concat([cal.assign(period='calibration'),test.assign(period='test')],ignore_index=True)
    pred.to_parquet(OUT/'predictions.parquet',index=False,compression='zstd',compression_level=7)
    report['predictions_sha256']=hashlib.sha256((OUT/'predictions.parquet').read_bytes()).hexdigest()
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
