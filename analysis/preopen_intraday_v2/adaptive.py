"""Monthly causal retraining: no threshold/hyperparameter search."""
from __future__ import annotations
import hashlib,json,pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
from sklearn.metrics import brier_score_loss,roc_auc_score,log_loss
import evaluate as ev
OUT=ev.OUT

def historical_rate(daily):
    return (daily.h/daily.o-1>=.03).astype(float).rolling(60,min_periods=40).mean().shift(1)

def enrich(frame):
    # Current rows are already label-qualified; historical denominator uses the
    # complete daily chart archive, not only days with liquid overnight trading.
    charts=json.loads((OUT/'charts.json').read_text());parts=[]
    for symbol,g in frame.groupby('symbol'):
        d=pd.DataFrame(charts[symbol]['daily'],columns=['day','o','h','l','c','v']).set_index('day')
        d['base']=historical_rate(d)
        p=g.copy();p['h_base3']=p.day.map(d.base);parts.append(p)
    f=pd.concat(parts,ignore_index=True).sort_values(['day','symbol']).reset_index(drop=True)
    for c in ['pre_range','pre_late','night_ret','relative_night_ret']:
        f[c+'_scaled']=f[c]/f.h_range.clip(lower=.001)
    return f

def main():
    panel=pd.read_parquet(OUT/'panel.parquet')
    frame=enrich(panel[panel.qqq_pre_gap.notna()&panel.qqq_night_ret.notna()].copy())
    frame['y']=(frame.mfe>=.03).astype(int)
    features={r:cols+['h_base3']+([] if r=='H' else ['pre_range_scaled','pre_late_scaled','night_ret_scaled'])+(['relative_night_ret_scaled'] if r=='relative' else []) for r,cols in ev.FEATURES.items()}
    test=frame[frame.day>='2026-01-01'].copy();test['fold']=test.day.str[:7]
    folds=[];predicted=[];modeldir=OUT/'models_adaptive';modeldir.mkdir(exist_ok=True)
    for fold in sorted(test.fold.unique()):
        prior=frame[frame.day<fold+'-01'];days=sorted(prior.day.unique())[-252:]
        if len(days)<100:continue
        train=prior[prior.day.isin(days[:-64])];fitcal=prior[prior.day.isin(days[-64:-32])];selectcal=prior[prior.day.isin(days[-32:])]
        target=test[test.fold==fold].copy()
        if len(train)<500 or min(len(fitcal),len(selectcal))<100:continue
        assert pd.to_datetime(prior.label_end).max()<pd.Timestamp(fold+'-01')
        rate=train.groupby('symbol').y.agg(['sum','count']);base=float(train.y.mean());base_map=((rate['sum']+base*20)/(rate['count']+20)).to_dict()
        target['score_B0']=target.symbol.map(base_map).fillna(base)
        info={'month':fold,'train_first':train.day.min(),'train_end':train.day.max(),'sigmoid_end':fitcal.day.max(),
              'selection_end':selectcal.day.max(),'n_train':len(train),'n_sigmoid':len(fitcal),'n_selection':len(selectcal),'n_test':len(target),'routes':{}}
        for route,cols in features.items():
            model=LGBMClassifier(n_estimators=120,max_depth=3,num_leaves=7,min_child_samples=80,learning_rate=.04,reg_lambda=15.,random_state=1003,n_jobs=1,verbosity=-1)
            model.fit(train[cols],train.y)
            raw=model.predict_proba(fitcal[cols])[:,1]
            calibrator=LogisticRegression(C=1.,random_state=1003).fit(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1),fitcal.y)
            selection=selectcal.copy();selection['score_'+route],_=ev.predict(model,calibrator,selection[cols])
            target['score_'+route],target['raw_'+route]=ev.predict(model,calibrator,target[cols])
            calstats=ev.stats(selection,selection['score_'+route]>=.70)
            gate=calstats['n']>=50 and calstats['days']>=10 and calstats['precision']>=.70 and calstats['lift']>=.03
            enrolled=[]
            for symbol,g in selection.groupby('symbol'):
                m=ev.stats(g,g['score_'+route]>=.70)
                if m['n']>=10 and m['precision']>=.70 and m['lift']>=.03:enrolled.append(symbol)
            target['enrolled_'+route]=target.symbol.isin(enrolled)
            target['calibration_gate_'+route]=gate
            amp=[]
            for q in (.2,.5,.8):
                reg=LGBMRegressor(objective='quantile',alpha=q,n_estimators=120,max_depth=3,num_leaves=7,min_child_samples=80,learning_rate=.04,reg_lambda=15.,random_state=1003,n_jobs=1,verbosity=-1)
                reg.fit(train[cols],train.mfe);target[f'mfe_{route}_{int(q*100)}']=np.maximum(0,reg.predict(target[cols]));amp.append(reg)
            info['routes'][route]={'calibration':calstats,'gate':gate,'enrolled':enrolled,'importance':sorted(zip(cols,model.feature_importances_.tolist()),key=lambda x:-x[1])}
            with (modeldir/f'{fold}_{route}.pkl').open('wb') as f:pickle.dump({'model':model,'calibrator':calibrator,'features':cols,'amplitude_models':amp},f)
        target['period']='test';target['train_end']=train.day.max();target['calibration_end']=selectcal.day.max();predicted.append(target);folds.append(info)
        print(json.dumps({'month':fold,'n':len(target),'train_end':train.day.max()}),flush=True)
    test=pd.concat(predicted,ignore_index=True)
    report={'protocol_sha256':hashlib.sha256((OUT/'ADAPTIVE_PROTOCOL.md').read_bytes()).hexdigest(),
        'panel_sha256':hashlib.sha256((OUT/'panel.parquet').read_bytes()).hexdigest(), 'version':'monthly_v21','target':.03,'folds':folds,
        'counts':{'all_qualified':len(panel),'common':len(frame),'qqq_missing':len(panel)-len(frame)},
        'periods':{'train':{'first':frame.day.min(),'last':'rolling'},'calibration':{'first':'rolling','last':'prior month'},
                   'test':{'n':len(test),'first':test.day.min(),'last':test.day.max(),'symbols':test.symbol.nunique(),'days':test.day.nunique(),'base':float(test.y.mean())}},
        'routes':{},'status':'historical_development_not_independent'}
    universe=json.loads((OUT/'universe.json').read_text());favorites={s for s,m in universe['members'].items() if '特别关注' in m['groups'] and m.get('stock_type')=='STOCK'}
    for route,cols in features.items():
        selected=test['score_'+route]>=.70;summary=ev.stats(test,selected);matched=test.symbol.map(test.groupby('symbol').y.mean()).to_numpy()
        liftci=ev.block_ci(test,selected.to_numpy()*(test.y.to_numpy()-matched),selected.astype(int).to_numpy())
        brierci=ev.block_ci(test,(test.y-test.score_H).to_numpy()**2-(test.y-test['score_'+route]).to_numpy()**2)
        symbols=[];months=[];bins=[];aucs=[];weights=[]
        for symbol,g in test.groupby('symbol'):
            m=ev.stats(g,g['score_'+route]>=.70);enrolled=bool(g['enrolled_'+route].any())
            m.update(symbol=symbol,enrolled_in_calibration=enrolled,signals_enrolled_beforehand=int(((g['score_'+route]>=.7)&g['enrolled_'+route]).sum()))
            # Require all counted signals to have prior enrollment for the strict five-stock claim.
            m['pass']=enrolled and m['n']>=30 and m['signals_enrolled_beforehand']==m['n'] and m['months']>=4 and m['precision']>=.7 and m['lift']>=.03
            symbols.append(m)
            if len(g)>=30 and g.y.nunique()==2:aucs.append(roc_auc_score(g.y,g['score_'+route]));weights.append(len(g))
        for month,g in test.groupby('fold'):
            m=ev.stats(g,g['score_'+route]>=.7);m['month']=month;months.append(m)
        for lo,hi in zip([0,.2,.4,.6,.7,.8,.9],[.2,.4,.6,.7,.8,.9,1.00001]):
            g=test[(test['score_'+route]>=lo)&(test['score_'+route]<hi)]
            bins.append({'lo':lo,'hi':min(hi,1),'n':len(g),'prediction':float(g['score_'+route].mean()) if len(g) else None,'observed':float(g.y.mean()) if len(g) else None})
        gate=all(x['routes'][route]['gate'] for x in folds)
        report['routes'][route]=dict(features=cols,importance=folds[-1]['routes'][route]['importance'],threshold=.70,threshold_candidates=[],
             calibration_gate=gate,calibration_passed_months=sum(x['routes'][route]['gate'] for x in folds),enrolled=sorted(test[test['enrolled_'+route]].symbol.unique()),
             summary=summary,favorite_summary=ev.stats(test[test.symbol.isin(favorites)],test[test.symbol.isin(favorites)]['score_'+route]>=.70),lift_block=liftci,brier=float(brier_score_loss(test.y,test['score_'+route])),h_brier=float(brier_score_loss(test.y,test.score_H)),
             brier_improvement=brierci,auc=float(roc_auc_score(test.y,test['score_'+route])),log_loss=float(log_loss(test.y,test['score_'+route])),
             symbols=symbols,months=months,reliability=bins,within_stock_auc=float(np.average(aucs,weights=weights)) if aucs else None,
             favorite_passed=[s['symbol'] for s in symbols if s['pass'] and s['symbol'] in favorites],
             pass_pre_holm=gate and summary['n']>=300 and summary['days']>=60 and summary['precision']>=.70 and liftci['ci'][0]>0 and brierci['ci'][0]>0,
             amplitude={'mae_median':float(np.abs(test[f'mfe_{route}_50']-test.mfe).mean()),'q20_q80_coverage':float(((test.mfe>=test[f'mfe_{route}_20'])&(test.mfe<=test[f'mfe_{route}_80'])).mean()),'quantile_crossing':int((test[f'mfe_{route}_20']>test[f'mfe_{route}_80']).sum())})
    routes=['continuation','repair','relative'];pvals=[max(report['routes'][r]['brier_improvement']['p'],report['routes'][r]['lift_block']['p']) for r in routes]
    order=np.argsort(pvals);adjusted=np.empty(3);running=0.
    for i,j in enumerate(order):running=max(running,(3-i)*pvals[j]);adjusted[j]=min(1.,running)
    for r,p in zip(routes,adjusted):
        report['routes'][r]['holm_p']=float(p);report['routes'][r]['pass']=report['routes'][r]['pass_pre_holm'] and p<.05;report['routes'][r]['favorite_5_gate']=len(report['routes'][r]['favorite_passed'])>=5
    report['pre_up_baseline']=ev.stats(test,test.pre_ret>0);report['b0_brier']=float(brier_score_loss(test.y,test.score_B0))
    test.to_parquet(OUT/'adaptive_predictions.parquet',index=False,compression='zstd',compression_level=7)
    report['predictions_sha256']=hashlib.sha256((OUT/'adaptive_predictions.parquet').read_bytes()).hexdigest()
    (OUT/'adaptive_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    for r in routes:print(json.dumps({'route':r,'summary':report['routes'][r]['summary'],'favorite_passed':report['routes'][r]['favorite_passed']}),flush=True)

if __name__=='__main__':main()
