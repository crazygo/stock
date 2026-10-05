"""A single calibrated distribution over first doubling time; chronological blocks."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

PARENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PARENT))
from data import FEATURES,CACHE
_spec=importlib.util.spec_from_file_location('doubling_v1_evaluation',PARENT/'model.py')
_evaluate_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_evaluate_module)
evaluate=_evaluate_module.evaluate
ALGORITHMS={'logistic':'multinomial LogisticRegression','hist_gbdt':'multiclass HistGradientBoostingClassifier'}


def event_class(rows):
    known=rows.y30.notna()&rows.y60.notna()
    if ((rows.loc[known,'y30']==1)&(rows.loc[known,'y60']==0)).any():raise ValueError('Impossible nested labels')
    y=np.full(len(rows),-1,dtype=int)
    y[known]=np.where(rows.loc[known,'y30']==1,2,np.where(rows.loc[known,'y60']==1,1,0))
    return y


def nested_probabilities(model,rows,features=FEATURES):
    q=model.predict_proba(rows[features].to_numpy())
    positions={int(c):i for i,c in enumerate(model.classes_)}
    if set(positions)!={0,1,2}:raise ValueError('All first-touch classes must be present')
    p30=q[:,positions[2]];p60=np.minimum(1.,p30+q[:,positions[1]])
    if not np.all((p30>=0)&(p30<=p60)&(p60<=1)):raise ValueError('Invalid nested probabilities')
    return p30,p60,q


def fit_month(d,month,cfg,features=FEATURES,version='joint_v2_price_control'):
    first=pd.Timestamp(month+'-01');start=first-pd.Timedelta(days=120);end=first-pd.Timedelta(days=60);mid=start+pd.Timedelta(days=40)
    valid=d.eligible&d.y30.notna()&d.y60.notna()
    train=d[valid&(d.date<start)&(d.mature_at60<start)]
    cal=d[valid&(d.date>=start)&(d.date<end)&(d.mature_at60<first)]
    fitcal=cal[cal.date<mid];choose=cal[cal.date>=mid]
    ys={n:event_class(q) for n,q in [('train',train),('calibrate',fitcal),('select',choose)]}
    audit={'version':version,'month':month,'evaluation_start':str(first.date()),
           'train_rows':len(train),'calibration_rows':len(fitcal),'selection_rows':len(choose),
           'calibration_start':str(start.date()),'calibration_fit_end_exclusive':str(mid.date()),'selection_end_exclusive':str(end.date()),
           'train_latest_label_maturity':str(train.mature_at60.max()) if len(train) else None,
           'cal_latest_label_maturity':str(cal.mature_at60.max()) if len(cal) else None,
           'class_counts':{n:np.bincount(y,minlength=3).tolist() for n,y in ys.items()},'algorithms':{},'features':features}
    if len(train)<500 or len(fitcal)<200 or len(choose)<200 or min(audit['class_counts']['train'])<20 or min(audit['class_counts']['calibrate'])<20:
        audit['status']='insufficient_mature_class_support';return {},audit
    models={}
    for name in ALGORITHMS:
        logistic_steps=([SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)] if features!=FEATURES else [])
        model=(make_pipeline(*logistic_steps,StandardScaler(),LogisticRegression(C=1,max_iter=500,random_state=cfg['seed'])) if name=='logistic' else
               HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=2,random_state=cfg['seed']))
        model.fit(train[features].to_numpy(),ys['train'])
        calibrated=CalibratedClassifierCV(FrozenEstimator(model),method='temperature').fit(fitcal[features].to_numpy(),ys['calibrate'])
        p30,p60,q=nested_probabilities(calibrated,choose,features)
        b30=float(brier_score_loss(choose.y30,p30));b60=float(brier_score_loss(choose.y60,p60))
        audit['algorithms'][name]={'algorithm':ALGORITHMS[name],'calibration':'joint temperature scaling',
             'selection_brier30':b30,'selection_brier60':b60,'selection_objective':(b30+b60)/2,
             'selection_p60_gt80_count':int((p60>cfg['threshold']).sum()),'selection_order_violations':int((p30>p60).sum())}
        models[name]={'model':model,'calibrated':calibrated,'features':features}
    audit['champion']=min(models,key=lambda n:audit['algorithms'][n]['selection_objective']);audit['status']='development_joint_calibrated'
    return models,audit


def forward_backtest(d,cfg,out,features=FEATURES,version='joint_v2_price_control',prediction_name='joint_v2_forward_predictions.parquet',signal_gate=None):
    out.mkdir(parents=True,exist_ok=True);allrows=[];audits=[]
    for month in cfg['evaluation_months']:
        print('joint fit',month,flush=True);models,audit=fit_month(d,month,cfg,features,version);audits.append(audit)
        if not models:continue
        target=d[d.eligible&(d.date.dt.strftime('%Y-%m')==month)]
        for name,m in models.items():
            p30,p60,classes=nested_probabilities(m['calibrated'],target,m['features'])
            for h,p in [(30,p30),(60,p60)]:
                q=target[['ticker','date','close',f'y{h}',f'mature_at{h}',f'first_touch{h}',f'end_close_double{h}',f'label_status{h}']].copy()
                q.columns=['ticker','date','reference_close','label','mature_at','first_touch','end_close_double','label_status']
                q['horizon']=h;q['algorithm']=name;q['probability']=p;q['p30']=p30;q['p60']=p60
                q['month']=month;q['selected_algorithm']=name==audit['champion'];q['version']=version
                q['signal_eligible']=target[signal_gate].to_numpy() if signal_gate else True
                allrows.append(q)
    (out/'split_audit.json').write_text(json.dumps(audits,indent=2))
    if not allrows:raise ValueError('No mature supported fold')
    allrows=pd.concat(allrows,ignore_index=True)
    allrows.to_parquet(CACHE/prediction_name,compression='zstd',compression_level=7,index=False)
    stats=evaluate(allrows[allrows.signal_eligible],cfg,out)
    for s in stats:
        s['algorithm_name']=ALGORITHMS.get(s['algorithm'],'monthly joint-model selection')
        s['model_version']=version;s['calibration']='joint temperature scaling'
        s['signal_gate']=signal_gate;s['all_price_eligible_prediction_rows']=int((allrows.horizon==s['horizon']).sum())
        s['blocking_evidence'].append('RTH_only_daily_label_unverified')
    (out/'backtest_summary.json').write_text(json.dumps(stats,indent=2))
    return allrows,audits,stats
