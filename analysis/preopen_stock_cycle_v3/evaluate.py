"""Pre-period model and threshold selection; every rejected cell stays visible."""
from __future__ import annotations
import hashlib,json,pickle,os,warnings,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from scipy.special import logit
import build as b
OUT=b.OUT
MODELS=[('solo_short','solo',120,'lgbm'),('solo_long','solo',240,'lgbm'),('solo_forest','solo',120,'forest'),('related_short','related',120,'lgbm'),('related_long','related',240,'lgbm'),('regime_related','regime',240,'lgbm')]
NAMES={'solo_short':'单股短窗量价','solo_long':'单股长窗量价','solo_forest':'单股非线性量价','related_short':'相关股短窗量价','related_long':'相关股长窗量价','regime_related':'环境匹配相关股'}
MODELS += [('compact_solo_short','solo',120,'compact'),('compact_solo_long','solo',240,'compact'),('compact_related_short','related',120,'compact'),('compact_related_long','related',240,'compact'),('logistic_solo_01','solo',120,'logistic01'),('logistic_solo_1','solo',120,'logistic1'),('logistic_related_01','related',240,'logistic01'),('analog_solo_15','solo',240,'knn15'),('analog_solo_30','solo',240,'knn30'),('analog_related_60','related',240,'knn60')]
NAMES.update(compact_solo_short='单股紧凑短窗',compact_solo_long='单股紧凑长窗',compact_related_short='相关股紧凑短窗',compact_related_long='相关股紧凑长窗',logistic_solo_01='单股平滑量价',logistic_solo_1='单股线性量价',logistic_related_01='相关股平滑量价',analog_solo_15='单股相似日 15',analog_solo_30='单股相似日 30',analog_related_60='相关股相似日 60')
COMPACT=['stock_id','h_span_3','h_span_20','h_span_30','h_base_3','h_base_20','h_base_30','h_return_30','h_rv_30','h_previous_mfe','pre_rvol','pre_trade_age','pre_range_scaled','gap_scaled','pre_gap_scaled','night_ret','w30_ret','w30_eff','w30_rvol','w30_signed_volume','w60_ret','w60_eff','w60_rvol','w120_ret','slide30_max','slide30_min','qqq_h_return_30','qqq_h_rv_30','qqq_pre_gap','peer_h_return_30','peer_breadth','peer_pre_gap']
MODELS += [('macro_solo_short','solo',120,'compact'),('macro_solo_long','solo',240,'compact'),('macro_related_long','related',240,'compact'),('macro_logistic_solo','solo',120,'liblinear'),('macro_analog_solo','solo',240,'knn15')]
NAMES.update(macro_solo_short='单股市场环境短窗',macro_solo_long='单股市场环境长窗',macro_related_long='相关股市场环境',macro_logistic_solo='单股市场环境平滑',macro_analog_solo='市场环境相似日')
THRESHOLDS=[.55,.60,.65,.70,.75,.80,.85,.90,.95]
FRAME=None;FEATURES=None;AUDIT=None
LABELS={'reference','mfe','mae','close_ret','y','label_end','touch_at','cost_proxy','delayed_mfe','delayed_y','delayed_cost_proxy'}
META={'symbol','day','group','decision_at','feature_end','max_source_available','pre_reference'}
def init(model_names=None):
 global FRAME,FEATURES,AUDIT,MODELS
 if model_names:MODELS=[m for m in MODELS if m[0] in model_names]
 FRAME=pd.read_parquet(OUT/'panel.parquet');AUDIT=json.loads((OUT/'audit.json').read_text())
 FRAME['stock_id']=FRAME.symbol.map({s:i for i,s in enumerate(sorted(FRAME.symbol.unique()))}).astype(float)
 FEATURES=[c for c in FRAME.columns if c not in LABELS|META and pd.api.types.is_numeric_dtype(FRAME[c])]
 FRAME[FEATURES]=FRAME[FEATURES].replace([np.inf,-np.inf],np.nan)
def wilson(tp,n):
 if not n:return [None,None]
 z=1.959963984540054;p=tp/n;den=1+z*z/n;center=(p+z*z/2/n)/den;half=z*np.sqrt(p*(1-p)/n+z*z/4/n/n)/den
 return [float(center-half),float(center+half)]
def stats(g,score=None,threshold=None):
 mask=np.ones(len(g),dtype=bool) if score is None else np.asarray(score)>=threshold
 sel=g.loc[mask];n=len(sel);tp=int(sel.y.sum());base=float(g.y.mean()) if len(g) else None
 return dict(eligible=len(g),n=n,tp=tp,fp=n-tp,precision=tp/n if n else None,ci=wilson(tp,n),base=base,
  lift=tp/n-base if n else None,recall=tp/int(g.y.sum()) if g.y.sum() else None,supply=n/len(g) if len(g) else 0,
  cost_proxy=float(sel.cost_proxy.mean()) if n else None,delayed_precision=float(sel.delayed_y.mean()) if n else None,
  delayed_cost_proxy=float(sel.delayed_cost_proxy.mean()) if n else None,mae_mean=float(sel.mae.mean()) if n else None,
  worst_mae=float(sel.mae.min()) if n else None,close_positive=float((sel.close_ret>0).mean()) if n else None)
def transform(raw,cal):
 return cal.predict_proba(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1))[:,1]
def solve(job):
 s,start,end,complete=job;prior=FRAME[FRAME.day<start];dates=sorted(prior.day.unique());history_dates=dates[-290:]
 target=FRAME[(FRAME.symbol==s)&(FRAME.day>=start)&(FRAME.day<=end)].copy()
 cell=dict(symbol=s,start=start,end=end,complete=complete,n_test=len(target),variants=[],selected=None,pass_research=False)
 if len(history_dates)<110 or len(target)==0:cell['reason']='insufficient_history_or_target';return cell,[],[]
 fitdates=history_dates[-50:-25];selectdates=history_dates[-25:];fit=prior[(prior.symbol==s)&prior.day.isin(fitdates)];selection=prior[(prior.symbol==s)&prior.day.isin(selectdates)]
 if len(fit)<14 or len(selection)<14 or fit.y.nunique()<2:cell['reason']='insufficient_sigmoid_or_selection';return cell,[],[]
 group=b.GROUPS.get(b.GROUP_OF.get(s,''),[s]);winner=None;variant_records=[]
 for name,scope,window,kind in MODELS:
  traindates=history_dates[:-50][-window:];pool=[s] if scope=='solo' else group
  train=prior[prior.symbol.isin(pool)&prior.day.isin(traindates)].copy()
  if scope=='regime':
   # Similarity uses known prior-30-day market return and volatility only.
   # Select the environment from the last PRIOR date only.
   env=prior[prior.symbol==s].sort_values('day').iloc[-1]
   for k in ['qqq_h_return_30','qqq_h_rv_30']:
    median=float(train[k].median());mad=float((train[k]-median).abs().median())
    if np.isfinite(env[k]) and mad>0:train=train[(train[k]-env[k]).abs()<=max(mad*2, .01 if 'return' in k else .003)]
  info=dict(model=name,name=NAMES[name],scope=scope,train_rows=len(train),train_days=train.day.nunique(),selection_days=len(selection),fit_days=len(fit),calibration_pass=False)
  cell['variants'].append(info)
  if train.day.nunique()<60 or len(train)<(60 if scope=='solo' else 180) or train.y.nunique()<2:info['reason']='insufficient_train';continue
  assert train.day.max()<min(fitdates) and max(fitdates)<min(selectdates) and max(selectdates)<start
  assert pd.to_datetime(train.label_end).max()<pd.Timestamp(start)
  cols=[k for k in FEATURES if (scope!='solo' or not k.startswith(('peer_','related_'))) and (name.startswith('macro_') or not k.startswith('env_'))]
  if kind not in ('lgbm','forest'):cols=[k for k in cols if k in COMPACT or k.startswith('env_')]
  if kind in ('lgbm','compact'):model=LGBMClassifier(n_estimators=160,max_depth=3,num_leaves=7,min_child_samples=8 if scope=='solo' else 20,learning_rate=.04,reg_lambda=10,random_state=1004,n_jobs=1,verbosity=-1)
  elif kind=='forest':model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),ExtraTreesClassifier(n_estimators=240,max_depth=5,min_samples_leaf=8,random_state=1004,n_jobs=1))
  else:
   learner=KNeighborsClassifier(n_neighbors=int(kind[3:]),weights='distance',n_jobs=1) if kind.startswith('knn') else LogisticRegression(C=.1 if kind in ('logistic01','liblinear') else 1.,max_iter=1000 if kind=='liblinear' else 300,solver='liblinear' if kind=='liblinear' else 'lbfgs',random_state=1004)
   model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),RobustScaler(),learner)
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',ConvergenceWarning);model.fit(train[cols],train.y)
  if any(issubclass(w.category,ConvergenceWarning) for w in caught):info['reason']='optimizer_not_converged';continue
  rawfit=model.predict_proba(fit[cols])[:,1];cal=LogisticRegression(C=1,random_state=1004).fit(logit(np.clip(rawfit,1e-5,1-1e-5)).reshape(-1,1),fit.y)
  score=transform(model.predict_proba(selection[cols])[:,1],cal);brier=float(brier_score_loss(selection.y,score));info.update(train_first=train.day.min(),train_end=train.day.max(),fit_end=fit.day.max(),selection_end=selection.day.max(),brier=brier,thresholds=[])
  candidate=None
  for t in THRESHOLDS:
   m=stats(selection,score,t);passed=m['n']>=8 and m['precision']>=.7 and m['lift']>=.03 and m['cost_proxy']>0
   info['thresholds'].append(dict(threshold=t,**m,pass_calibration=passed))
   if passed:
    rank=(m['ci'][0],m['n'],-brier)
    if candidate is None or rank>candidate[0]:candidate=(rank,t,m)
  if candidate is None:info['reason']='no_prior_calibrated_threshold';continue
  info['calibration_pass']=True;info['threshold']=candidate[1];info['calibration']=candidate[2]
  testscore=transform(model.predict_proba(target[cols])[:,1],cal);info['outer']=stats(target,testscore,candidate[1]);info['outer']['brier']=float(brier_score_loss(target.y,testscore));info['outer']['auc']=float(roc_auc_score(target.y,testscore)) if target.y.nunique()==2 else None
  path=OUT/'models'/f'{s}_{start}_{name}.pkl';path.parent.mkdir(exist_ok=True)
  with path.open('wb') as h:pickle.dump(dict(model=model,calibrator=cal,features=cols,threshold=candidate[1],stock_id_map={v:i for i,v in enumerate(sorted(FRAME.symbol.unique()))}),h)
  info['artifact']=path.name;info['artifact_sha256']=b.sha(path)
  vt=target.copy();vt['score']=testscore;vt['signal']=testscore>=candidate[1];vt['model']=name;vt['threshold']=candidate[1];vt['cycle_start']=start;vt['cycle_end']=end;vt['train_end']=info['train_end'];vt['selection_end']=info['selection_end'];variant_records.extend(vt.to_dict('records'))
  if winner is None or candidate[0]>winner[0]:winner=(candidate[0],name,candidate[1],model,cal,cols,testscore,info)
 if winner is None:cell['reason']='no_model_selected_before_cycle';return cell,[],variant_records
 _,name,t,model,cal,cols,score,info=winner
 m=stats(target,score,t);cell.update(selected=name,threshold=t,calibration=info['calibration'],metrics=m,train_first=info['train_first'],train_end=info['train_end'],fit_end=info['fit_end'],selection_end=info['selection_end'],brier=info['outer']['brier'],auc=info['outer']['auc'])
 cell['pass_research']=bool(complete and len(target)>=14 and m['n']>=12 and m['precision']>=.7 and m['lift']>=.03 and m['cost_proxy']>0)
 cell['reason']='research_gate_passed' if cell['pass_research'] else 'outer_gate_failed'
 importance=model.feature_importances_ if hasattr(model,'feature_importances_') else model[-1].feature_importances_[:len(cols)] if hasattr(model[-1],'feature_importances_') else np.abs(model[-1].coef_[0][:len(cols)]) if hasattr(model[-1],'coef_') else np.zeros(len(cols))
 cell['importance']=sorted(zip(cols,importance.tolist()),key=lambda x:-x[1])[:30]
 path=OUT/'models'/f'{s}_{start}_{name}.pkl';path.parent.mkdir(exist_ok=True)
 with path.open('wb') as h:pickle.dump(dict(model=model,calibrator=cal,features=cols,threshold=t,stock_id_map={v:i for i,v in enumerate(sorted(FRAME.symbol.unique()))}),h)
 cell['artifact']=path.name;cell['artifact_sha256']=b.sha(path)
 target['score']=score;target['signal']=score>=t;target['model']=name;target['threshold']=t;target['cycle_start']=start;target['cycle_end']=end;target['train_end']=cell['train_end'];target['selection_end']=cell['selection_end']
 return cell,target.to_dict('records'),variant_records
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--alignment',action='store_true');args=parser.parse_args()
 names=['macro_solo_short','macro_solo_long','macro_related_long','compact_related_short','analog_solo_15','related_long'] if args.alignment else None
 init(names);end=pd.Timestamp(b.END);jobs=[]
 for anchor in (['2025-04-11','2025-04-21'] if args.alignment else ['2025-04-01']):
  for start in pd.date_range(anchor,end,freq='30D'):
   stop=start+pd.Timedelta(days=29)
   for s in AUDIT['favorites']:jobs.append((s,str(start.date()),str(min(stop,end).date()),stop<=end))
 cells=[];pred=[];variant_records=[]
 with ProcessPoolExecutor(max_workers=4,initializer=init,initargs=(names,)) as pool:
  futures={pool.submit(solve,j):j for j in jobs}
  for i,f in enumerate(as_completed(futures)):
   c,p,v=f.result();cells.append(c);pred.extend(p);variant_records.extend(v)
   if i%16==0 or c['pass_research']:print(json.dumps({'done':i+1,'total':len(jobs),'symbol':c['symbol'],'start':c['start'],'selected':c['selected'],'pass':c['pass_research'],'signals':c.get('metrics',{}).get('n'),'precision':c.get('metrics',{}).get('precision')}),flush=True)
 cells.sort(key=lambda c:(c['start'],c['symbol']));passed=[c for c in cells if c['pass_research']];distinct=sorted(set(c['symbol'] for c in passed));predframe=pd.DataFrame(pred).sort_values(['day','symbol']) if pred else pd.DataFrame()
 predpath=OUT/('alignment_predictions.parquet' if args.alignment else 'predictions.parquet');enrolledpath=OUT/('alignment_enrolled_predictions.parquet' if args.alignment else 'enrolled_predictions.parquet')
 predframe.to_parquet(predpath,index=False,compression='zstd',compression_level=7)
 pd.DataFrame(variant_records).to_parquet(enrolledpath,index=False,compression='zstd',compression_level=7)
 report=dict(version='stock_cycle_v31',protocol_sha256=b.sha(OUT/'PROTOCOL.md'),compact_protocol_sha256=b.sha(OUT/'COMPACT_PROTOCOL.md'),panel_sha256=b.sha(OUT/'panel.parquet'),predictions_sha256=b.sha(OUT/'predictions.parquet'),models=NAMES,features=FEATURES,cells=cells,passed=passed,distinct_passed=distinct,goal_pass=len(distinct)>=5,policy=stats(predframe,predframe.score,predframe.threshold) if len(predframe) else None,status='historical_forward_development_not_independent')
 enrolled=[]
 for c in cells:
  for v in c['variants']:
   if not v.get('calibration_pass'):continue
   m=v['outer'];ok=bool(c['complete'] and m['eligible']>=14 and m['n']>=12 and m['precision']>=.7 and m['lift']>=.03 and m['cost_proxy']>0)
   enrolled.append(dict(symbol=c['symbol'],start=c['start'],end=c['end'],complete=c['complete'],selected=v['model'],threshold=v['threshold'],calibration=v['calibration'],metrics=m,train_first=v['train_first'],train_end=v['train_end'],fit_end=v['fit_end'],selection_end=v['selection_end'],artifact=v['artifact'],artifact_sha256=v['artifact_sha256'],pass_research=ok,reason='enrolled_model_passed' if ok else 'enrolled_model_failed',brier=m['brier'],auc=m['auc'],variants=[]))
 report.update(version='stock_cycle_v33_alignment' if args.alignment else 'stock_cycle_v32',macro_protocol_sha256=b.sha(OUT/'MACRO_ENROLLMENT_PROTOCOL.md'),enrolled_combinations=enrolled,enrolled_passed=[c for c in enrolled if c['pass_research']],combination_goal_pass=sum(c['pass_research'] for c in enrolled)>=5,enrolled_predictions_sha256=b.sha(enrolledpath),predictions_sha256=b.sha(predpath))
 if args.alignment:report['alignment_protocol_sha256']=b.sha(OUT/'ALIGNMENT_PROTOCOL.md')
 (OUT/('alignment_results.json' if args.alignment else 'results.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps({'passed_combinations':len(passed),'enrolled_passed':sum(c['pass_research'] for c in enrolled),'enrolled_stocks':sorted(set(c['symbol'] for c in enrolled if c['pass_research'])),'distinct':distinct,'goal_pass':report['goal_pass'],'policy':report['policy']}),flush=True)
if __name__=='__main__':main()
