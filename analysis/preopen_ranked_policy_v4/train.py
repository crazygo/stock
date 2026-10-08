"""Monthly pre-period fits. Evaluate the top-one policy, never cherry-pick stock wins."""
from __future__ import annotations
import argparse,json,pickle,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from lightgbm import LGBMClassifier
from common import *
from data import STATIC,META,LABELS
from policy import replay,gate
from store import persist

FRAME=None
def init():
 global FRAME
 FRAME=pd.read_parquet(OUT/'panel.parquet');FRAME['day']=FRAME.day.astype(str);FRAME['symbol']=FRAME.symbol.astype(str)
 numeric=FRAME.select_dtypes('number').columns;FRAME[numeric]=FRAME[numeric].replace([np.inf,-np.inf],np.nan)
def weights(f):return 1/f.groupby(['symbol','day'],observed=True).symbol.transform('size').to_numpy()
def score(artifact,f):
 raw=artifact['model'].predict_proba(f[artifact['features']])[:,1]
 return artifact['calibrator'].predict_proba(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1))[:,1]
def features(route):
 num=[c for c in FRAME.select_dtypes('number') if c not in set(META+LABELS+['liquidity_dollars','last_volume','rth_volume','prior_volume','stock_id','ex_action'])]
 if route=='hazard_linear':return [k for k in num if k in STATIC+['remaining','elapsed','rth_ret','rth_gap','rth_range','rth_rvol','i15_rv','i60_rv']]+['symbol']
 if route=='recovery_forest':return [k for k in num if not k.startswith(('qqq_','peer_','relative_'))]+['stock_id']
 return num+['stock_id']
def estimator(route,cols):
 if route=='hazard_linear':
  transform=ColumnTransformer([('number',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',RobustScaler())]),[k for k in cols if k!='symbol']),('stock',OneHotEncoder(handle_unknown='ignore'),['symbol'])],sparse_threshold=1.)
  return Pipeline([('features',transform),('learner',LogisticRegression(C=.1,max_iter=1500,solver='liblinear',random_state=1004))])
 if route=='relative_flow':return LGBMClassifier(n_estimators=180,max_depth=4,num_leaves=15,min_child_samples=80,reg_lambda=10,learning_rate=.04,random_state=1004,n_jobs=1,verbosity=-1)
 return Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('learner',ExtraTreesClassifier(n_estimators=160,max_depth=8,min_samples_leaf=80,random_state=1004,n_jobs=1))])

def baseline(prior,test):
 valid=prior[prior.y.notna()];counts=valid.groupby(['symbol','minute'],observed=True).y.agg(['sum','count']);globalminute=valid.groupby('minute',observed=True).y.mean()
 lookup=(counts['sum']+20*counts.index.get_level_values('minute').map(globalminute))/(counts['count']+20)
 return np.asarray([lookup.get((s,int(m)),globalminute.get(int(m),.1)) for s,m in zip(test.symbol,test.minute)],dtype=float)

def solve(job):
 route,month=job;start=month+'-01';prior=FRAME[FRAME.day<start];dates=sorted(prior.day.unique())[-240:]
 train_dates=dates[:-40];fitdates=dates[-40:-20];seldates=dates[-20:]
 # Fixed 15-minute training grid reduces adjacent-bar replication; live scoring remains every 5m.
 train=prior[prior.day.isin(train_dates)&(prior.minute%15==5)&prior.y.notna()].copy();fit=prior[prior.day.isin(fitdates)&prior.y.notna()].copy();sel=prior[prior.day.isin(seldates)].copy()
 counts=train.groupby('symbol',observed=True).day.nunique();registered=sorted(counts[counts>=60].index);train=train[train.symbol.isin(registered)];fit=fit[fit.symbol.isin(registered)];sel=sel[sel.symbol.isin(registered)]
 assert train.day.max()<min(fitdates) and max(fitdates)<min(seldates) and max(seldates)<start
 assert pd.to_datetime(train.label_end).max()<pd.Timestamp(start)
 cols=features(route);model=estimator(route,cols)
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always',ConvergenceWarning)
  if route=='relative_flow':model.fit(train[cols],train.y,sample_weight=weights(train))
  else:model.fit(train[cols],train.y,learner__sample_weight=weights(train))
 if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError(route+' optimizer not converged')
 raw=model.predict_proba(fit[cols])[:,1];cal=LogisticRegression(C=1.,max_iter=1000,random_state=1004).fit(logit(np.clip(raw,1e-5,1-1e-5)).reshape(-1,1),fit.y,sample_weight=weights(fit))
 artifact=dict(model=model,calibrator=cal,features=cols,registered=registered,route=route,month=month,stock_map=FRAME[['symbol','stock_id']].drop_duplicates().set_index('symbol').stock_id.to_dict())
 sel['score']=score(artifact,sel);sel['baseline']=baseline(prior[prior.day.isin(train_dates+fitdates)],sel)
 brier=float(brier_score_loss(sel.loc[sel.y.notna(),'y'],sel.loc[sel.y.notna(),'score'],sample_weight=weights(sel[sel.y.notna()])))
 choices=[]
 for t in THRESHOLDS:
  sim=replay(sel,t);m=sim['metrics'];choices.append(dict(threshold=t,metrics=m,passed=gate(m,n=8,days=6)))
 passed=[c for c in choices if c['passed']];best=max(passed,key=lambda c:(c['metrics']['ci'][0],c['metrics']['n'],-brier)) if passed else None;threshold=best['threshold'] if best else None
 artifact['threshold']=threshold;artifact['baseline_table']={f'{s}|{int(m)}':float(p) for s,m,p in zip(sel.symbol,sel.minute,sel.baseline)}
 # Baseline for October is learned only from pre-October historical labels.
 counts2=prior[prior.y.notna()].groupby(['symbol','minute'],observed=True).y.agg(['sum','count']);gm=prior[prior.y.notna()].groupby('minute',observed=True).y.mean();artifact['baseline_table']={f'{s}|{m}':float((r['sum']+20*gm.get(m,.1))/(r['count']+20)) for (s,m),r in counts2.iterrows()};artifact['baseline_minute']=gm.to_dict()
 path=OUT/'models'/f'{route}_{month}.pkl';path.parent.mkdir(exist_ok=True);path.write_bytes(pickle.dumps(artifact))
 meta=dict(route=route,name=NAMES[route],month=month,train_first=train.day.min(),train_end=train.day.max(),calibration_start=min(fitdates),calibration_end=max(fitdates),selection_start=min(seldates),selection_end=max(seldates),train_rows=len(train),train_days=len(train_dates),registered=registered,features=cols,threshold=threshold,prior_gate_pass=best is not None,brier=brier,choices=choices,artifact=path.name,artifact_sha256=sha(path),protocol_sha256=sha(OUT/'PROTOCOL.md'))
 if month in ['2026-08','2026-09']:
  test=FRAME[(FRAME.day.str[:7]==month)&FRAME.symbol.isin(registered)].copy();test['score']=score(artifact,test);test['baseline']=baseline(prior,test);sim=replay(test,threshold);meta['outer']=sim['metrics'];meta['outer_gate_pass']=gate(meta['outer'])
  save(test,OUT/f'{route}_{month}.parquet');write(OUT/f'{route}_{month}_replay.json',sim);write(OUT/f'{route}_{month}_meta.json',meta)
 else:write(OUT/f'{route}_{month}_meta.json',meta)
 print(json.dumps({'route':route,'month':month,'threshold':threshold,'calibration':best['metrics'] if best else None,'outer':meta.get('outer')}),flush=True)
 return meta

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--october',action='store_true');ap.add_argument('--finalize',action='store_true');a=ap.parse_args();months=['2026-10'] if a.october else ['2026-08','2026-09'];metas=[]
 if a.finalize:metas=[json.loads((OUT/f'{r}_{m}_meta.json').read_text()) for m in months for r in ROUTES]
 else:
  with ProcessPoolExecutor(max_workers=3,initializer=init) as pool:
   for f in as_completed([pool.submit(solve,(r,m)) for m in months for r in ROUTES]):metas.append(f.result())
 if not a.october:
  init();route_summary=[]
  for route in ROUTES:
   combined=pd.concat([pd.read_parquet(OUT/f'{route}_{m}.parquet') for m in months],ignore_index=True);monthsims=[]
   for month in months:
    meta=next(x for x in metas if x['route']==route and x['month']==month);sim=json.loads((OUT/f'{route}_{month}_replay.json').read_text());persist(f'v4:{route}:{month}',route,month,combined[combined.day.str[:7]==month],sim,meta);monthsims.append(sim)
   from policy import metrics
   ts=sum([s['trades'] for s in monthsims],[]);ds=sum([s['decisions'] for s in monthsims],[])
   # Monthly accounts each start at 100k; continuous Aug-Sep replay below uses month's frozen threshold.
   combo=combined.copy();ths={m:next(x for x in metas if x['route']==route and x['month']==m)['threshold'] for m in months}
   combo['score_original']=combo.score;combo['threshold']=pd.to_numeric(combo.day.str[:7].map(ths),errors='coerce');combo['score']=np.where(combo.threshold.notna()&(combo.score>=combo.threshold),combo.score,-1.)
   continuous=replay(combo,0.);m=continuous['metrics'];route_summary.append(dict(route=route,name=NAMES[route],metrics=m,passed=gate(m),months=[x for x in metas if x['route']==route]));write(OUT/f'{route}_continuous_replay.json',continuous)
   meta=next(x for x in metas if x['route']==route and x['month']=='2026-09').copy();meta.update(threshold=0.,month_thresholds=ths,account='continuous August September settled cash');persist(f'v4:{route}:2026-08_09',route,'2026-08_09',combo,continuous,meta)
  passed=[x for x in route_summary if x['passed']];selected=max(passed,key=lambda x:(x['metrics']['ci'][0],x['metrics']['n']))['route'] if passed else None
  write(OUT/'results.json',dict(version='ranked_policy_v4',cycles=months,routes=route_summary,selected_route=selected,status='historical_forward_development',october_exposed=['2026-10-01','2026-10-02'],protocol_sha256=sha(OUT/'PROTOCOL.md'),panel_sha256=sha(OUT/'panel.parquet')))
  print(json.dumps({'selected':selected,'summary':[{k:r[k] for k in ['route','metrics','passed']} for r in route_summary]}),flush=True)
 else:
  results=json.loads((OUT/'results.json').read_text());selected=results['selected_route'];meta=next((m for m in metas if m['route']==selected),None)
  write(OUT/'deployment.json',dict(frozen_at=datetime.now(ET).isoformat(),observe_from='2026-10-05',observe_through='2026-10-30',route=selected,admitted=bool(meta and meta['prior_gate_pass']),metadata=meta,all_routes=metas,already_exposed=['2026-10-01','2026-10-02'],selection_basis='August September top-one policy only'))
if __name__=='__main__':main()
