"""Finite preregistered monthly arms, temporal probability calibration and top-one replay."""
from __future__ import annotations
import argparse, json, pickle, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMClassifier
from common import OUT, OLD, DATES, SESSIONS, sha, write, save, now

PANEL=None;PANEL_SHA=None;PANEL_PATH=None;MONTHS=[f'2026-{m:02d}' for m in range(5,10)]
ARMS=['A0','A1','B1','B2','C1','ALG_LR','ALG_ET']
NAMES={'A0':'基础分钟量价','A1':'多日日线机会与分钟入场','B1':'相关股与QQQ','B2':'板块广度与领涨持续性','C1':'已公布事件与量价确认','ALG_LR':'基础输入算法对照','ALG_ET':'基础输入算法对照'}
ALG={a:('LogisticRegression' if a=='ALG_LR' else 'ExtraTrees' if a=='ALG_ET' else 'LightGBM') for a in ARMS}
THRESHOLDS=[.60,.65,.70,.75,.80,.85,.90,.95]
FIVE=['ALAB','AMD','MRVL','TER','TXG']
VERSION='preopen_5d5pct_v6_R00'
REPORT_TITLE='五日 +5% 免费数据 R00 结果'
GLOBAL_TOP_CALIBRATION=False
TRAINING_ONLY_VOL_EDGES=False

def init():
    global PANEL,PANEL_SHA
    PANEL_SHA=sha(OUT/'cache/panel.parquet')
    assert PANEL_SHA==json.loads((OUT/'coverage.json').read_text())['panel_sha256'],'prepare not finalized'
    PANEL=pd.read_parquet(OUT/'cache/panel.parquet');PANEL['symbol']=PANEL.symbol.astype(str);PANEL['day']=PANEL.day.astype(str)
    nums=PANEL.select_dtypes('number').columns;PANEL[nums]=PANEL[nums].replace([np.inf,-np.inf],np.nan)

def weights(f):return 1/f.groupby(['symbol','day'],observed=True).symbol.transform('size').to_numpy()

def split(month):
    cursor=next(i for i,d in enumerate(DATES) if d>=month+'-01')
    blocks=[]
    for n in [20,20,20,200]:
        cursor-=10;start=max(0,cursor-n);blocks.append(DATES[start:cursor]);cursor=start
    selection,topcal,candidatecal,training=blocks
    available=set(PANEL.day);training=[d for d in training if d in available]
    return training,candidatecal,topcal,selection

def columns(arm):
    excluded={'_row_id','minute','entry_minute','reference','feature_available','prior_volume','last_volume','liquidity_dollars','stock_id','ex_action','entry','target','y','label_end','future_status'}
    base=[c for c in PANEL.select_dtypes('number') if c not in excluded and not c.startswith(('qqq_','peer_','relative_','d_','b_','e_'))]
    if arm=='A1':base += [c for c in PANEL if c.startswith('d_')]
    if arm in ['B1','B2']:base += [c for c in PANEL if c.startswith(('qqq_','peer_','relative_'))]
    if arm=='B2':base += [c for c in PANEL if c.startswith('b_')]
    if arm=='C1':base += [c for c in PANEL if c.startswith('e_')]
    return base

def estimator(arm):
    if arm=='ALG_LR':return Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',RobustScaler()),('model',LogisticRegression(C=.1,max_iter=1500,solver='liblinear',random_state=1005))])
    if arm=='ALG_ET':return Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('model',ExtraTreesClassifier(n_estimators=160,max_depth=8,min_samples_leaf=80,random_state=1005,n_jobs=1))])
    return LGBMClassifier(n_estimators=180,max_depth=4,num_leaves=15,min_child_samples=80,reg_lambda=10,learning_rate=.04,random_state=1005,n_jobs=1,verbosity=-1)

def raw_predict(model,cols,f):return model.predict_proba(f[cols])[:,1]
def transformed(p):return logit(np.clip(p,1e-5,1-1e-5)).reshape(-1,1)
def apply_cal(cal,p):return cal.predict_proba(transformed(p))[:,1] if cal is not None else p
def fit_cal(f,p):
    valid=f.y.notna().to_numpy()
    if valid.sum()<40 or f.loc[valid,'y'].nunique()<2:return None
    model=LogisticRegression(C=1,max_iter=1000,random_state=1005).fit(transformed(p[valid]),f.loc[valid,'y'],sample_weight=weights(f.loc[valid]))
    return model if model.coef_[0,0]>0 else None

def baseline_table(history,edge_history=None):
    valid=history[history.y.notna()].copy();edges=valid if edge_history is None else edge_history[edge_history.y.notna()]
    quantiles=np.quantile(edges.h_rv_30.dropna(),[1/3,2/3])
    valid['_volbin']=np.searchsorted(quantiles,valid.h_rv_30.fillna(-1))
    gm=valid.groupby('minute',observed=True).y.mean().to_dict()
    sm=valid.groupby(['symbol','minute'],observed=True).y.agg(['sum','count'])
    sm={k:float((r['sum']+20*gm[k[1]])/(r['count']+20)) for k,r in sm.iterrows()}
    vm=valid.groupby(['symbol','minute','_volbin'],observed=True).y.agg(['sum','count'])
    vm={k:float((r['sum']+20*sm[k[:2]])/(r['count']+20)) for k,r in vm.iterrows()}
    return dict(vol_edges=quantiles.tolist(),minute=gm,stock_minute=sm,matched=vm)

def baseline(table,f):
    bins=np.searchsorted(table['vol_edges'],f.h_rv_30.fillna(-1))
    return np.asarray([table['matched'].get((s,int(m),int(v)),table['stock_minute'].get((s,int(m)),table['minute'].get(int(m),.5))) for s,m,v in zip(f.symbol,f.minute,bins)])

def replay(f,thresholds):
    # Future labels are carried for later scoring, never read by the selection loop.
    events=[];ticks=[];issued=set();dayprev=None
    fields=['symbol','day','minute','score','baseline','y','entry','target','label_end','future_status','reference','feature_available','h_rv_30']
    ordered=f.sort_values(['day','minute','score','symbol'],ascending=[True,True,False,True])
    for (day,minute),g in ordered.groupby(['day','minute'],sort=True,observed=True):
        if day!=dayprev:issued=set();dayprev=day
        phase='pre' if minute<=570 else 'regular';threshold=thresholds.get(phase)
        eligible=g[~g.symbol.isin(issued)&g.score.notna()]
        if threshold is None:reason='threshold_not_admitted';eligible=eligible.iloc[:0]
        else:eligible=eligible[eligible.score>=threshold];reason='low_confidence_or_deduplicated'
        if len(eligible):
            e=eligible.iloc[0][fields].to_dict();e.update(phase=phase,threshold=threshold);events.append(e);issued.add(e['symbol']);reason='issued'
        ticks.append(dict(day=day,minute=int(minute),phase=phase,reason=reason,candidates=len(g)))
    return events,ticks

def block_ci(events,days,iterations=1000):
    weeks=sorted({str((pd.Timestamp(d)-pd.Timedelta(days=pd.Timestamp(d).weekday())).date()) for d in days})
    if not weeks:return [None,None]
    counts=np.zeros((len(weeks),2))
    for e in events:
        if pd.isna(e['y']):continue
        week=str((pd.Timestamp(e['day'])-pd.Timedelta(days=pd.Timestamp(e['day']).weekday())).date());counts[weeks.index(week)]+=[e['y'],1]
    if not counts[:,1].sum():return [None,None]
    rng=np.random.default_rng(1005);values=[]
    for _ in range(iterations):
        starts=rng.integers(len(weeks),size=(len(weeks)+1)//2);ix=np.array([[i,(i+1)%len(weeks)] for i in starts]).ravel()[:len(weeks)];total=counts[ix].sum(0)
        if total[1]:values.append(total[0]/total[1])
    return np.quantile(values,[.025,.975]).tolist()

def metrics(events,ticks,include_ci=True):
    known=[e for e in events if pd.notna(e['y'])];tp=sum(int(e['y']) for e in known);n=len(known);alln=len(events)
    base=float(np.mean([e['baseline'] for e in known])) if n else None
    days=sorted({t['day'] for t in ticks});signal_days=len({e['day'] for e in events})
    weeks=len({str((pd.Timestamp(d)-pd.Timedelta(days=pd.Timestamp(d).weekday())).date()) for d in days})
    p=tp/n if n else None
    return dict(signals=alln,mature=n,tp=tp,fp=n-tp,pending=alln-n,precision=p,conservative_precision=tp/alln if alln else None,
         date_count=signal_days,days=len(days),weeks=weeks,signals_per_week=alln/weeks if weeks else 0,
         abstentions=len(ticks)-alln,abstention_ratio=(len(ticks)-alln)/len(ticks) if ticks else None,
         baseline=base,lift=p-base if n else None,
         mean_signal_probability=float(np.mean([e['score'] for e in known])) if n else None,
         brier=float(np.mean([(e['score']-e['y'])**2 for e in known])) if n else None,
         baseline_brier=float(np.mean([(e['baseline']-e['y'])**2 for e in known])) if n else None,
         block_ci=block_ci(events,days) if include_ci else None,
         lift_ci=block_ci([dict(e,y=e['y']-e['baseline']) for e in known],days) if include_ci else None,
         mature_date_count=len({e['day'] for e in known}))

def probability_report(f):
    f=f[f.y.notna()].copy()
    if f.empty:return dict(stock_days=0,brier=None,baseline_brier=None,bins=[])
    w=weights(f);bins=[]
    for low in np.arange(0,1,.1):
        ix=(f.score>=low)&(f.score<low+.1 if low<.89 else f.score<=1)
        if ix.any():bins.append(dict(lower=float(low),upper=float(low+.1),rows=int(ix.sum()),stock_days=f.loc[ix,['symbol','day']].drop_duplicates().shape[0],
             predicted=float(np.average(f.loc[ix,'score'],weights=w[ix])),observed=float(np.average(f.loc[ix,'y'],weights=w[ix]))))
    return dict(stock_days=f[['symbol','day']].drop_duplicates().shape[0],brier=float(np.average((f.score-f.y)**2,weights=w)),baseline_brier=float(np.average((f.baseline-f.y)**2,weights=w)),
       predicted=float(np.average(f.score,weights=w)),observed=float(np.average(f.y,weights=w)),bins=bins)

def fit(job):
    arm,month=job;dest=OUT/'cache/runs'/f'{arm}_{month}';meta_path=dest/'meta.json'
    if meta_path.exists():return json.loads(meta_path.read_text())
    td,cd,hd,sd=split(month);cols=columns(arm)
    tr=PANEL[PANEL.day.isin(td)&(PANEL.minute%15==5)&PANEL.y.notna()].copy()
    registered=tr.groupby('symbol').day.nunique();registered=sorted(registered[registered>=60].index)
    tr=tr[tr.symbol.isin(registered)];ca=PANEL[PANEL.day.isin(cd)&PANEL.symbol.isin(registered)].copy()
    hc=PANEL[PANEL.day.isin(hd)&PANEL.symbol.isin(registered)].copy();se=PANEL[PANEL.day.isin(sd)&PANEL.symbol.isin(registered)].copy()
    test=PANEL[(PANEL.day.str[:7]==month)&PANEL.symbol.isin(registered)].copy()
    assert all(pd.to_datetime(a.label_end).max()<pd.Timestamp(b[0]) for a,b in [(tr,cd),(ca,hd),(hc,sd),(se,[month+'-01'])])
    if arm=='C1' and (tr.e_missing==0).sum()<200:
        meta=dict(arm=arm,month=month,status='blocked_free_event_PIT_coverage',algorithm=ALG[arm],name=NAMES[arm],events=int((tr.e_missing==0).sum()))
        write(meta_path,meta);return meta
    assert not set(cols)&{'y','entry','target','label_end','future_status'}
    model=estimator(arm)
    with warnings.catch_warnings(record=True) as caught:
        if arm.startswith('ALG_'):model.fit(tr[cols],tr.y,model__sample_weight=weights(tr))
        else:model.fit(tr[cols],tr.y,sample_weight=weights(tr))
    cals={};tops={};topcounts={}
    ca['raw']=raw_predict(model,cols,ca);hc['raw']=raw_predict(model,cols,hc)
    for phase in ['pre','regular']:
        f=ca[ca.minute<=570] if phase=='pre' else ca[ca.minute>570]
        cals[phase]=fit_cal(f,f.raw.to_numpy())
    if GLOBAL_TOP_CALIBRATION:
        hc['score']=np.nan;hc['baseline']=.5
        for phase in ['pre','regular']:
            ix=hc.minute<=570 if phase=='pre' else hc.minute>570
            hc.loc[ix,'score']=apply_cal(cals[phase],hc.loc[ix,'raw'].to_numpy())
        all_firsts,_=replay(hc,dict(pre=0.,regular=0.));first_cohort=pd.DataFrame(all_firsts)
    for phase in ['pre','regular']:
        h=hc[hc.minute<=570].copy() if phase=='pre' else hc[hc.minute>570].copy()
        if GLOBAL_TOP_CALIBRATION:
            top=first_cohort[first_cohort.phase==phase]
        else:
            h['score']=apply_cal(cals[phase],h.raw.to_numpy());h['baseline']=.5
            es,_=replay(h,dict(pre=0.,regular=0.));top=pd.DataFrame(es)
        tops[phase]=fit_cal(top,top.score.to_numpy()) if len(top) else None
        topcounts[phase]=dict(rows=len(top),calibrator_fitted=tops[phase] is not None)
    history=PANEL[PANEL.day.isin(td+cd+hd)&PANEL.symbol.isin(registered)&PANEL.y.notna()]
    base=baseline_table(history,tr if TRAINING_ONLY_VOL_EDGES else None);variants={}
    for variant in ['candidate_only','top_one']:
        def predict(frame):
            raw=raw_predict(model,cols,frame);p=np.empty(len(frame))
            for phase in ['pre','regular']:
                ix=(frame.minute<=570).to_numpy() if phase=='pre' else (frame.minute>570).to_numpy()
                p[ix]=apply_cal(cals[phase],raw[ix])
                if variant=='top_one':p[ix]=apply_cal(tops[phase],p[ix])
            return p
        sel=se.copy();sel['score']=predict(sel);sel['baseline']=baseline(base,sel)
        thresholds={};choices={}
        for phase in ['pre','regular']:
            f=sel[sel.minute<=570] if phase=='pre' else sel[sel.minute>570];cs=[]
            for threshold in THRESHOLDS:
                events,ticks=replay(f,dict(pre=threshold,regular=threshold));m=metrics(events,ticks)
                passed=bool(cals[phase] is not None and m['mature']>=20 and m['date_count']>=10 and m['precision']>=.8 and m['lift']>=.05 and m['pending']/max(m['signals'],1)<=.05)
                cs.append(dict(threshold=threshold,metrics=m,passed=passed))
            ok=[c for c in cs if c['passed']];best=max(ok,key=lambda c:(c['metrics']['block_ci'][0] or 0,c['metrics']['signals'])) if ok else None
            thresholds[phase]=best['threshold'] if best else None;choices[phase]=cs
        outer=test.copy();outer['score']=predict(outer);outer['baseline']=baseline(base,outer)
        events,ticks=replay(outer,thresholds);first,firstticks=replay(outer,dict(pre=0.,regular=0.))
        m=metrics(events,ticks)
        variants[variant]=dict(thresholds=thresholds,choices=choices,outer=m,all_candidates=probability_report(outer),
                              first_candidates=probability_report(pd.DataFrame(first)) if first else {},signal_probability=probability_report(pd.DataFrame(events)) if events else {})
        save(outer[['symbol','day','minute','score','baseline','y','entry','target','label_end','future_status','reference','feature_available','h_rv_30']],dest/f'{variant}.parquet')
        write(dest/f'{variant}_replay.json',dict(events=events,ticks=ticks))
        print(json.dumps(dict(arm=arm,month=month,variant=variant,precision=m['precision'],signals=m['signals'],thresholds=thresholds)),flush=True)
    artifact=dict(arm=arm,month=month,features=cols,model=model,candidate_calibrators=cals,top_calibrators=tops,registered=registered,baseline=base,variants={v:x['thresholds'] for v,x in variants.items()})
    assert sha(PANEL_PATH or OUT/'cache/panel.parquet')==PANEL_SHA,'panel mutated while fitting'
    model_path=OUT/'models'/f'{arm}_{month}.pkl';model_path.parent.mkdir(exist_ok=True);model_path.write_bytes(pickle.dumps(artifact))
    meta=dict(status='completed_development',arm=arm,month=month,name=NAMES[arm],algorithm=ALG[arm],target='5d5pct',training=[td[0],td[-1]],train_days=len(td),
              candidate_calibration=[cd[0],cd[-1]],top_calibration=[hd[0],hd[-1]],selection=[sd[0],sd[-1]],
              features=cols,registered=registered,top_calibration_counts=topcounts,variants=variants,model_sha256=sha(model_path),
              top_calibration_dedup='global_stock_day_both_phases' if GLOBAL_TOP_CALIBRATION else 'phase_local_stock_day',baseline_vol_edges='training_only' if TRAINING_ONLY_VOL_EDGES else 'training_candidate_and_top_blocks',
              panel_sha256=PANEL_SHA,panel_file=(PANEL_PATH or OUT/'cache/panel.parquet').name,protocol_sha256=sha(OUT/'PROTOCOL.md'),warnings=[str(w.message)[:240] for w in caught],completed_at=now())
    write(meta_path,meta);return meta

def aggregate():
    report=dict(version=VERSION,status='exposed_historical_development',independent_pass=False,created_at=now(),arms=[])
    for arm in ARMS:
        metas=[json.loads(p.read_text()) for p in sorted((OUT/'cache/runs').glob(f'{arm}_*/meta.json'))]
        if not metas:continue
        variants=[]
        for variant in ['candidate_only','top_one']:
            events=[];ticks=[]
            for m in metas:
                p=OUT/'cache/runs'/f"{arm}_{m['month']}"/f'{variant}_replay.json'
                if p.exists():r=json.loads(p.read_text());events+=r['events'];ticks+=r['ticks']
            m=metrics(events,ticks);stocks=[]
            for symbol in sorted({e['symbol'] for e in events}):
                sm=metrics([e for e in events if e['symbol']==symbol],ticks);stocks.append(dict(symbol=symbol,**sm,favorite_required=symbol in FIVE))
            weeks=[];months=[]
            for period in sorted({t['day'][:7] for t in ticks}):
                months.append(dict(month=period,**metrics([e for e in events if e['day'].startswith(period)],[t for t in ticks if t['day'].startswith(period)])))
            ws=sorted({str((pd.Timestamp(t['day'])-pd.Timedelta(days=pd.Timestamp(t['day']).weekday())).date()) for t in ticks})
            for w in ws:
                end=str((pd.Timestamp(w)+pd.Timedelta(days=6)).date());weeks.append(dict(week=w,**metrics([e for e in events if w<=e['day']<=end],[t for t in ticks if w<=t['day']<=end],False)))
            last={};nonoverlap=[]
            for e in events:
                pos=DATES.index(e['day'])
                if pos-last.get(e['symbol'],-100)>=5:nonoverlap.append(e);last[e['symbol']]=pos
            variants.append(dict(variant=variant,metrics=m,stocks=stocks,months=months,weeks=weeks,
                nonoverlap=metrics(nonoverlap,ticks),signal_probability=probability_report(pd.DataFrame(events)) if events else {},
                favorite_five_passed=[s['symbol'] for s in stocks if s['symbol'] in FIVE and s['mature']>=20 and s['precision']>=.75],
                historical_core_constraints_met=bool(m['mature']>=100 and m['date_count']>=40 and m['precision']>=.8 and m['block_ci'][0]>=.7 and m['lift']>=.05 and m['weeks']>=12 and m['signals_per_week']>=3),
                historical_all_constraints_met=False,probability_audit_status='run calibration_audit.py before assessing the full historical gate'))
            write(OUT/f'{arm}_{variant}_signals.json',dict(events=events,status='development',issued_live=False))
        report['arms'].append(dict(arm=arm,name=NAMES[arm],algorithm=ALG[arm],months_completed=len(metas),statuses=[m['status'] for m in metas],variants=variants))
    write(OUT/'results.json',report)
    lines=['# '+REPORT_TITLE,'', '已暴露历史开发回测。最终独立验证尚未开始；任何历史高达成率均不赋予未来买入资格。','', '| 输入思路 / 算法 | 校准 | 成熟 / 全部 | 达成率 | 日期 | 每周信号 | 匹配基准 | 增量 | 信号 Brier |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def pct(x):return '不可评分' if x is None else f'{x:.2%}'
    for arm in report['arms']:
        for v in arm['variants']:
            m=v['metrics'];lines.append(f"| {arm['name']} / {arm['algorithm']} | {v['variant']} | {m['mature']}/{m['signals']} | {pct(m['precision'])} | {m['date_count']} | {m['signals_per_week']:.2f} | {pct(m['baseline'])} | {pct(m['lift'])} | {m['brier'] if m['brier'] is not None else '不可评分'} |")
    lines+=['','逐周、逐月、逐股、第一名群体及信号校准、未知和不重叠敏感性见 results.json 与 cache/runs/*/meta.json。没有信号为不可评分，不把低置信度候选计入准确率。','', '股票池为当前快照回溯；既有五日标签与原始价格绑定。事件审计并不证明历史预期可知。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fit',action='store_true');ap.add_argument('--aggregate',action='store_true');ap.add_argument('--arms',nargs='+',default=ARMS);ap.add_argument('--months',nargs='+',default=MONTHS);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    if a.fit:
        with ProcessPoolExecutor(max_workers=a.workers,initializer=init) as pool:
            fs={pool.submit(fit,(arm,month)):(arm,month) for arm in a.arms for month in a.months}
            for future in as_completed(fs):
                try:future.result()
                except Exception as e:
                    arm,month=fs[future];write(OUT/'cache/runs'/f'{arm}_{month}'/'failure.json',dict(at=now(),status='failed',error=repr(e)));print(json.dumps(dict(arm=arm,month=month,error=repr(e))),flush=True)
        aggregate()
    if a.aggregate:aggregate()
if __name__=='__main__':main()
