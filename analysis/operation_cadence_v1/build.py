"""Fit frozen development probabilities and retain all forward evaluations."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import hashlib, json, math
import numpy as np
import pandas as pd
from model import *

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d):(OUT/name).write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def finite(x):return [float(v) if v is not None and np.isfinite(v) else None for v in x]

def score(rows,key):
    ok=[r for r in rows if r['labels'][key]['status']=='mature' and r['p'].get(key) is not None]
    if not ok:return {'n':0,'dates':0,'pending':sum(r['labels'][key]['status']=='pending' for r in rows)}
    date_rows=defaultdict(list)
    for r in ok:date_rows[r['day']].append(r)
    daily=[]
    for day,rs in sorted(date_rows.items()):
        y=np.array([r['labels'][key]['y'] for r in rs]);p=np.array([r['p'][key] for r in rs]);b=np.array([r['baseline'][key] for r in rs]);g=np.array([r['global_baseline'][key] for r in rs])
        daily.append([day,float(np.mean((p-y)**2)),float(np.mean((b-y)**2)),float(np.mean((g-y)**2)),float(np.mean(-(y*np.log(np.clip(p,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-p,1e-7,1-1e-7)))) )])
    a=np.array([r[1:] for r in daily]);bss=1-a[:,0].mean()/a[:,1].mean();block=max(20,TARGETS[key]['horizon']);ci=None
    if len(a)>=2*block:
        rng=np.random.default_rng(618+block);draws=[]
        for _ in range(400):
            starts=rng.integers(0,len(a)-block+1,size=math.ceil(len(a)/block));ix=(starts[:,None]+np.arange(block)).reshape(-1)[:len(a)]
            z=a[ix];draws.append(1-z[:,0].mean()/z[:,1].mean())
        ci=np.quantile(draws,[.1,.9]).tolist()
    # Date-equal reliability bins, with raw stock-date counts also visible.
    weights={d:1/(len(date_rows)*len(rs)) for d,rs in date_rows.items()};bins=[]
    for bin_id,lo in enumerate([0,.2,.4,.6,.8]):
        rs=[r for r in ok if min(4,int(r['p'][key]*5))==bin_id]
        if not rs:continue
        w=np.array([weights[r['day']] for r in rs]);w/=w.sum()
        bins.append({'lo':lo,'hi':lo+.2,'n':len(rs),'dates':len({r['day'] for r in rs}),
            'forecast':float(np.sum(w*np.array([r['p'][key] for r in rs]))),
            'observed':float(np.sum(w*np.array([r['labels'][key]['y'] for r in rs])))})
    return {'n':len(ok),'dates':len(daily),'positive':sum(r['labels'][key]['y'] for r in ok),'first':daily[0][0],'last':daily[-1][0],
         'brier':float(a[:,0].mean()),'stock_baseline_brier':float(a[:,1].mean()),'global_baseline_brier':float(a[:,2].mean()),
         'brier_skill':float(bss),'skill_interval':ci,'log_loss':float(a[:,3].mean()),'block_days':block,
         'nonoverlap_windows':len(a)//TARGETS[key]['horizon'],'reliability':bins,
         'pending':sum(r['labels'][key]['status']=='pending' for r in rows),'missing':sum(r['labels'][key]['status']=='missing_label_sessions' for r in rows),
         'status':'historical_increment_only' if ci and ci[0]>0 and a[:,0].mean()<a[:,2].mean() else 'not_demonstrated'}

def fundamental(code,screen,quality):
    s=screen.get(code,{});q=quality.get(code,{})
    f=s.get('financial_metrics',{});derived=f.get('derived',{});metrics=f.get('metrics',{})
    v=lambda k: (derived.get(k) or {}).get('value')
    risks=q.get('risks',[]);latest=s.get('financial_freshness',{})
    leader={'status':'待核验','segment':s.get('industry'),'source':None,'basis':'尚无细分市场份额、竞品集合与排名核验；不把评级或股价当头部'}
    if code=='US.SNPS':leader={'status':'官方自述头部','segment':'EDA','source':'https://www.synopsys.com/silicon-design.html','observed_at':'2026-10-08',
        'basis':'公司官网自述#1 EDA提供商；本次核对，缺独立同口径市场份额'}
    facts={'revenue_yoy':v('revenue_yoy'),'fcf_margin':v('fcf_margin'),'net_margin':v('net_margin'),'cash_runway_months':v('cash_runway_months')}
    if not s:return {'status':'未覆盖','facts':facts,'leader':leader,'risks':[],'sources':[],'grade':None}
    good=facts['fcf_margin'] is not None and facts['fcf_margin']>0 and facts['net_margin'] is not None and facts['net_margin']>0
    return {'status':'缓存财务支持盈利与现金流' if good else '亏损／现金流或证据需审查','facts':facts,
        'as_of':s.get('as_of'),'freshness':latest,'grade':q.get('grade'),'screen_grade':s.get('grade'),
        'grade_type':'正式核查' if q.get('grade') else '研究初评','leader':leader,'risks':risks,
        'missing':q.get('missing',[])[:6],'sources':q.get('sources',[])[:3],
        'training_included':False,'long_eligibility':'待头部与估值核验' if good else '待盈利/现金流、头部与估值核验'}

def main():
    acquisition=json.loads((OUT/'acquisition.json').read_text());member=json.loads((OUT/'membership.json').read_text());cut=acquisition['requested_end']
    sessions=calendar();names={r['code']:r for r in member['members']};records=[];samples=[];history=[];prices={};inputs=[]
    screen_path=ROOT/'analysis/ai_value_chain_map_v1/quality/screen_current.json';quality_path=ROOT/'analysis/ai_value_chain_map_v1/quality/current.json'
    screen=json.loads(screen_path.read_text());quality=json.loads(quality_path.read_text())
    for code,item in names.items():
        meta=acquisition['datasets'].get(code);bars=[]
        if meta:
            p=ROOT/meta['path']
            if sha(p)!=meta['sha256']:raise ValueError('Input hash mismatch: '+code)
            inputs.append({'code':code,'path':meta['path'],'sha256':sha(p)})
            f=pd.read_parquet(p);f['day']=f.day.astype(str).str[:10] if 'day' in f else f.time_key.astype(str).str[:10]
            f=f[(f.day>='2024-10-08')&(f.day<=cut)].sort_values('day');bars=[[str(r.day),float(r.open),float(r.high),float(r.low),float(r.close),float(r.volume)] for r in f.itertuples()]
        prices[code]=bars
        rec={**item,'coverage':meta,'latest_features':None,'feature_status':'no_price_data','p':{},'ranges':{},'cadence':'覆盖不足',
             'fundamental':fundamental(code,screen['records'],quality['records']), 'year':None,'history_status':'no_history'}
        if bars:
            c=np.array([r[4] for r in bars]);recent=turns(c[-21:]);times=[b['i'] for b in recent]
            rec['recent']={'turns20':len(recent),'median_leg_days':float(np.median(np.diff(times))) if len(times)>1 else None,
                           'return20':float(c[-1]/c[-21]-1) if len(c)>20 else None,'return60':float(c[-1]/c[-61]-1) if len(c)>60 else None}
            if len(c)>=253:
                a=np.diff(np.log(c[-253:]));pos=np.maximum(a,0);rec['year']={'return':float(c[-1]/c[-253]-1),
                     'best20':float(max(c[i]/c[i-20]-1 for i in range(len(c)-253+20,len(c)))),
                     'positive_energy_top10':float(np.sort(pos)[-26:].sum()/pos.sum()) if pos.sum()>0 else None,
                     'bars':252,'first':bars[-253][0],'last':bars[-1][0]}
        in_domain=item['kind']=='STOCK' and code.startswith('US.')
        rec['in_domain']=in_domain
        if not in_domain:
            rec['fundamental']={'status':'基金/非模型域；公司财务不套用','facts':{k:None for k in ['revenue_yoy','fcf_margin','net_margin','cash_runway_months']},
                'leader':{'status':'公司头部指标不适用','segment':None,'source':None,'basis':'需另查基金持仓、费率、复位周期及产品条款'},
                'risks':[],'missing':['产品结构尚未逐项核验'],'sources':[],'long_eligibility':'产品条款待审核'}
        if not in_domain:rec['feature_status']='非美国公司股票训练域；保留真实价格，另需产品结构模型'
        elif bars:
            for idx,b in enumerate(bars):
                if b[0] not in sessions:continue
                x,status=feature(bars,idx,sessions)
                if idx==len(bars)-1:rec['latest_features']=finite(x) if x is not None else None;rec['feature_status']=status
                if x is None or b[0]<'2025-05-01':continue
                ls={k:label(bars,idx,k,sessions,cut) for k in TARGETS}
                row={'code':code,'day':b[0],'x':finite(x),'labels':ls,'p':{},'baseline':{},'global_baseline':{}}
                samples.append(row)
                if b[0]>='2026-06-01':history.append(row)
            rec['history_status']='historical_reconstruction'
        records.append(rec)
    trained={};details={};rng=np.random.default_rng(731)
    for key,config in TARGETS.items():
        train=[r for r in samples if r['day']<='2025-10-31' and r['labels'][key]['status']=='mature' and r['labels'][key]['maturity']<'2026-02-02']
        cal=[r for r in samples if '2026-02-02'<=r['day']<='2026-03-02' and r['labels'][key]['status']=='mature' and r['labels'][key]['maturity']<'2026-06-01']
        td=sorted({r['day'] for r in train});cd=sorted({r['day'] for r in cal});positive=sum(r['labels'][key]['y'] for r in cal)
        details[key]={'training_n':len(train),'training_dates':len(td),'training_range':[min(td),max(td)] if td else [],
             'calibration_n':len(cal),'calibration_dates':len(cd),'calibration_positive':positive,'calibration_range':[min(cd),max(cd)] if cd else [],
             'max_training_label_end':max((r['labels'][key]['maturity'] for r in train),default=None),
             'max_calibration_label_end':max((r['labels'][key]['maturity'] for r in cal),default=None)}
        if not train or not cal:continue
        global_prior=(positive+.5)/(len(cal)+1);per=defaultdict(list)
        for r in cal:per[r['code']].append(r['labels'][key]['y'])
        prior={code:(sum(v)+20*global_prior)/(len(v)+20) for code,v in per.items()}
        if min(positive,len(cal)-positive)<10 or len({r['labels'][key]['y'] for r in train})<2:
            m={'constant':global_prior,'status':'insufficient_calibration_prior_only'}
        else:
            m=fit([r['x'] for r in train],[r['labels'][key]['y'] for r in train],dateweights(train))
            calibrate(m,[r['x'] for r in cal],[r['labels'][key]['y'] for r in cal],dateweights(cal));m['status']='fitted'
        m['calibration_global_prior']=global_prior;m['stock_baseline']=prior;trained[key]=m;details[key]['model_status']=m['status']
        pred=predict(m,[r['x'] for r in history])
        for r,p in zip(history,pred):r['p'][key]=float(p);r['baseline'][key]=prior.get(r['code'],global_prior);r['global_baseline'][key]=global_prior
        current=[r for r in records if r['latest_features'] is not None]
        current_p=predict(m,[r['latest_features'] for r in current])
        # Refits use full contemporaneous date blocks; the calibrator remains fixed.
        boot=[]
        if m.get('status')=='fitted' and len(td)>=40:
            byday={d:[r for r in train if r['day']==d] for d in td}
            for rep in range(32):
                starts=rng.integers(0,len(td)-19,size=math.ceil(len(td)/20));ix=(starts[:,None]+np.arange(20)).reshape(-1)[:len(td)]
                br=[r for i in ix for r in byday[td[i]]]
                if len({r['labels'][key]['y'] for r in br})<2:continue
                # Repeated sampled dates must retain their multiplicity in the bootstrap.
                bw=np.array([1/len(byday[r['day']]) for r in br]);bw*=len(br)/bw.sum()
                bm=fit([r['x'] for r in br],[r['labels'][key]['y'] for r in br],bw);bm['calibrator']=m['calibrator']
                boot.append(predict(bm,[r['latest_features'] for r in current]))
        intervals=np.quantile(boot,[.1,.9],axis=0) if boot else None
        for i,(rec,p) in enumerate(zip(current,current_p)):
            rec['p'][key]=float(p);rec['ranges'][key]=[float(intervals[0,i]),float(intervals[1,i])] if intervals is not None else None
        print(json.dumps({'target':key,**details[key]}),flush=True)
    for r in records:
        r['cadence']=cadence(r['p']);r['validation']={k:score([s for s in history if s['code']==r['code']],k) for k in TARGETS}
        if r['latest_features'] is not None and 'hold' in trained and trained['hold'].get('coef'):
            m=trained['hold'];z=transform(m,r['latest_features'])[0];a=z*np.array(m['coef'])*m['calibrator']['slope'];fn=FEATURES+[k+'_missing' for k in FEATURES]
            r['hold_contributions']=sorted([{'feature':k,'logit_contribution':float(v)} for k,v in zip(fn,a)],key=lambda x:abs(x['logit_contribution']),reverse=True)[:8]
    metrics={k:score(history,k) for k in TARGETS}
    model_doc={'version':VERSION,'features':FEATURES,'targets':TARGETS,'models':trained,'fit_details':details}
    save('models.json',model_doc);save('evaluation.json',{'version':VERSION,'as_of':cut,'rows':history,'metrics':metrics})
    hk=json.loads((ROOT/'analysis/ai_trend_quadrant_v1/hk_calendar_2026.json').read_text())
    save('prices.json',{'as_of':cut,'requested_start':'2024-10-08','records':prices,'us_sessions':sessions,'us_early_close':EARLY,'hk_calendar':hk,
       'sources':['https://www.nyse.com/publicdocs/ICE_NYSE_2024_Yearly_Trading_Calendar.pdf','https://www.nyse.com/publicdocs/ICE_NYSE_2025_Yearly_Trading_Calendar.pdf',
       'https://www.nyse.com/publicdocs/nyse/ICE_NYSE_2026_Yearly_Trading_Calendar.pdf','https://ir.nasdaq.com/news-releases/news-release-details/nasdaq-announces-closure-its-us-markets-honor-national-day-0',
       'https://www.nyse.com/trade/hours-calendars']})
    report={'version':VERSION,'as_of':cut,'member_observed_at':member['observed_at'],'generated_at':datetime.now(timezone.utc).isoformat(),
       'features':FEATURES,'targets':TARGETS,'records':records,'metrics':metrics,'fit_details':details,
       'summary':{'members':len(records),'stocks':sum(r['kind']=='STOCK' for r in records),'etfs':sum(r['kind']=='ETF' for r in records),
         'scored':sum(bool(r['p']) for r in records),'current_prices':sum(r.get('coverage',{}).get('actual_end')==cut for r in records if r.get('coverage')),
         'formal_quality':sum(bool(r['fundamental'].get('grade')) for r in records),'leader_independently_verified':0},
       'inputs':inputs+[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in [screen_path,quality_path,OUT/'membership.json',OUT/'PROTOCOL.md',OUT/'model.py',OUT/'build.py']],
       'model_sha256':sha(OUT/'models.json'),'independent_validation':False,'cadence_status':'exploratory_only'}
    report['run_id']='cadence_'+hashlib.sha256(json.dumps({'inputs':report['inputs'],'as_of':cut},sort_keys=True).encode()).hexdigest()[:16]
    save('results.json',report);print(json.dumps({'summary':report['summary'],'metrics':metrics},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
