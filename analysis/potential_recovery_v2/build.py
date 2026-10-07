"""Consume an immutable manifest; compute descriptors and research decisions offline."""
from pathlib import Path
from datetime import datetime,timezone,timedelta,date
from collections import Counter
import hashlib,importlib.util,json,math,sys,time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.model_history_calendar import calendar
spec=importlib.util.spec_from_file_location('five_traits',HERE/'five_traits.py');model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
START='2024-10-07';END='2026-10-06'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def descriptors(f,day,sessions):
    g=f[f.day<=day].tail(141);values=[None]*5;states=['insufficient_history']*5
    if g.empty:return values,states,None
    a=np.diff(np.log(g.close.to_numpy(float)));m=model.measure(a)[0]
    for k,n in enumerate(model.WINDOWS):
        if len(g)<n+1:continue
        segment=g.tail(n+1);expected=[d for d in sessions if segment.day.iloc[0]<=d<=day]
        if segment.day.tolist()!=expected:states[k]='missing_sessions';continue
        if not np.isfinite(m[k]):states[k]='undefined_zero_variation';continue
        values[k]=float(m[k]);states[k]='scored'
    return values,states,a
def price_metrics(f,day,sessions,benchmarks):
    g=f[f.day<=day].copy();c=g.close.to_numpy(float);out={};d=g.day.tolist()
    for n in [20,60,120]:
        expected=[x for x in sessions if len(d)>n and d[-n-1]<=x<=day]
        out['ret'+str(n)]=float(c[-1]/c[-n-1]-1) if len(c)>n and d[-n-1:]==expected else None
    if any(out[k] is None for k in ['ret20','ret60','ret120']):return out
    out.update(ma20=float(c[-20:].mean()),ma60=float(c[-60:].mean()),ma20_slope10=float(c[-20:].mean()/c[-30:-10].mean()-1))
    out['vs_ma20']=float(c[-1]/out['ma20']-1);out['vs_ma60']=float(c[-1]/out['ma60']-1)
    low_index=int(np.argmin(c[-60:]));out['days_since_60d_close_low']=59-low_index
    out['new_60d_close_lows20']=int(sum(c[i]<=c[i-59:i].min() for i in range(len(c)-20,len(c))))
    for symbol,b in benchmarks.items():
        bb=b[b.day<=day];bc=bb.close.to_numpy(float)
        for n in [20,60]:
            aligned=len(bb)>n and bb.day.tolist()[-n-1:]==d[-n-1:]
            out['relative_'+symbol+'_'+str(n)]=float(out['ret'+str(n)]-(bc[-1]/bc[-n-1]-1)) if aligned else None
    year=g[g.day>='2026-01-01'];peak=year.high.idxmax();post=g.loc[peak:];low=post.low.idxmin()
    out.update(year_peak=float(g.loc[peak,'high']),year_peak_date=str(g.loc[peak,'day']),peak_after_low=float(g.loc[low,'low']),peak_after_low_date=str(g.loc[low,'day']),reference_price=float(c[-1]))
    out['year_ratio']=out['year_peak']/c[-1];den=out['year_peak']-out['peak_after_low'];out['recovery_fraction']=float((c[-1]-out['peak_after_low'])/den) if den>0 else None;out['rebound_from_peak_after_low']=float(c[-1]/out['peak_after_low']-1)
    out['year_missing_sessions']=sorted(set(d for d in sessions if '2026-01-01'<=d<=day)-set(g.day))
    return out
def decide(r):
    p=r['price'];rv=r.get('focused_review');reasons=[]
    if r['coverage']['status']!='complete':return '行情待补',['本次完整收盘截止或必要历史不足；旧数据不冒充当下']
    if any(p.get(k) is None for k in ['ret20','ret60','ret120','relative_QQQ_20']):return '行情待补',['价格/基准窗口缺失']
    if p.get('year_missing_sessions') or not r['price_basis_clean']:return '价格待核查',['全年缺口或既有公司行动疑点未解决']
    long_down=p['ret120']<-.15 and p['ret60']<-.10 and p['vs_ma60']<0
    fresh_low=p['days_since_60d_close_low']<=5 and p['ret20']<=0
    short_turn=p['ret20']>0 and p['vs_ma20']>0 and p['ma20_slope10']>0
    confirmed=short_turn and p['days_since_60d_close_low']>5 and p['vs_ma60']>=0 and p['relative_QQQ_20']>=0
    r['trend_flags']={'long_down':long_down,'fresh_low':fresh_low,'short_turn':short_turn,'repair_confirmed':confirmed}
    if long_down:reasons.append('120/60日仍明显下行且低于MA60')
    if fresh_low:reasons.append('最近5日出现60日收盘低点，20日收益未转正')
    if rv and rv['business_state']=='repair_needed':return '业务待修复',reasons+['官方经营资料显示尚待解决的增长/执行问题']
    if fresh_low or (long_down and not short_turn):return '持续下行',reasons
    if not r['quality_tier'].startswith('A'):return '财务待核查',['原财务初筛未达到A，保留未通过项']
    if p['year_ratio']<1.2:return '空间不足',['距离旧年高点不足1.2×；并非公司质量结论']
    if p['rebound_from_peak_after_low']>.5 or p['recovery_fraction']>=.65:return '已明显恢复',['低点涨幅超过50%或已收复至少65%跌幅']
    if p['recovery_fraction']>=.35 or p['rebound_from_peak_after_low']>.35:return '修复中',['已超出本次早期恢复范围，保留作比较']
    if not rv:return '业务待核查',reasons+['当前业务/事件核查尚未完成']
    if rv['business_state']=='ai_materiality_pending':return 'AI重要性待证',['主营增长已有证据，AI商业贡献仍待证明']
    if confirmed:return '优先跟踪',reasons+['近期收益/均线/低点/相对QQQ均满足修复观察条件']
    if p['ma20_slope10']<=0:reasons.append('MA20尚未上行')
    if p['vs_ma60']<0:reasons.append('仍低于MA60')
    if p['ret20']<=0:reasons.append('20日收益未转正')
    if p['relative_QQQ_20']<0:reasons.append('20日表现仍弱于QQQ')
    return '反转待确认',reasons or ['价格修复条件尚不齐全']
def main():
    begun=time.monotonic();pointer=json.loads((HERE/'DATA_POINTER.json').read_text());mp=ROOT/pointer['manifest'];assert sha(mp)==pointer['sha256'];manifest=json.loads(mp.read_text());sessions=[s['session_date'] for s in calendar(START,END)['sessions']]
    oldpath=ROOT/'analysis/potential_recovery_v1/screen_results.json';baseline=json.loads(oldpath.read_text());business=json.loads((HERE/'business_reviews.json').read_text());evidence=json.loads((HERE/'EVIDENCE_POINTER.json').read_text());es={s['url']:s for s in evidence['records']};quality_path=HERE/'quality_reference.json';quality=json.loads(quality_path.read_text())['records']
    frames={}
    for code,d in manifest['datasets'].items():
        assert sha(ROOT/d['path'])==d['sha256'];f=pd.read_parquet(ROOT/d['path']);f['day']=f.time_key.astype(str).str[:10];frames[code]=f.sort_values('day')
    benchmark={t:frames['US.'+t] for t in ['QQQ','SMH'] if 'US.'+t in frames};records=[]
    for old in baseline['stocks']:
        if old['ai_grade'] not in ['++','+']:continue
        r=dict(old);ticker=r['ticker'];code='US.'+ticker;f=frames[code];d=manifest['datasets'][code];cut=str(f.day.iloc[-1]);r['coverage']=d;r['price']=price_metrics(f,cut,sessions,benchmark);r['v'],r['trait_status'],a=descriptors(f,cut,sessions);raw=model.boot(a,code,cut) if a is not None else [None]*5;r['uncertainty']=[raw[k] if r['trait_status'][k]=='scored' else None for k in range(5)];r['price_cutoff']=cut
        rv=business['records'].get(ticker);r['focused_review']=dict(rv) if rv else None
        if rv:r['focused_review']['sources']=[es[s['url']] for s in rv['sources']]
        qr=quality.get(code);r['formal_quality_reference']={'grade':qr.get('grade'),'as_of':qr.get('as_of'),'expires_at':qr.get('expires_at'),'status':qr.get('review_status'),'missing':qr.get('missing',[])} if qr else None
        r['research_status'],r['decision_reasons']=decide(r);r['review_previous_status']=(r.get('review') or {}).get('status');r['trait_history']=[]
        for day in sessions:
            if (date.fromisoformat(cut)-timedelta(days=59)).isoformat()<=day<=cut:
                v,status,_=descriptors(f,day,sessions);r['trait_history'].append({'day':day,'v':v,'status':status,'observation_mode':'historical_reconstruction'})
        r['financial_evidence_as_of']=old['financial_sources']['financial_snapshot_cutoff'];r['bars']=[[row.day,float(row.open),float(row.high),float(row.low),float(row.close),float(row.volume)] for row in f.itertuples()];r.update(r['price']);records.append(r)
    priority=[r['ticker'] for r in records if r['research_status']=='优先跟踪' and not r['comparison_only']];lookup={r['ticker']:r for r in records};priority.sort(key=lambda t:(lookup[t]['ai_grade']!='++',{'优先跟踪':0,'有条件优先跟踪':1,'次级跟踪':2}.get(lookup[t]['focused_review']['status'],3),t))
    counts=dict(Counter(r['research_status'] for r in records if not r['comparison_only']));scored_current=sum(r['coverage']['status']=='complete' for r in records if not r['comparison_only'])
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [mp,oldpath,HERE/'business_reviews.json',HERE/'EVIDENCE_POINTER.json',HERE/'PROTOCOL.md',Path(__file__),HERE/'five_traits.py',quality_path]};fp=json.dumps({'data':pointer['sha256'],'model':'recovery_trend_v2','inputs':inputs},sort_keys=True);run='recovery_report_'+hashlib.sha256(fp.encode()).hexdigest()[:16]
    meta={'report_run_id':run,'data_run_id':pointer['data_run_id'],'evaluation_cutoff':datetime.now(timezone.utc).isoformat(),'completed_price_cutoff':END,'observation_mode':'live_latest_completed_session','history_mode':'historical_reconstruction','model_version':'recovery_trend_v2','traits_version':'five_traits_v2','windows':model.WINDOWS,'universe_count':149,'comparison_count':1,'current_stocks':scored_current,'stale_stocks':149-scored_current,'scope_limit':'仅复核v1已归类AI++/+；501只未分类仍未覆盖；不宣称全市场无漏选','status_counts':counts,'priority':priority,'focused_review_issuers':len(business['records']),'news_monitor':'focused_issuer_disclosures_only_not_full_pool','valuation':'reasonable_value_not_estimated; old_snapshot_PE_not_refreshed','financial_snapshot':'inherited_v1_standardized_facts_not_redownloaded; focused_latest_releases_separate','inputs':inputs,'resources':{'calculation_seconds':round(time.monotonic()-begun,3),'data':manifest['resources'],'evidence':evidence['resources'],'model_api_calls':0,'paid_api_fees':None},'navigation_start':START,'navigation_end':END,'independent_prediction_effect_verified':False,'probabilities_or_orders_created':False}
    full=clean({'metadata':meta,'stocks':records,'sessions':sessions,'keys':model.KEYS,'names':model.NAMES,'benchmark_returns':{t:price_metrics(f,END,sessions,{}) for t,f in benchmark.items()}})
    rawpath=ROOT/'.cache/stock_report_refresh/runs'/run/'render_data.json';sp=HERE/'runs'/run/'snapshot.json';reused=sp.exists()
    if reused:
        snapshot=json.loads(sp.read_text());meta=snapshot['metadata']
    else:
        dump(rawpath,full);snapshot={**full,'stocks':[{k:v for k,v in r.items() if k!='bars'} for r in full['stocks']]};dump(sp,snapshot)
    dump(HERE/'results.json',snapshot);dump(HERE/'REPORT_POINTER.json',{'report_run_id':run,'data_run_id':pointer['data_run_id'],'render_data':str(rawpath.relative_to(ROOT)),'sha256':sha(rawpath),'snapshot':str(sp.relative_to(ROOT)),'snapshot_sha256':sha(sp)})
    attempts=ROOT/'.cache/stock_report_refresh/attempts.jsonl';attempts.parent.mkdir(parents=True,exist_ok=True)
    with attempts.open('a') as log:log.write(json.dumps({'attempted_at':datetime.now(timezone.utc).isoformat(),'report_run_id':run,'status':'reused_snapshot' if reused else 'published','elapsed_seconds':round(time.monotonic()-begun,3)})+'\n')
    print(json.dumps(meta,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
