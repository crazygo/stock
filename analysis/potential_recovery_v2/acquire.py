"""Bounded immutable daily capture for recovery research; no uploads or trading."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, re, shutil, subprocess, sys, time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.model_history_calendar import calendar
END='2026-10-06'; START='2024-10-07'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def valid(f):
    f=f.copy();f['day']=f.time_key.astype(str).str[:10];f=f[(f.day>=START)&(f.day<=END)].sort_values('day')
    if f.day.duplicated().any():raise ValueError('duplicate_sessions')
    a=f[['open','high','low','close']].to_numpy(float)
    ok=np.isfinite(a).all(axis=1)&(a>0).all(axis=1)&(a[:,1]>=np.maximum(a[:,0],a[:,3]))&(a[:,2]<=np.minimum(a[:,0],a[:,3]))&(a[:,1]>=a[:,2])
    if not ok.all():raise ValueError('invalid_OHLC')
    return f
def worker(run,budget):
    begin=time.monotonic(); deadline=begin+budget
    rp=ROOT/'.cache/stock_data_v1/runs'/run;cp=ROOT/'market_data/stock_data_v1/captures'/run/'daily';cp.mkdir(parents=True,exist_ok=True)
    source=ROOT/'analysis/potential_recovery_v1/screen_results.json';s=json.loads(source.read_text())
    rows=[r for r in s['stocks'] if r['ai_grade'] in ['++','+']]
    members=[{'security_id':'US.'+r['ticker'],'issuer_id':r['cik'],'ai_grade':r['ai_grade'],'financial_tier':r['quality_tier'][0],'comparison_only':r['comparison_only']} for r in rows]
    save(rp/'members.json',members)
    html=ROOT/'analysis/potential_recovery_v1/index.html';payload=json.loads(re.search(r'<script type="application/json" id="embedded-data">(.*?)</script>',html.read_text(),re.S).group(1))
    old={r['ticker']:r for r in payload['stocks']}
    priority=[r['ticker'] for r in rows if r['quality_tier'].startswith('A') or r['ai_grade']=='++' or r['comparison_only']]
    # Former >=1.2 A comes first; baseline TTD is included to quantify exclusion.
    priority.sort(key=lambda t:(not(old[t]['quality_tier'].startswith('A') and old[t]['year_ratio']>=1.2),t))
    priority+=['QQQ','SMH']
    sessions=[s['session_date'] for s in calendar(START,END)['sessions']]
    manifest={'schema_version':'stock_data_manifest_v1','data_run_id':run,'observation_mode':'live','requested_cutoff':now(),'target_completed_session':END,'started_at':now(),'status':'partial',
      'universe':{'path':str((rp/'members.json').relative_to(ROOT)),'sha256':sha(rp/'members.json'),'observed_at':s['metadata']['generated_at'],'source':'explicit_existing_report_AI_++_+_scope','current_members_not_PIT':True},
      'requirements':{'start':START,'end':END,'warmup_closes':141,'daily':'regular_session_QFQ_USD','calendar':'official_Nasdaq','network_budget_seconds':budget,'news':'focused_official_issuer_review_not_full_pool_monitor'},'datasets':{},'changes':[],'errors':[],
      'resources':{'cache_hits':0,'OpenD_requests':0,'R2_requests':0,'retries':0,'bytes':0,'supplier_fees':None,'supplier_fees_reason':'not_reported_by_provider'}}
    def publish():save(rp/'manifest.json',manifest)
    def record(ticker,f,origin,received=None,parent=None):
        f=valid(f);p=cp/('US.'+ticker+'.parquet');tmp=p.with_suffix('.tmp');f.to_parquet(tmp,compression='zstd',compression_level=7,index=False);tmp.replace(p)
        days=f.day.tolist();missing=sorted(set(sessions)-set(days));warm=[d for d in sessions if d>=days[-141] if len(days)>=141] if len(days)>=141 else sessions
        recent_missing=sorted(set(warm)-set(days));year_missing=sorted(set(d for d in sessions if d>='2026-01-01')-set(days))
        current=bool(days and days[-1]==END and len(days)>=141 and not recent_missing and not year_missing)
        manifest['datasets']['US.'+ticker]={'dataset_id':run+':US.'+ticker,'capture':run,'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source':origin,'source_parent':parent,'price_basis':'single_full_series_QFQ','requested_start':START,'requested_end':END,'actual_start':days[0] if days else None,'actual_end':days[-1] if days else None,'complete_through':END if current else None,'missing_sessions':missing,'missing_year_sessions':year_missing,'warmup_missing_sessions':recent_missing,'bars':len(f),'status':'complete' if current else 'stale' if days and days[-1]<END else 'missing_intervals','received_at':received,'checked_at':now(),'available_at':None,'published_at':None,'source_update':'reused_cache' if origin.startswith('local') else 'new_capture','zero_volume_rows':int((f.volume<=0).sum())}
        manifest['changes'].append(manifest['datasets']['US.'+ticker]['dataset_id']);return current
    pending=[]
    previous={}
    ptr=HERE/'DATA_POINTER.json'
    if ptr.exists():
        ref=json.loads(ptr.read_text());pp=ROOT/ref['manifest']
        if pp.exists() and sha(pp)==ref['sha256']:previous=json.loads(pp.read_text())['datasets']
    tq=json.loads((ROOT/'analysis/ai_trend_quadrant_v2/results.json').read_text());known={r['symbol']:r for r in tq['records']}
    for ticker in priority:
        local=ROOT/'market_data/trend_quadrant_v1/daily'/('US.'+ticker+'.parquet');hit=False
        prior=previous.get('US.'+ticker)
        if prior:
            try:
                source=ROOT/prior['path'];current=record(ticker,pd.read_parquet(source),'local_immutable_prior_capture',prior.get('received_at'),{'path':prior['path'],'sha256':prior['sha256']})
                hit=current and not manifest['datasets']['US.'+ticker]['missing_sessions']
            except Exception as e:manifest['errors'].append({'code':ticker,'stage':'prior_capture','reason':type(e).__name__})
        if not hit and local.exists():
            try:hit=record(ticker,pd.read_parquet(local),'local_recent_QFQ',known.get(ticker,{}).get('coverage',{}).get('received_at'),{'path':str(local.relative_to(ROOT)),'sha256':sha(local)})
            except Exception as e:manifest['errors'].append({'code':ticker,'stage':'local','reason':type(e).__name__})
            hit=hit and not manifest['datasets'].get('US.'+ticker,{}).get('missing_sessions',[True])
        if hit:manifest['resources']['cache_hits']+=1
        else:pending.append(ticker)
    publish()
    from scripts.r2_client import R2Client
    client=R2Client(timeout=5,deadline=min(deadline,time.monotonic()+12))
    try:
        manifest['resources']['R2_requests']+=1;remote={r['key']:r for r in client.list_objects('market_data/trend_quadrant_v1/daily/')}
        manifest['r2_namespace']={'status':'read','objects':len(remote)}
    except Exception as e:remote={};manifest['r2_namespace']={'status':'unavailable','reason':type(e).__name__}
    open_pending=[]
    for ticker in pending:
        key='market_data/trend_quadrant_v1/daily/US.'+ticker+'.parquet';hit=False
        if key in remote and time.monotonic()<deadline:
            try:
                p=rp/'r2'/('US.'+ticker+'.parquet');manifest['resources']['R2_requests']+=1;manifest['resources']['bytes']+=client.get_object(key,p)
                current=record(ticker,pd.read_parquet(p),'R2_daily_QFQ',now(),{'key':key});hit=current and not manifest['datasets']['US.'+ticker]['missing_sessions']
            except Exception as e:manifest['errors'].append({'code':ticker,'stage':'R2','reason':type(e).__name__})
        if not hit:open_pending.append(ticker)
    publish()
    if open_pending and time.monotonic()+3<deadline:
        import futu as ft
        ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
        try:
            for i,ticker in enumerate(open_pending):
                if time.monotonic()+3>deadline:break
                t=time.monotonic();manifest['resources']['OpenD_requests']+=1
                try:
                    ret,f,key=q.request_history_kline('US.'+ticker,start=START,end=END,ktype=ft.KLType.K_DAY,autype=ft.AuType.QFQ,max_count=1000)
                    if ret!=ft.RET_OK:raise RuntimeError(str(f)[:160])
                    if key is not None:raise ValueError('unexpected_pagination')
                    record(ticker,f,'OpenD_daily_QFQ',now())
                except Exception as e:manifest['errors'].append({'code':ticker,'stage':'OpenD','reason':type(e).__name__,'detail':str(e)[:180]})
                publish();print(json.dumps({'fetched':i+1,'remaining':len(open_pending)-i-1,'ticker':ticker,'status':manifest['datasets'].get('US.'+ticker,{}).get('status','unavailable')}),flush=True)
                time.sleep(max(0,1.2-(time.monotonic()-t)))
        finally:q.close()
    # Preserve old versions for every unrefreshed member, clearly stale.
    for r in rows:
        ticker=r['ticker']
        if manifest['datasets'].get('US.'+ticker,{}).get('status')=='complete':continue
        if 'US.'+ticker not in manifest['datasets']:
            b=old[ticker]['bars'];f=pd.DataFrame(b,columns=['day','open','high','low','close','volume']);f['time_key']=f.day+' 00:00:00'
            record(ticker,f,'local_v1_frozen_QFQ',None,{'path':str(html.relative_to(ROOT)),'sha256':sha(html)})
    manifest['requirements']['network_refresh_subset']=['US.'+t for t in priority]
    manifest['resources']['elapsed_seconds']=round(time.monotonic()-begin,3);manifest['finished_at']=now();manifest['status']='complete' if all(d['status']=='complete' and not d['missing_sessions'] for d in manifest['datasets'].values()) else 'partial';publish()
    save(ROOT/'.cache/stock_data_v1/current.json',{'data_run_id':run,'manifest':str((rp/'manifest.json').relative_to(ROOT)),'sha256':sha(rp/'manifest.json')})
    save(HERE/'DATA_POINTER.json',{'data_run_id':run,'manifest':str((rp/'manifest.json').relative_to(ROOT)),'sha256':sha(rp/'manifest.json')})
    print(json.dumps({'data_run_id':run,'current':sum(d['status']=='complete' for d in manifest['datasets'].values()),'datasets':len(manifest['datasets']),'errors':manifest['errors'],'resources':manifest['resources']}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--run');p.add_argument('--budget',type=float,default=120);a=p.parse_args()
    if a.worker:return worker(a.run,a.budget)
    run='recovery_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');sub=subprocess.Popen([sys.executable,__file__,'--worker','--run',run,'--budget',str(a.budget)])
    try:code=sub.wait(timeout=a.budget+8)
    except subprocess.TimeoutExpired:
        sub.kill();sub.wait();code=124
        rp=ROOT/'.cache/stock_data_v1/runs'/run;mp=rp/'manifest.json'
        if mp.exists():
            d=json.loads(mp.read_text());d['status']='partial';d['finished_at']=now();d['errors'].append({'stage':'worker','reason':'terminated_total_budget'});save(mp,d);save(HERE/'DATA_POINTER.json',{'data_run_id':run,'manifest':str(mp.relative_to(ROOT)),'sha256':sha(mp)})
    print('worker_exit',code,flush=True)
if __name__=='__main__':main()
