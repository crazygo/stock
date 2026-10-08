"""Separate full-session 60m audit view. Does not affect daily models or frozen inputs."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json,sys,time,subprocess,hashlib
import pandas as pd

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
START='2026-06-01';END='2026-10-07'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def publish(d):
    p=OUT/'intraday.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,separators=(',',':')));t.replace(p)
def projection(f):
    rows=[]
    for r in f.sort_values('time_key').drop_duplicates('time_key').itertuples():
        ts=str(r.time_key);end_dt=datetime.fromisoformat(ts)
        # This capture's time_key is a completion time, verified with 1m/daily opens.
        dt=end_dt-timedelta(seconds=1);hh=dt.hour+dt.minute/60+dt.second/3600
        session_day=dt.date()
        if hh>=20:
            session_day+=timedelta(days=1)
            # Only actual provider bars are retained; Sunday overnight belongs to Monday.
        day=session_day.isoformat()
        if not START<=day<=END:continue
        kind='夜盘' if hh>=20 or hh<4 else '盘前' if hh<9.5 else '常规盘' if hh<16 else '盘后'
        minutes=30 if end_dt.strftime('%H:%M') in ['09:30','16:00'] else 60
        start_dt=end_dt-timedelta(minutes=minutes)
        rows.append([ts,float(r.open),float(r.high),float(r.low),float(r.close),float(r.volume),day,kind,
                    start_dt.isoformat(sep=' '),end_dt.isoformat(sep=' ')])
    return rows
def worker():
    began=time.monotonic();deadline=began+220
    members=json.loads((OUT/'membership.json').read_text())['members'];members=[r for r in members if r['code'].startswith('US.')]
    d={'requested_start':START,'requested_end':END,'granularity_minutes':60,'session':'ALL','records':{},'sources':{},'errors':[],
       'view_only':True,'model_input':False,'no_upload':True,'time_key_role':'bar_end_empirically_verified',
       'time_contract_evidence':'time_contract_probe.json; US.SNPS/AMD/MXL 1m 09:31 Open matches daily Open'}
    from scripts.r2_client import R2Client
    client=R2Client(timeout=5,deadline=min(deadline,began+30))
    try:remote={r['key']:r for r in client.list_objects('market_data/us_60m/')};d['r2_status']='read'
    except Exception as e:remote={};d['r2_status']=type(e).__name__
    capture='hourly_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=ROOT/'market_data/operation_cadence_v1'/capture/'hourly';dest.mkdir(parents=True,exist_ok=True)
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    try:
        for i,item in enumerate(members):
            code=item['code'];ticker=code[3:];local=ROOT/'market_data/us_60m'/ticker/'2026.parquet';best=None
            if local.exists():
                f=pd.read_parquet(local);best=(projection(f),local,'local_existing')
            if best and best[0] and best[0][0][6]<=START and best[0][-1][6]>=END:
                d['records'][code]=best[0];d['sources'][code]={'path':str(best[1].relative_to(ROOT)),'sha256':sha(best[1]),'source':best[2]};continue
            key='market_data/us_60m/'+ticker+'/2026.parquet'
            if key in remote and time.monotonic()<client.deadline:
                try:
                    p=dest/(code+'.r2.parquet');client.get_object(key,p);f=pd.read_parquet(p);b=projection(f)
                    if b and (best is None or b[-1][0]>best[0][-1][0]):best=(b,p,'r2')
                    if b and b[0][6]<=START and b[-1][6]>=END:
                        d['records'][code]=b;d['sources'][code]={'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source':'r2'};continue
                except Exception as e:d['errors'].append({'code':code,'stage':'r2','reason':type(e).__name__})
            if time.monotonic()+2<deadline:
                t=time.monotonic()
                try:
                    frames=[];page=None
                    while True:
                        ret,f,page=q.request_history_kline(code,start='2026-05-31',end=END,ktype=ft.KLType.K_60M,
                            autype=ft.AuType.QFQ,max_count=1000,page_req_key=page,extended_time=True,session=ft.Session.ALL)
                        if ret!=ft.RET_OK:raise RuntimeError(str(f)[:140])
                        frames.append(f)
                        if not page:break
                        if time.monotonic()+1>=deadline:raise TimeoutError('pagination_budget')
                        time.sleep(.25)
                    f=pd.concat(frames).sort_values('time_key').drop_duplicates('time_key');p=dest/(code+'.parquet');temp=p.with_suffix('.tmp')
                    f.to_parquet(temp,compression='zstd',compression_level=7,index=False);temp.replace(p)
                    best=(projection(f),p,'OpenD_QFQ_ALL_full_series')
                except Exception as e:d['errors'].append({'code':code,'stage':'opend','reason':type(e).__name__,'detail':str(e)[:140]})
                time.sleep(max(0,1.2-(time.monotonic()-t)))
            if best:
                d['records'][code]=best[0];d['sources'][code]={'path':str(best[1].relative_to(ROOT)),'sha256':sha(best[1]),'source':best[2],
                      'first':best[0][0][0] if best[0] else None,'last':best[0][-1][0] if best[0] else None,
                      'complete_extended_sessions':'not_asserted_from_endpoint_presence'}
            else:d['records'][code]=[]
            publish(d)
            if i%10==0:print(json.dumps({'hourly_progress':i+1,'total':len(members),'code':code,'bars':len(d['records'][code])}),flush=True)
    finally:q.close()
    d['elapsed_seconds']=round(time.monotonic()-began,2);publish(d)
    print(json.dumps({'hourly_members':len(d['records']),'errors':len(d['errors']),'seconds':d['elapsed_seconds']}))
if __name__=='__main__':
    if '--reproject' in sys.argv:
        d=json.loads((OUT/'intraday.json').read_text())
        for code,s in d['sources'].items():
            p=ROOT/s['path']
            if sha(p)!=s['sha256']:raise ValueError('Changed hourly input: '+code)
            d['records'][code]=projection(pd.read_parquet(p))
            s.update(first=d['records'][code][0][0],last=d['records'][code][-1][0])
        d['time_key_role']='bar_end_empirically_verified';d['time_contract_evidence']='time_contract_probe.json'
        publish(d);print(json.dumps({'reprojected':len(d['records'])}))
    elif '--worker' in sys.argv:worker()
    else:
        try:subprocess.run([sys.executable,__file__,'--worker'],timeout=225,check=True)
        except subprocess.TimeoutExpired:print('Hourly budget reached; published verified bars retained.')
