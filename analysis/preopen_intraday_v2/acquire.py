"""Isolated, resumable ALL/NONE 5m supplementation; never uploads or orders."""
from __future__ import annotations
import argparse, fcntl, json, sys, time
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.r2_client import R2Client
from scripts.backfill_model_history import normalize

def verified_month(symbol,start,end,paths):
    frames=[];rejected=[]
    for path in paths:
        if not path.exists():continue
        f=pd.read_parquet(path)
        if not {'start_at_et','end_at_et','price_basis'}.issubset(f.columns) or set(f.price_basis.dropna())!={'NONE'}:
            rejected.append(str(path.relative_to(ROOT)));continue
        f['s']=pd.to_datetime(f.start_at_et,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        f['e']=pd.to_datetime(f.end_at_et,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
        frames.append(f)
    if not frames:return None,{'rejected_unknown_schema_or_basis':rejected,'complete_days':0}
    f=pd.concat(frames,ignore_index=True).drop_duplicates('s',keep='last').sort_values('s')
    prices=f[['open','high','low','close']].to_numpy(float);valid=np.isfinite(prices).all(axis=1)&(prices>0).all(axis=1)&(f.high>=prices.max(axis=1))&(f.low<=prices.min(axis=1))&np.isfinite(f.volume)&(f.volume>=0)&(f.e-f.s==pd.Timedelta(minutes=5));f=f[valid]
    sessions=json.loads((ROOT/'market_data/model_training_history_v1/calendar.json').read_text())['sessions'];complete=0;expected=0
    for session in sessions:
        d=session['session_date']
        if not start<=d<=end:continue
        expected+=1;day=pd.Timestamp(d);r=f[(f.s>=day+pd.Timedelta(minutes=570))&(f.s<day+pd.Timedelta(minutes=570+session['duration_minutes']))]
        p=f[(f.s>=day+pd.Timedelta(hours=4))&(f.e<=day+pd.Timedelta(minutes=565))];n=f[(f.s>=day-pd.Timedelta(hours=4))&(f.e<=day+pd.Timedelta(hours=4))]
        slots=pd.date_range(day+pd.Timedelta(minutes=570),periods=session['duration_minutes']//5,freq='5min')
        if len(r)==len(slots) and np.array_equal(r.s.to_numpy(),slots.to_numpy()) and len(p)>=50 and p.e.max()==day+pd.Timedelta(minutes=565) and len(n)>=6:complete+=1
    info={'complete_days':complete,'expected_days':expected,'rejected_unknown_schema_or_basis':rejected}
    if expected and complete==expected:
        return f[(f.s>=pd.Timestamp(start)-pd.Timedelta(days=1))&(f.s<pd.Timestamp(end)+pd.Timedelta(days=1))].drop(columns=['s','e']),info
    return None,info

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--symbols',nargs='+',default=['QQQ']);ap.add_argument('--favorites',action='store_true');ap.add_argument('--start',default='2026-01-01');ap.add_argument('--end',default='2026-10-02');args=ap.parse_args()
    if args.favorites:
        u=json.loads((OUT/'universe.json').read_text());args.symbols=sorted(s for s,m in u['members'].items() if '特别关注' in m['groups'] and m.get('stock_type')=='STOCK')
    client=R2Client()
    # Explicit local -> R2 inventory; remote errors are preserved, never silently ignored.
    try: inventory={o['key']:o for o in client.list_objects('us_5m/')};r2_status='ok'
    except Exception as e:inventory={};r2_status=type(e).__name__+': '+str(e)[:100]
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    lock=open('/tmp/stock_futu_acquisition.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    try:
        jobs=[(s,max(args.start,str(m.start_time.date())),min(args.end,str(m.end_time.date()))) for s in args.symbols for m in pd.period_range(args.start,args.end,freq='M')]
        failures=[]
        for symbol,start,end in jobs:
            path=OUT/'cache'/symbol/f'{start}_{end}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
            empty_marker=path.with_suffix('.empty.json')
            if empty_marker.exists():continue
            if path.exists():print(json.dumps({'symbol':symbol,'source':'local_cache'}),flush=True);continue
            prior=str((pd.Period(start,freq='M')-1));local_paths=list((OUT/'cache'/symbol).glob(f'{start[:7]}_*.parquet'))+list((OUT/'cache'/symbol).glob(f'{prior}_*.parquet'))+list((ROOT/'market_data/model_training_history_v1/parts'/symbol).glob(f'{start[:7]}/bars.parquet'))+list((ROOT/'market_data/model_training_history_v1/parts'/symbol).glob(f'{prior}/bars.parquet'))
            for year in {start[:4],prior[:4]}:
                local_paths += [ROOT/'market_data/model_training_history_v1/us_5m'/symbol/(year+'.parquet'),ROOT/'market_data/us_5m'/symbol/(year+'.parquet')]
            norm,quality=verified_month(symbol,start,end,local_paths)
            if norm is not None:
                norm.to_parquet(path,index=False,compression='zstd',compression_level=7);print(json.dumps({'symbol':symbol,'source':'verified_local','quality':quality}),flush=True);continue
            remotes=[key for key in inventory if key.startswith(f'us_5m/{symbol}/')]
            for key in remotes:
                dest=OUT/'r2_cache'/key;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():client.get_object(key,dest)
            remote_paths=[OUT/'r2_cache'/key for key in remotes];norm,quality=verified_month(symbol,start,end,local_paths+remote_paths)
            if norm is not None:
                norm.to_parquet(path,index=False,compression='zstd',compression_level=7);print(json.dumps({'symbol':symbol,'source':'verified_R2','quality':quality}),flush=True);continue
            print(json.dumps({'symbol':symbol,'cache_quality':quality,'r2':r2_status,'remote_files':len(remotes),'fallback':'OpenD: joint night/pre completeness not established by remote date bounds'}),flush=True)
            pages=[];page_key=None
            checkpoint=path.with_suffix('.partial.parquet')
            resume=path.with_suffix('.resume.json')
            if checkpoint.exists() and resume.exists():
                pages=[pd.read_parquet(checkpoint)]; saved=json.loads(resume.read_text());page_key=bytes.fromhex(saved['key']) if saved['key'] else None
            count=0
            while True:
                ret,f,next_key=q.request_history_kline('US.'+symbol,start=start,end=end,ktype=ft.KLType.K_5M,
                    autype=ft.AuType.NONE,fields=ft.KL_FIELD.ALL,max_count=1000,page_req_key=page_key,extended_time=True,session=ft.Session.ALL)
                if ret!=ft.RET_OK:
                    failures.append({'symbol':symbol,'start':start,'end':end,'error':str(f)[:300]})
                    (OUT/'acquisition_failures.json').write_text(json.dumps(failures,indent=2))
                    print(json.dumps(failures[-1]),flush=True)
                    break
                pages.append(f);count+=1
                if count%20==0:
                    pd.concat(pages,ignore_index=True).to_parquet(checkpoint,index=False,compression='zstd',compression_level=7)
                    resume.write_text(json.dumps({'key':next_key.hex() if next_key else None}))
                    print(json.dumps({'symbol':symbol,'pages':count,'rows':sum(len(f) for f in pages)}),flush=True)
                page_key=next_key
                if not page_key:break
                time.sleep(.25)
            if ret!=ft.RET_OK:
                if len(failures)>=3 and all('TimeOut' in x['error'] for x in failures[-3:]):break
                continue
            raw=pd.concat(pages,ignore_index=True)
            if raw.empty:
                empty_marker.write_text(json.dumps({'symbol':symbol,'start':start,'end':end,'status':'source_returned_no_bars','source':'Futu ALL NONE 5m'}))
                print(json.dumps({'symbol':symbol,'start':start,'status':'empty'}),flush=True);time.sleep(1.2);continue
            raw=raw.drop_duplicates('time_key',keep='last').sort_values('time_key')
            norm=normalize(raw,symbol)
            temp=path.with_suffix('.writing');norm.to_parquet(temp,index=False,compression='zstd',compression_level=7);temp.replace(path)
            checkpoint.unlink(missing_ok=True);resume.unlink(missing_ok=True)
            print(json.dumps({'symbol':symbol,'status':'saved','rows':len(norm),'first':norm.time_key.min(),'last':norm.time_key.max()}),flush=True)
            time.sleep(1.2)
    finally:q.close();lock.close()

if __name__=='__main__':main()
