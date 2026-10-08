"""Bounded, immutable Local -> R2 -> OpenD preparation. No trading or uploads."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, subprocess, sys, time
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def publish(d):
    p = OUT / 'acquisition.json'; t = p.with_suffix('.tmp')
    t.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n'); t.replace(p)

def worker(end, budget):
    start = time.monotonic(); deadline = start + budget
    members = json.loads((OUT / 'membership.json').read_text())['members']
    d = {'version':'cadence_inputs_v1','requested_start':'2024-10-08','requested_end':end,
         'started_at':datetime.now(timezone.utc).isoformat(),'datasets':{},'r2_checks':[],
         'no_upload':True,'budget_seconds':budget,'old_inputs':[],'errors':[]}
    candidates = {}
    pointers = [ROOT / '.cache/stock_data_v1/current.json',
                ROOT / '.cache/stock_traits_daily_v1/ai_basket_AplusA/data/current.json']
    for p in pointers:
        if not p.exists(): continue
        ptr = json.loads(p.read_text()); ref = ptr.get('manifest') or ptr.get('manifest_path')
        mp = ROOT / ref
        if not mp.exists(): mp = p.parent / ref
        expected = ptr.get('sha256') or ptr.get('manifest_sha256')
        if expected and sha(mp) != expected: raise ValueError('Manifest hash mismatch')
        d['old_inputs'].append({'path':str(mp.relative_to(ROOT)),'sha256':sha(mp)})
        m = json.loads(mp.read_text()); ds = m.get('datasets',[])
        if isinstance(ds,dict): ds = list(ds.values())
        for r in ds:
            path = r.get('path') or r.get('local_path')
            if path and '/daily/' in path and path.endswith('.parquet'):
                candidates.setdefault(Path(path).stem, []).append(r)
    base = json.loads((ROOT / 'analysis/ai_trend_quadrant_v2/results.json').read_text())
    for r in base['provenance']:
        candidates.setdefault(r['code'], []).append({**r,'source':'local_v2_pinned','price_basis':'QFQ'})
    # Other previous captures are local candidates only; choose one complete series, never splice.
    for item in members:
        code = item['code']
        for p in (ROOT / 'market_data/stock_data_v1/captures').glob('*/daily/' + code + '.parquet'):
            candidates.setdefault(code, []).append({'path':str(p.relative_to(ROOT)),'source':'local_immutable_capture','price_basis':'QFQ'})
    loaded = {}; pending = []
    for item in members:
        code = item['code']; best = None
        for r in candidates.get(code,[]):
            p = ROOT / r['path']
            if not p.exists(): continue
            expected = r.get('sha256') or r.get('file_sha256')
            if expected and sha(p) != expected: raise ValueError('Input changed: ' + str(p))
            f = pd.read_parquet(p)
            if 'day' not in f: f['day'] = f['time_key'].astype(str).str[:10] if 'time_key' in f else f['session_date'].astype(str)
            f['day'] = f.day.astype(str).str[:10]
            f = f[(f.day >= '2024-10-08') & (f.day <= end)].sort_values('day').drop_duplicates('day')
            listing = item.get('listing_date','')
            if listing > '1970-01-01': f = f[f.day >= listing]
            if f.empty: continue
            rank = (f.day.iloc[-1], len(f))
            if best is None or rank > best[0]: best = (rank,f,r,p)
        if best:
            _,f,r,p = best; loaded[code] = (f,r,p)
            d['datasets'][code] = {'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source':r.get('source'),
                 'price_basis':r.get('price_basis','QFQ'),'actual_start':str(f.day.iloc[0]),'actual_end':str(f.day.iloc[-1]),
                 'bars':len(f),'status':'current' if str(f.day.iloc[-1]) == end else 'stale',
                 'received_at':r.get('received_at'),'checked_at':datetime.now(timezone.utc).isoformat()}
            if str(f.day.iloc[-1]) == end: continue
        pending.append(code)
    publish(d)
    # A failed R2 config is explicit. Never copy credentials into the manifest/log.
    from scripts.r2_client import R2Client
    client = R2Client(timeout=5, deadline=min(deadline,time.monotonic()+15))
    try:
        remote = {r['key']:r for r in client.list_objects('market_data/operation_cadence_v1/')}
        d['r2_checks'].append({'prefix':'market_data/operation_cadence_v1/','objects':len(remote),'status':'read'})
    except Exception as e:
        remote={}; d['r2_checks'].append({'status':'failed','error':type(e).__name__})
    # Dedicated legacy daily files are also checked before OpenD; no hourly/daily splicing.
    try:
        legacy = {r['key']:r for r in client.list_objects('market_data/trend_quadrant_v1/daily/')}
        remote.update(legacy); d['r2_checks'].append({'prefix':'market_data/trend_quadrant_v1/daily/','objects':len(legacy),'status':'read'})
    except Exception as e: d['r2_checks'].append({'status':'failed','error':type(e).__name__})
    publish(d)
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    q = ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    capture = 'cadence_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = ROOT / 'market_data/operation_cadence_v1' / capture / 'daily'; dest.mkdir(parents=True,exist_ok=True)
    try:
        for i,code in enumerate(pending):
            if time.monotonic()+2 > deadline:
                d['errors'].append({'code':code,'reason':'budget_exhausted'}); continue
            began = time.monotonic()
            try:
                # Existing R2 files are reused only as a whole verified daily series.
                keys = [k for k in remote if k.endswith('/' + code + '.parquet')]
                for key in keys[:1]:
                    rp = dest / (code + '.r2.parquet')
                    client.get_object(key,rp)
                    rf=pd.read_parquet(rp); rf['day']=rf.time_key.astype(str).str[:10] if 'day' not in rf else rf.day.astype(str)
                    if len(rf) and str(rf.day.max()) >= end:
                        d['datasets'][code]={'path':str(rp.relative_to(ROOT)),'sha256':sha(rp),'source':'r2_whole_daily',
                            'price_basis':'dedicated_QFQ','actual_start':str(rf.day.min()),'actual_end':str(rf.day.max()),'bars':len(rf),'status':'current'}
                        break
                if d['datasets'].get(code,{}).get('status') == 'current': continue
                ret,f,key=q.request_history_kline(code,start='2024-10-08',end=end,ktype=ft.KLType.K_DAY,
                    autype=ft.AuType.QFQ,max_count=1000)
                if ret!=ft.RET_OK: raise RuntimeError(str(f)[:160])
                f['day']=f.time_key.astype(str).str[:10]; f=f[f.day<=end].sort_values('day').drop_duplicates('day')
                item=next(x for x in members if x['code']==code)
                if item.get('listing_date','') > '1970-01-01': f=f[f.day>=item['listing_date']]
                if f.empty: raise ValueError('empty_series')
                px=f[['open','high','low','close']].to_numpy(float)
                if not np.isfinite(px).all() or (px<=0).any() or (px[:,1]<px[:,[0,2,3]].max(axis=1)).any() or (px[:,2]>px[:,[0,1,3]].min(axis=1)).any():
                    raise ValueError('invalid_ohlc')
                p=dest/(code+'.parquet'); temp=p.with_suffix('.tmp');f.to_parquet(temp,compression='zstd',compression_level=7,index=False);temp.replace(p)
                d['datasets'][code]={'path':str(p.relative_to(ROOT)),'sha256':sha(p),'source':'OpenD_daily_QFQ_full_capture',
                    'price_basis':'single_full_series_QFQ','actual_start':str(f.day.min()),'actual_end':str(f.day.max()),
                    'bars':len(f),'status':'current' if str(f.day.max())==end else 'stale',
                    'received_at':datetime.now(timezone.utc).isoformat(),'available_at':None,'capture':capture}
            except Exception as e:
                d['errors'].append({'code':code,'reason':type(e).__name__,'detail':str(e)[:170]})
            publish(d)
            if i%10==0: print(json.dumps({'progress':i+1,'total':len(pending),'code':code,'status':d['datasets'].get(code,{}).get('status','missing')}),flush=True)
            time.sleep(max(0,1.2-(time.monotonic()-began)))
    finally: q.close()
    d.update(finished_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.monotonic()-start,2),
             known_provider_cost=None,provider_cost_status='not_reported')
    publish(d); print(json.dumps({'datasets':len(d['datasets']),'errors':len(d['errors']),'seconds':d['elapsed_seconds']}))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--as-of',default='2026-10-07');p.add_argument('--budget',type=float,default=180);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker: worker(a.as_of,a.budget)
    else:
        try: subprocess.run([sys.executable,__file__,'--worker','--as-of',a.as_of,'--budget',str(a.budget)],timeout=a.budget+5,check=True)
        except subprocess.TimeoutExpired:
            d=json.loads((OUT/'acquisition.json').read_text());d['hard_timeout']=True;publish(d)
            print('Hard timeout: verified old/new files preserved.')
