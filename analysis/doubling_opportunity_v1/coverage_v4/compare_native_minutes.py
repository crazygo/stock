#!/usr/bin/env python3
"""Audit every existing registered NONE 5m cache against daily OHLC, without relabelling or fitting."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent;ROOT=PARENT.parents[1]
sys.path.insert(0,str(PARENT));from context import CACHE
sys.path.insert(0,str(HERE));from prepare_actions import load_audit
from scripts.model_history_calendar import calendar

def sha(raw):return hashlib.sha256(raw).hexdigest()
def inspect_minutes(q,t,cal,asof):
    required={'symbol','price_basis','time_key','start_at','end_at','available_at','open','high','low','close','volume'}
    if not required.issubset(q) or set(q.price_basis.dropna())!={'NONE'} or set(q.symbol.dropna())!={t}:raise ValueError('Native price basis/security/fields not verified')
    start=pd.to_datetime(q.start_at,utc=True);end=pd.to_datetime(q.end_at,utc=True);available=pd.to_datetime(q.available_at,utc=True)
    local=end.dt.tz_convert('America/New_York')
    if end.duplicated().any() or not (end-start==pd.Timedelta(minutes=5)).all() or not (available>=end).all() or not (q.time_key.astype(str)==local.dt.strftime('%Y-%m-%d %H:%M:%S')).all():
        raise ValueError('Duplicate/timestamp/bar-end/availability mismatch')
    f=q.copy();f['bar_start']=start.dt.tz_convert('America/New_York');f['day']=f.bar_start.dt.strftime('%Y-%m-%d');f['minute']=f.bar_start.dt.hour*60+f.bar_start.dt.minute
    sessions={x['session_date']:x for x in cal['sessions'] if x['session_date']<=asof};rows=[]
    for d,g in f[f.day.isin(sessions)].groupby('day',sort=True):
        s=sessions[d];opening=pd.Timestamp(s['open_at']).tz_convert('America/New_York');closing=pd.Timestamp(s['close_at']).tz_convert('America/New_York')
        a=opening.hour*60+opening.minute;b=closing.hour*60+closing.minute;g=g[(g.minute>=a)&(g.minute<b)].sort_values('bar_start')
        expected=np.arange(a,b,5);vals=g[['open','high','low','close','volume']].to_numpy(float)
        valid=(len(g)==len(expected) and np.array_equal(g.minute.to_numpy(),expected) and np.isfinite(vals).all() and
               (vals[:,:4]>0).all() and (vals[:,4]>=0).all() and (g.high>=g[['open','low','close']].max(axis=1)).all() and (g.low<=g[['open','high','close']].min(axis=1)).all())
        record={'ticker':t,'date':d,'expected_5m':len(expected),'actual_5m':len(g),'complete_rth_prefix':bool(valid)}
        if valid:record.update(minute_open=float(g.open.iloc[0]),minute_high=float(g.high.max()),minute_low=float(g.low.min()),minute_close=float(g.close.iloc[-1]),minute_volume=float(g.volume.sum()))
        rows.append(record)
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--wait',action='store_true');args=ap.parse_args();deadline=time.monotonic()+240*60
    lock=(CACHE/'native_minute_audit_v4.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    while True:
        p=HERE/'ACTION_AUDIT_READY.json'
        current=json.loads((CACHE/'futu_daily_v4_latest.json').read_text())
        ready=json.loads(p.read_text()) if p.exists() else {}
        if ready.get('ready_for_first_training') and ready.get('source_snapshot_path')==current['path']:break
        if not args.wait:raise RuntimeError('Full audited native source is not ready')
        if time.monotonic()>deadline:raise TimeoutError('Source audit wait expired')
        time.sleep(45)
    pointer=json.loads((CACHE/'futu_daily_v4_latest.json').read_text());audit,ready=load_audit(pointer['path'],validate_sources=True)
    asof=audit['asof'];cal=calendar('2023-01-01',asof);sources=[];snapshots=Path(pointer['path'])/'native_minute_snapshots';snapshots.mkdir(exist_ok=True)
    # Enumerate the entire available local scope before examining any prices/results.
    files=[(t,ROOT/'market_data/us_5m'/t/'2026.parquet') for t in sorted(audit['registry']) if (ROOT/'market_data/us_5m'/t/'2026.parquet').exists()]
    captured=[]
    for t,p in files:
        raw=p.read_bytes();digest=sha(raw);snapshot=snapshots/(t+'_'+digest[:16]+'.parquet')
        if not snapshot.exists():snapshot.write_bytes(raw)
        captured.append({'ticker':t,'original_path':str(p.resolve()),'snapshot_path':str(snapshot.resolve()),'sha256':digest})
    registration={'source_audit':ready,'scope':'all existing local registered NONE 5m sources; neither full market nor independent source',
                  'members':captured,'asof':asof,'code_sha256':sha(Path(__file__).read_bytes()),'no_label_change':True,'no_model_selection':True}
    key=sha(json.dumps(registration,sort_keys=True).encode());out=Path(pointer['path'])/'native_minute_audits'/key[:16];out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if old['registration']!=registration:raise ValueError('Minute audit registration collision')
        for p,digest in old['files'].items():
            if sha(Path(p).read_bytes())!=digest:raise ValueError('Frozen minute audit artifact SHA mismatch: '+p)
        summary=json.loads((out/'summary.json').read_text())
        (CACHE/'native_minute_audit_latest.json').write_text(json.dumps({'path':str(out.resolve()),'summary':summary},indent=2))
        print('reusing unchanged frozen native minute audit',key,flush=True);return
    (out/'registration.json').write_text(json.dumps(registration,indent=2));rows=[];missing=[]
    for n,registered in enumerate(captured):
        source=dict(registered);t=source['ticker'];snapshot=Path(source['snapshot_path'])
        daily=audit['selected_sources'].get(t)
        if not daily:source['status']='native_daily_unavailable';missing.append(source);continue
        try:minute_rows=inspect_minutes(pd.read_parquet(snapshot),t,cal,asof)
        except ValueError as exc:source.update(status='minute_source_unverified',reason=str(exc));missing.append(source);continue
        day=pd.read_parquet(daily['daily_path']);day['date']=day.time_key.astype(str).str[:10];day=day.set_index('date')
        for r in minute_rows:
            if r['date'] not in day.index:r['native_daily_available']=False
            else:
                z=day.loc[r['date']];r['native_daily_available']=True
                for k in ['open','high','low','close','volume']:r['daily_'+k]=float(z[k])
                if r['complete_rth_prefix']:
                    r['max_ohlc_relative_difference']=max(abs(r['daily_'+k]/r['minute_'+k]-1) for k in ['open','high','low','close'])
                    r['daily_high_above_rth_5m']=r['daily_high']/r['minute_high']-1
            rows.append(r)
        source.update(status='audited',daily_path=daily['daily_path'],daily_sha256=daily['daily_sha256']);sources.append(source)
        if n%25==0:print('native minute source comparison',n,'/',len(files),flush=True)
    d=pd.DataFrame(rows);d.to_parquet(out/'comparison.parquet',compression='zstd',compression_level=7,index=False)
    complete=d[d.complete_rth_prefix & d.native_daily_available].copy() if len(d) else d
    summary={'completed_at':datetime.now(timezone.utc).isoformat(),'registered_existing_minute_stocks':len(files),'audited_stocks':len(sources),'unavailable_or_unverified_stocks':len(missing),
      'stock_days':len(d),'complete_rth_5m_and_native_daily_days':len(complete),'ohlc_difference_within_0_1_percent':int((complete.max_ohlc_relative_difference<=.001).sum()) if len(complete) else 0,
      'daily_high_above_complete_rth_at_least_0_5_percent':int((complete.daily_high_above_rth_5m>=.005).sum()) if len(complete) else 0,
      'max_ohlc_relative_difference':float(complete.max_ohlc_relative_difference.max()) if len(complete) else None,
      'asof':asof,'full_registered_market_rth_label_verified':False,'no_label_change':True,'no_independent_effect_evidence':True,'scope':registration['scope']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'sources.json').write_text(json.dumps({'sources':sources,'unavailable':missing},indent=2))
    (out/'manifest.json').write_text(json.dumps({'registration':registration,'source_sha_checks':sources,'files':{str(p.resolve()):sha(p.read_bytes()) for p in [out/'comparison.parquet',out/'summary.json',out/'sources.json',out/'registration.json']}},indent=2))
    (CACHE/'native_minute_audit_latest.json').write_text(json.dumps({'path':str(out.resolve()),'summary':summary},indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
