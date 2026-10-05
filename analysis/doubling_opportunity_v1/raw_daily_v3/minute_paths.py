"""Read frozen native minute snapshots for truthful diagnostic paths, without altering labels."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));from market import split_state
from actions import audited_events
from prepare_actions import load_audit
sys.path.insert(0,str(HERE.parent));from data import CACHE

def classify(start,session_map,session_days):
    """Classify actual bar starts, including genuine Sunday night before Monday."""
    d=start.strftime('%Y-%m-%d');minute=start.hour*60+start.minute;weekday=start.weekday()
    if minute>=1200:
        if weekday not in [6,0,1,2,3]:return None
        following=next((x for x in session_days if x>d),None)
        if not following or (pd.Timestamp(following)-pd.Timestamp(d)).days!=1:return None
        return following,'夜盘'
    s=session_map.get(d)
    if s is None:return None
    if minute<240:return d,'夜盘'
    opening=pd.Timestamp(s['open_at']).tz_convert('America/New_York');closing=pd.Timestamp(s['close_at']).tz_convert('America/New_York')
    if minute<opening.hour*60+opening.minute:return d,'盘前'
    if minute<closing.hour*60+closing.minute:return d,'常规盘'
    return d,'盘后'

def records(frame,cal,events,asof):
    frame=frame.assign(_ordered_end=pd.to_datetime(frame.end_at,utc=True) if 'end_at' in frame else pd.to_datetime(frame.start_at,utc=True)).sort_values('_ordered_end')
    index=pd.DatetimeIndex([s['session_date'] for s in cal['sessions']]);factor,_,_,_=split_state(index,events,asof);by_day=dict(zip(index.strftime('%Y-%m-%d'),factor))
    split_days=set(events.loc[pd.to_numeric(events.get('split_ratio',pd.Series(index=events.index,dtype=float)),errors='coerce').fillna(1).ne(1),'ex_div_date']) if len(events) else set()
    session_map={s['session_date']:s for s in cal['sessions']};session_days=sorted(session_map)
    start=pd.to_datetime(frame.start_at,utc=True).dt.tz_convert('America/New_York');output=[];excluded=0;invalid=0
    for z,stamp in zip(frame.itertuples(index=False),start):
        ss=classify(stamp,session_map,session_days)
        if ss is None or ss[0]>asof:excluded+=1;continue
        date,type=ss;basis=by_day.get(date)
        # An ex-date does not establish the intraday conversion time outside RTH.
        if date in split_days and type in ['夜盘','盘前']:basis=None
        values=[float(getattr(z,k)) for k in ['open','high','low','close','volume']]
        if not np.isfinite(values).all() or min(values[:4])<=0 or values[4]<0 or values[1]<max(values[0],values[2],values[3]) or values[2]>min(values[0],values[1],values[3]):
            invalid+=1;values=[None]*5
        # Raw NONE prices remain intact; the optional factor converts the barrier, never these bars.
        output.append([str(z.time_key),*values,float(basis) if basis is not None else None,date,type])
    return output,{'outside_reference_or_unverified_session_rows_excluded':excluded,'invalid_ohlc_rows_retained_as_missing':invalid}

def export(lineage,data_dir):
    pointer=CACHE/'native_minute_audit_latest.json'
    if not pointer.exists():return {},{'status':'minute_source_audit_not_yet_complete','available_tickers':[]}
    latest=json.loads(pointer.read_text());root=Path(latest['path']);manifest=json.loads((root/'manifest.json').read_text())
    if manifest['registration']['source_audit']!=lineage['action_source_audit']:return {},{'status':'minute_source_audit_belongs_to_other_daily_snapshot','available_tickers':[]}
    for p,digest in manifest['files'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=digest:raise ValueError('Minute audit artifact SHA mismatch: '+p)
    audit,ready=load_audit(validate_sources=True);metadata={};hashes={}
    for source in manifest['source_sha_checks']:
        t=source['ticker'];snapshot=Path(source['snapshot_path'])
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Frozen minute snapshot SHA mismatch: '+t)
        native=lineage['native_sources'].get(t)
        if not native or native['daily_sha256']!=source['daily_sha256']:raise ValueError('Minute/daily comparison basis snapshot mismatch: '+t)
        events,_=audited_events(pd.read_parquet(native['rehab_path']),audit['registry'][t],lineage['asof'])
        q,issues=records(pd.read_parquet(snapshot),lineage['calendar'],events,lineage['asof'])
        state=audit['registry'][t]
        info={'ticker':t,'basis':'raw NONE dollars; target barrier converts per official session split factor; no relabelling',
          'rows':len(q),'first_bar_end_et':q[0][0] if q else None,'last_bar_end_et':q[-1][0] if q else None,
          'whole_history_unknown':state['whole_history_unknown'],'unresolved_action_dates':sorted({r['ex_date'] for r in state['unknown_events']}),
          'source_sha256':source['sha256'],'diagnostic_only':True,**issues}
        path=data_dir/(t+'.5m.json');path.write_text(json.dumps({'ticker':t,'minute5':q,'metadata':info},separators=(',',':'),allow_nan=False))
        metadata[t]=info;hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    return metadata,{'status':'frozen_existing_native_cache_exported','available_tickers':sorted(metadata),'audit_path':str(root.resolve()),
      'scope':manifest['registration']['scope'],'output_hashes':hashes,'full_market_minute_coverage':False,'source_summary':latest['summary']}
