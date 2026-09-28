"""Retrospective T/P opportunity counts; no model, data acquisition, or trades."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')

def representatives(candidates):
    """Half-open intervals: strongest historical representative, no overlap."""
    result=[]
    for c in sorted(candidates,key=lambda c:(-c['gain'],c['duration'],c['start'])):
        if all(c['end']<=o['start'] or c['start']>=o['end'] for o in result): result.append(c)
    return sorted(result,key=lambda c:c['start'])

def candidates_for_width(prices,valid,k):
    result=[]
    eligible=[]
    for i in range(len(prices)-k+1):
        if not valid[i:i+k].all(): continue
        eligible.append(i)
        peak=i+int(np.argmax(prices[i:i+k,1]))
        base=prices[i,0]; gain=prices[peak,1]/base-1
        if gain < .03-1e-12: continue
        end=peak+1; entry=i+3
        remaining=prices[entry:end,1].max()/prices[entry,0]-1 if entry<end else None
        remaining_close=prices[entry:end,3].max()/prices[entry,0]-1 if entry<end else None
        entry_gain=prices[entry,0]/base-1 if entry<end else None
        first2=i+int(np.argmax(prices[i:end,3]>=base*1.02)) if (prices[i:end,3]>=base*1.02).any() else None
        # Close at (j+1)*5; +30s signal +120s human => next 5m start (j+2)*5.
        reactive_entry=first2+2 if first2 is not None else None
        reactive_remaining=prices[reactive_entry:end,1].max()/prices[reactive_entry,0]-1 if reactive_entry is not None and reactive_entry<end else None
        result.append({'start':i,'end':end,'duration':(end-i)*5,'peak_lower':(peak-i)*5,
                       'gain':float(gain),'close_gain':float(prices[i:end,3].max()/base-1),
                       'base':float(base),'peak':float(prices[peak,1]),
                       'r10':float(prices[i+1,3]/base-1),'entry_gain':None if entry_gain is None else float(entry_gain),
                       'remaining':None if remaining is None else float(remaining),
                       'remaining_close':None if remaining_close is None else float(remaining_close),
                       'first2_index':first2,'reactive_entry_index':reactive_entry,
                       'reactive_remaining':None if reactive_remaining is None else float(reactive_remaining)})
    return result,eligible

def run(out):
    cfg=json.loads((HERE/'config.json').read_text());out.mkdir(parents=True,exist_ok=False)
    members=[m for m in json.loads((ROOT/cfg['universe']).read_text())['members'] if m['role']=='candidate']
    sessions=[s for s in json.loads((ROOT/cfg['calendar']).read_text())['sessions'] if cfg['start_date']<=s['session_date']<=cfg['end_date']]
    files=[HERE/'config.json',HERE/'PROTOCOL.md',HERE/'inventory.py',ROOT/cfg['calendar'],ROOT/cfg['universe']]
    files += [ROOT/cfg['source_dir']/m['symbol']/'2026.parquet' for m in members]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files if p.exists()}
    manifest={'started_at':datetime.now(timezone.utc).isoformat(),'config':cfg,'sources':hashes,
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
              'evidence':'retrospective_current_universe_no_predictive_claim'}
    write(out/'manifest_at_launch.json',manifest)
    events={};grids={};coverage=[];days={};source=[];ids=0
    for m in members:
        symbol=m['symbol'];path=ROOT/cfg['source_dir']/symbol/'2026.parquet'
        if not path.exists():
            source.append({'symbol':symbol,'status':'missing'});continue
        f=pd.read_parquet(path)
        if set(f.price_basis.dropna())!={'NONE'}:raise ValueError('price basis '+symbol)
        for col in ['start_at','end_at','available_at']:f[col]=pd.to_datetime(f[col],utc=True)
        if f.start_at.duplicated().any():raise ValueError('duplicate '+symbol)
        f=f.set_index('start_at').sort_index()
        counts=Counter();monthly=Counter()
        bucket={f'{scope}|{t}|{p}':[] for scope in cfg['scopes'] for t in cfg['widths_minutes'] for p in cfg['amplitudes']}
        for session in sessions:
            date=session['session_date'];opening=pd.Timestamp(session['open_at']);closing=pd.Timestamp(session['close_at'])
            index=pd.date_range(opening,closing,freq='5min',inclusive='left')
            a=f.reindex(index);v=a[['open','high','low','close','volume']].to_numpy(float)
            valid=(np.isfinite(v).all(axis=1)&(v[:,:4]>0).all(axis=1)&(v[:,4]>0)&(v[:,1]>=v[:,[0,2,3]].max(axis=1))&(v[:,2]<=v[:,[0,1,3]].min(axis=1)))
            valid &= a.session_type.eq('regular').to_numpy() & a.end_at.eq(index+pd.Timedelta(minutes=5)).to_numpy()
            valid &= a.available_at.ge(a.end_at).to_numpy() & a.available_at.le(pd.Timestamp.now(tz='UTC')).to_numpy()
            counts['official_days']+=1;counts['valid_rth_bars']+=int(valid.sum());counts['expected_rth_bars']+=len(valid)
            counts['full_rth_days']+=int(valid.all());counts['any_rth_days']+=int(valid.any())
            for t in cfg['widths_minutes']:
                cand,eligible=candidates_for_width(v[:,:4],valid,t//5)
                for scope,limit in cfg['scopes'].items():
                    elig=[i for i in eligible if i<limit]
                    coverage.append({'symbol':symbol,'date':date,'scope':scope,'width':t,'valid_starts':len(elig),'possible_starts':max(0,min(limit,len(valid)-t//5+1))})
                    subset=[c for c in cand if c['start']<limit]
                    for p in cfg['amplitudes']:
                        picked=representatives([c for c in subset if c['gain']>=p-1e-12])
                        key=f'{scope}|{t}|{p}'
                        for c in picked:
                            eid=f'{symbol}|{date}|{c["start"]}|{c["end"]}'
                            if eid not in events:
                                events[eid]={'symbol':symbol,'date':date,**c}
                            bucket[key].append(eid)
                            daykey=f'{symbol}|{date}'
                            if daykey not in days:
                                days[daykey]=[[float(z) for z in row[:4]] if good else None for row,good in zip(v,valid)]
        for key,eids in bucket.items():grids.setdefault(key,{})[symbol]=eids
        source.append({'symbol':symbol,'status':'read',**dict(counts),'rows':len(f),'source_start':f.index.min().isoformat(),'source_end':f.end_at.max().isoformat()})
        print(f'{symbol}: unique paths={len(events)}',flush=True)
    # Explicit zero rows for every stock, grid and scope, including missing files.
    for scope in cfg['scopes']:
        for t in cfg['widths_minutes']:
            for p in cfg['amplitudes']:
                key=f'{scope}|{t}|{p}'
                for m in members:grids.setdefault(key,{}).setdefault(m['symbol'],[])
    cov=pd.DataFrame(coverage);cov.to_parquet(out/'coverage.parquet',index=False)
    agg=cov.groupby(['symbol','scope','width']).agg(covered_days=('valid_starts',lambda x:int((x>0).sum())),valid_starts=('valid_starts','sum'),possible_starts=('possible_starts','sum')).reset_index()
    rows=[]
    for key,stocks in grids.items():
        scope,t,p=key.split('|');t=int(t);p=float(p)
        for symbol,eids in stocks.items():
            selected=[events[e] for e in eids if events[e]['duration']>=cfg['min_peak_upper_minutes']]
            cc=agg[(agg.symbol==symbol)&(agg.scope==scope)&(agg.width==t)]
            covered=int(cc.covered_days.iloc[0]) if len(cc) else 0
            rows.append({'symbol':symbol,'scope':scope,'T':t,'P':p,'count':len(selected),'all_durations_count':len(eids),
                         'opportunity_days':len({e['date'] for e in selected}),
                         'close_confirmed':sum(e['close_gain']>=p-1e-12 for e in selected),
                         'after_two8':sum(e['remaining'] is not None and e['remaining']>=.08-1e-12 for e in selected),
                         'early2_after_two8':sum(e['remaining'] is not None and e['remaining']>=.08-1e-12 and 0<=e['entry_gain']<=.02+1e-12 for e in selected),
                         'observed2_then8':sum(e['reactive_remaining'] is not None and e['reactive_remaining']>=.08-1e-12 for e in selected),
                         'covered_days':covered,'count_per20':len(selected)*20/covered if covered else None})
    write(out/'counts.json',rows)
    write(out/'coverage_summary.json',agg.to_dict(orient='records'))
    payload={'config':cfg,'symbols':[m['symbol'] for m in members],'official_days':len(sessions),'source':source,
             'events':events,'grids':grids,'days':days,'counts':rows,'coverage':agg.to_dict(orient='records')}
    write(out/'evidence.json',payload)
    if any(sha(ROOT/p)!=h for p,h in hashes.items()):raise RuntimeError('sources changed during inventory')
    summary={'symbols':len(members),'official_days':len(sessions),'unique_representative_paths':len(events),
             'source_hashes_unchanged':True,'grid_cells':len(grids),'stock_grid_rows':len(rows),'completed_at':datetime.now(timezone.utc).isoformat()}
    write(out/'summary.json',summary)
    write(out/'output_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output.resolve())
