"""Restore reporting precision without changing inputs, labels, probabilities or selection."""
import json,shutil
import numpy as np
import pandas as pd
from common import OUT,OLD,write,save,now

def main():
    keys=pd.read_parquet(OLD/'weekly_v1/panel.parquet',columns=['symbol','day','minute'])
    keys['symbol']=keys.symbol.astype(str);keys['day']=keys.day.astype(str)
    labels=pd.read_parquet(OLD/'horizon_v1/labels_5d5pct.parquet',columns=['entry','target'])
    index=pd.MultiIndex.from_frame(keys);assert index.is_unique
    values=labels.to_numpy();fixed=0
    for path in sorted((OUT/'cache/runs').glob('*/*_replay.json')):
        backup=OUT/'cache/reporting_float32_backup'/path.parent.name/path.name
        if not backup.exists():backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,backup)
        sim=json.loads(path.read_text());events=sim['events']
        if events:
            ix=pd.MultiIndex.from_tuples([(e['symbol'],e['day'],int(e['minute'])) for e in events]);positions=index.get_indexer(ix);assert (positions>=0).all()
            for e,(entry,target) in zip(events,values[positions]):e['entry']=float(entry) if np.isfinite(entry) else None;e['target']=float(target) if np.isfinite(target) else None;fixed+=1
            write(path,sim)
    for path in sorted((OUT/'cache/runs').glob('*/*.parquet')):
        f=pd.read_parquet(path);f['symbol']=f.symbol.astype(str);f['day']=f.day.astype(str)
        positions=index.get_indexer(pd.MultiIndex.from_frame(f[['symbol','day','minute']]));assert (positions>=0).all()
        f['entry']=values[positions,0];f['target']=values[positions,1];save(f,path)
    write(OUT/'reporting_precision_repair.json',dict(at=now(),events=fixed,status='restored_original_float64',
       unchanged=['features','y','unknown statuses','probabilities','thresholds','signal selection','precision'],backup='cache/reporting_float32_backup'))
    from research import aggregate
    aggregate()
    print(json.dumps(dict(restored_events=fixed)),flush=True)
if __name__=='__main__':main()
