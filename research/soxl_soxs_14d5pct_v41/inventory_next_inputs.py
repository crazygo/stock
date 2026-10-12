from pathlib import Path
import sys,json,datetime,time,hashlib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path.cwd()))
from scripts.r2_client import R2Client
symbols=['SPY','IWM','TLT','HYG','LQD','SHY','UUP','XLY','XLP','VIX'];local=[]
for p in Path('market_data').rglob('*.parquet'):
 found=next((s for s in symbols if s in p.parts or p.stem in [s,'US.'+s]),None)
 if not found:continue
 d=pd.read_parquet(p);col=next((c for c in ['time_key','date','end_at','ex_date','ex_div_date'] if c in d),None)
 local.append({'symbol':found,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(d),'columns':list(d.columns),'time_column':col,
 'first':str(d[col].min()) if col and len(d) else None,'last':str(d[col].max()) if col and len(d) else None})
missing=[s for s in symbols if not any(x['symbol']==s for x in local)]
# Inventory only; distinct expected namespaces, bounded read requests, no downloads or provider fallback yet.
prefixes=[f'market_data/{interval}/{s}/' for s in missing for interval in ['us_5m','us_60m']]
prefixes += [f'market_data/trend_quadrant_v1/daily/US.{s}.parquet' for s in missing]
deadline=time.monotonic()+30
def probe(prefix):
 try:return {'prefix':prefix,'objects':R2Client(timeout=5,deadline=deadline).list_objects(prefix=prefix,max_keys=100)}
 except Exception as e:return {'prefix':prefix,'error_type':type(e).__name__}
with ThreadPoolExecutor(max_workers=4) as pool:remote=list(pool.map(probe,prefixes))
result={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'next-experiment feasibility inventory; not used by v41 training or selection','symbols':symbols,'local_files':local,'remote_prefixes':remote,
 'scope_limit':'R2 only conventional us5m/us60m/trendquadrant namespaces; empty matches do not prove no source elsewhere. File presence/range does not verify causal full coverage.','downloads':0,'opend_requests':0,'uploads':0}
Path('.cache/soxl_soxs_14d5pct_v41/next_input_inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print({'local':[{k:x[k] for k in ['symbol','path','rows','first','last']} for x in local],'remote_checked':len(remote),'remote_with_objects':sum(bool(x.get('objects')) for x in remote),'errors':sum('error_type' in x for x in remote)})
