"""Post-run read-only coverage inventory; no prices fetched, no strategy changed."""
from pathlib import Path
import sys,json,time,hashlib,datetime
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pandas as pd
import pyarrow.parquet as pq
from scripts.r2_client import R2Client
SYMBOLS=['SOXL','SOXS','SOXX','QQQ','TQQQ','SQQQ','UPRO','SPXU','SPY']
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v75'


def main():
    local=[]
    for symbol in SYMBOLS:
        candidates=[]
        for base in ['market_data/us_5m','market_data/model_training_history_v1/us_5m','market_data/model_training_history_v1/parts']:
            folder=ROOT/base/symbol
            if folder.exists():candidates.extend(sorted(folder.rglob('*.parquet')))
        for path in candidates:
            fields=pq.ParquetFile(path).schema_arrow.names;key=next((k for k in ['time_key','bar_end','bar_end_utc'] if k in fields),None)
            entry={'symbol':symbol,'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':pq.ParquetFile(path).metadata.num_rows,'timestamp_field':key}
            if key:
                dates=pd.read_parquet(path,columns=[key])[key];entry.update(first=str(dates.min()),last=str(dates.max()))
            if 'adjustment' in fields:entry['adjustment']=sorted(pd.read_parquet(path,columns=['adjustment']).adjustment.astype(str).unique().tolist())
            local.append(entry)
    prefixes=[f'{namespace}/{symbol}/' for symbol in SYMBOLS for namespace in ['us_5m','model_training_history_v1/us_5m','model_training_history_v1/parts']]
    deadline=time.monotonic()+30
    def probe(prefix):
        try:return {'prefix':prefix,'objects':R2Client(timeout=5,deadline=deadline).list_objects(prefix=prefix)}
        except Exception as error:return {'prefix':prefix,'error_type':type(error).__name__}
    with ThreadPoolExecutor(max_workers=4) as pool:remote=list(pool.map(probe,prefixes))
    capture=ROOT/'.cache/model_history_20260927/requests/QQQ/2023-04-01_2023-12-31/attempt_1790513115296276000/result.json'
    receipt={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local':local,'r2':remote,
             'legacy_QQQ_source_receipt':{'path':str(capture.relative_to(ROOT)),'sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),'reported_quality':json.loads(capture.read_text())},
             'quote_requests':0,'R2_uploads':0,'raw_data_downloads':0,'new_strategy_evaluated':False,
             'coverage_status':'Inventory only. Legacy completeness flags are source reports, not fresh acceptance. Local/R2 absence is limited to listed namespaces; historical scope and corporate actions still need verification.',
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (CACHE/'older_history_inventory.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print({'local_files':len(local),'remote_prefixes':len(remote),'remote_errors':[r for r in remote if 'error_type' in r],
           'objects_by_prefix':{r['prefix']:len(r['objects']) for r in remote if 'objects' in r}},flush=True)


if __name__=='__main__':main()
