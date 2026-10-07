"""Version original issuer releases with bounded read-only HTTP requests."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,time,urllib.request
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def run():
    start=time.monotonic();j=json.loads((HERE/'business_reviews.json').read_text());urls={s['url']:s for r in j['records'].values() for s in r['sources']};out=ROOT/'.cache/stock_data_v1/disclosures';out.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        url,s=item;checked=datetime.now(timezone.utc).isoformat()
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'stock-research/1.0 contact research@example.com'})
            with urllib.request.urlopen(req,timeout=5) as response:raw=response.read();mime=response.headers.get('content-type')
            digest=hashlib.sha256(raw).hexdigest();p=out/(digest+('.pdf' if raw.startswith(b'%PDF') else '.html'));ledger=out/(digest+'.json')
            if not p.exists():p.write_bytes(raw)
            if not ledger.exists():ledger.write_text(json.dumps({'received_at':datetime.now(timezone.utc).isoformat(),'sha256':digest,'url':url},indent=2)+'\n')
            original=json.loads(ledger.read_text());return {**s,'status':'captured','path':str(p.relative_to(ROOT)),'sha256':digest,'mime':mime,'bytes':len(raw),'received_at':original['received_at'],'checked_at':checked,'available_at':None}
        except Exception as e:return {**s,'status':'browser_reviewed_raw_capture_failed','raw_capture_error':type(e).__name__,'checked_at':checked,'received_at':None,'available_at':None}
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(fetch,urls.items()))
    p=HERE/'EVIDENCE_POINTER.json';p.write_text(json.dumps({'scope':'focused_8_issuers_9_releases_not_full_pool_news','records':records,'resources':{'elapsed_seconds':round(time.monotonic()-start,3),'requests':len(urls),'fees':None}},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'sources':len(records),'raw_captured':sum(r['status']=='captured' for r in records),'elapsed_seconds':round(time.monotonic()-start,3)}))
if __name__=='__main__':run()
