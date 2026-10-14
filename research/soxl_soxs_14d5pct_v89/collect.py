"""Bounded official public-document collection; no market data or old-cache writes."""
import argparse,subprocess,time,json,hashlib,re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from urllib.parse import urlparse
from catalogue import ROOT,SOURCE,CACHE,sha,dump

ALLOWED={'nvidianews.nvidia.com','ir.amd.com','investors.broadcom.com','investors.micron.com','ir.appliedmaterials.com','newsroom.lamresearch.com','ir.kla.com','www.intc.com'}
def collect(limit=None):
    start=time.monotonic();budget=900;folder=CACHE/'http';folder.mkdir(parents=True,exist_ok=True)
    rows=json.loads((SOURCE/'catalogue.json').read_text())['records']
    if limit:rows=[next(r for r in rows if r['symbol']==s) for s in dict.fromkeys(r['symbol'] for r in rows)][:limit]
    local_probe=CACHE/'old_body_cache_probe.json'
    local={r['url']:r for r in json.loads(local_probe.read_text())['matches']} if local_probe.exists() else {}
    def one(row):
        key=row['id'];meta=folder/(key+'.json');body=folder/(key+'.html')
        if meta.exists():
            d=json.loads(meta.read_text())
            if d.get('cache_file'):assert d['sha256']==sha(ROOT/d['cache_file'])
            return d
        if row['url'] in local:
            old=ROOT/local[row['url']]['file'];saved=json.loads(old.read_text())
            d=dict(row,status='browser_body_review_pending',received_at=saved['observed_at'],cache_file=str(old.relative_to(ROOT)),sha256=sha(old),bytes=old.stat().st_size,source_tier='existing_local')
            dump(meta,d);return d
        assert urlparse(row['url']).hostname in ALLOWED
        if time.monotonic()-start>budget:
            d=dict(row,status='not_attempted_budget',received_at=datetime.now(timezone.utc).isoformat());dump(meta,d);return d
        cmd=['curl','--http1.1','--location','--fail-with-body','--silent','--show-error','--max-time','15','--max-filesize','4000000','--output',str(body),'--write-out','%{http_code}\n%{url_effective}',row['url']]
        try:
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=18)
            info=p.stdout.splitlines();html=body.read_text(errors='replace') if body.exists() else ''
            complete=bool(p.returncode==0 and re.search(r'</html\s*>',html,re.I))
            d=dict(row,status='html_body_review_pending' if complete else 'http_or_incomplete_failure',received_at=datetime.now(timezone.utc).isoformat(),http_status=info[0] if info else None,final_url=info[1] if len(info)>1 else None,
                returncode=p.returncode,error=p.stderr[-1000:],cache_file=str(body.relative_to(ROOT)) if body.exists() else None,sha256=sha(body) if body.exists() else None,bytes=body.stat().st_size if body.exists() else 0)
        except subprocess.TimeoutExpired:
            d=dict(row,status='process_timeout',received_at=datetime.now(timezone.utc).isoformat(),cache_file=str(body.relative_to(ROOT)) if body.exists() else None,sha256=sha(body) if body.exists() else None)
        dump(meta,d);time.sleep(.25);return d
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,d in enumerate(pool.map(one,rows)):print(i+1,d['symbol'],d['published_date'],d['status'],d.get('bytes'),flush=True)
    dump(CACHE/'http_manifest.json',dict(created_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,requested=len(rows),records=[json.loads((folder/(r['id']+'.json')).read_text()) for r in rows]))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');a=p.parse_args();collect(8 if a.probe else None)
