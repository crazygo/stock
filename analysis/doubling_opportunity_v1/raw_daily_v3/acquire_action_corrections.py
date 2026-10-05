#!/usr/bin/env python3
"""Recover only exact DOM-registered links lost by compression; keep original requests intact."""
import hashlib,json,sys,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));from data import CACHE

def main():
    source=HERE/'ARCHIVE_LINK_CORRECTIONS.json';registered=json.loads(source.read_text());root=CACHE/'corporate_actions_v3/corrections';root.mkdir(parents=True,exist_ok=True)
    path=root/'acquisition.json';previous=json.loads(path.read_text()) if path.exists() else {};digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if previous.get('terminal') and previous.get('registration_sha256')==digest and all(Path(r['path']).exists() and hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in previous['results'].values() if r['status']=='available'):
        print('unchanged actual-link correction source retained');return
    state={'registration_sha256':digest,'registered_count':len(registered['corrections']),'terminal':False,'results':{}}
    for r in registered['corrections']:
        target=root/(hashlib.sha256(r['fetch_url'].encode()).hexdigest()[:16]+'.html');record=dict(r)
        for attempt in range(3):
            try:
                req=urllib.request.Request(r['fetch_url'],headers={'User-Agent':'stock-research public corporate-action source correction'})
                with urllib.request.urlopen(req,timeout=30) as response:raw=response.read()
                if b'Equity Corporate Actions Alert' not in raw:raise ValueError('Unexpected public article document')
                target.write_bytes(raw);record.update(status='available',path=str(target.resolve()),sha256=hashlib.sha256(raw).hexdigest());break
            except Exception as exc:
                record.update(status='unavailable',error_type=type(exc).__name__)
                if attempt<2:time.sleep(2+attempt)
        record['received_at']=datetime.now(timezone.utc).isoformat();state['results'][r['compressed_url']]=record
    state['terminal']=True;path.write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))

if __name__=='__main__':main()
