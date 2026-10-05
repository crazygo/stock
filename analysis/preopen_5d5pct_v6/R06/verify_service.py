"""Real success/failure/unknown API paths with the new source-date evidence."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from urllib.request import build_opener,ProxyHandler
from urllib.parse import urlencode
from common import OUT,write,now
DIRECT=build_opener(ProxyHandler({}))

def get(path,q=None):
    with DIRECT.open('http://127.0.0.1:8771'+path+('?' + urlencode(q) if q else ''),timeout=40) as response:return json.load(response)

def main():
    arms=['T0','TC','TF','TO','TV','TS','TA','TL'];checks=[];r=get('/api/results');assert not r['independent_pass'];assert all(a in [x['arm'] for x in r['arms']] for a in arms)
    for arm in arms:
        for variant in ['candidate_only','top_one']:
            events=get('/api/events',dict(arm=arm,variant=variant));assert len({(e['symbol'],e['day']) for e in events})==len(events)
            for y in [1,0,None]:
                e=next((e for e in events if e['y']==y),None)
                if e is None:continue
                detail=get('/api/inspect',dict(arm=arm,variant=variant,symbol=e['symbol'],day=e['day']));assert detail['event']['y']==y and detail['event']['label_end'] and detail['features'];assert detail['source_evidence']
                if y!=1:assert detail['first_touch'] is None
                for key,value in detail['source_evidence'].items():
                    if key.endswith('_source_day') and value is not None:assert value<e['day']
                    if key.endswith('_available_day') and value is not None:assert value<=e['day']
                checks.append(dict(arm=arm,variant=variant,outcome=y,symbol=e['symbol'],day=e['day'],passed=True))
    chart=get('/api/chart',dict(symbol='ALAB',start='2026-07-01',end='2026-07-08'));assert chart['bars'] and chart['daily'] and chart['grain_minutes']==5
    checks.append(dict(scenario='real_complete_chart',bars=len(chart['bars']),passed=True))
    write(OUT/'R06/service_verification.json',dict(at=now(),status='passed',checks=checks,note='Read-only engineering verification, no model-effect admission.'));print(json.dumps(dict(status='passed',checks=len(checks))))

if __name__=='__main__':main()
