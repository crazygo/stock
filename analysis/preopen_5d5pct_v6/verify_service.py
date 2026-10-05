"""Read-only integration checks of real chart, signal and unknown-result paths."""
import json
from urllib.request import build_opener,ProxyHandler
from urllib.parse import urlencode
from common import OUT,write,now

DIRECT=build_opener(ProxyHandler({}))

def get(path,query=None):
    url='http://127.0.0.1:8771'+path+('?' + urlencode(query) if query else '')
    with DIRECT.open(url,timeout=40) as response:return json.load(response)

def main():
    results=get('/api/results');assert results['independent_pass'] is False
    checks=[]
    for arm in ['A0','S0','P1','F1','H3','D1']:
        route=next((r for r in results['arms'] if r['arm']==arm),None)
        if route is None:continue
        events=get('/api/events',dict(arm=arm,variant='top_one'))
        for outcome in [1,0,None]:
            event=next((e for e in events if e['y']==outcome),None)
            if event is None:continue
            detail=get('/api/inspect',dict(arm=arm,variant='top_one',symbol=event['symbol'],day=event['day']))
            assert detail['event']['y']==outcome and detail['event']['label_end'] and detail['features']
            if outcome!=1:assert detail['first_touch'] is None
            checks.append(dict(arm=arm,outcome=outcome,symbol=event['symbol'],day=event['day'],passed=True))
    chart=get('/api/chart',dict(symbol='ALAB',start='2026-07-01',end='2026-07-08'))
    assert chart['grain_minutes']==5 and chart['bars'] and chart['daily']
    assert all(len(b)==7 and min(b[2:6])>0 and b[6]>=0 for b in chart['bars'])
    assert len({b[0] for b in chart['bars']})==len(chart['bars'])
    checks.append(dict(scenario='real_five_minute_price_and_volume',bars=len(chart['bars']),passed=True))
    write(OUT/'service_verification.json',dict(at=now(),status='passed',checks=checks,note='Read-only UI/API evidence; no independent model or historical execution effectiveness claim.'))
    print(json.dumps(dict(status='passed',checks=len(checks))))
if __name__=='__main__':main()
