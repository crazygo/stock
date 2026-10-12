#!/usr/bin/env python3
"""Poll public feeds, save raw snapshots, record first-seen time per item. Usage: poller.py OUTDIR N_POLLS INTERVAL_S"""
import sys, os, time, json, re, subprocess, hashlib, datetime as dt
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
OUT=sys.argv[1]; N=int(sys.argv[2]); IV=int(sys.argv[3])
FEEDS={
 'prn_all':('https://www.prnewswire.com/rss/news-releases-list.rss','curl/8'),
 'prn_energy':('https://www.prnewswire.com/rss/energy-latest-news/energy-latest-news-list.rss','curl/8'),
 'edgar_8k':('https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&company=&dateb=&owner=include&start=0&count=100&output=atom','RayStockResearch research-bot@example.com'),
 'edgar_all':('https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&company=&dateb=&owner=include&start=0&count=100&output=atom','RayStockResearch research-bot@example.com'),
 'nasdaq_halts':('https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts','Mozilla/5.0'),
 'gnews_prwire':('https://news.google.com/rss/search?q=%22PRNewswire%22+OR+%22GLOBE+NEWSWIRE%22+OR+%22BUSINESS+WIRE%22+when:1d&hl=en-US&gl=US&ceid=US:en','Mozilla/5.0'),
}
extra=os.environ.get('FEEDS_JSON')
if extra: FEEDS=json.loads(extra)
os.makedirs(OUT+'/snap',exist_ok=True)
seenf=OUT+'/first_seen.jsonl'
seen=set()
if os.path.exists(seenf):
    for l in open(seenf): seen.add(json.loads(l)['key'])
def parse(name,data):
    items=[]
    try: root=ET.fromstring(data)
    except Exception as e: return items
    ns={'a':'http://www.w3.org/2005/Atom'}
    for it in root.iter('item'):
        g=(it.findtext('guid') or it.findtext('link') or '').strip()
        items.append(dict(id=g,title=(it.findtext('title') or '').strip(),link=(it.findtext('link') or '').strip(),pub=(it.findtext('pubDate') or it.findtext('{http://purl.org/dc/elements/1.1/}date') or '').strip()))
    for e in root.iter('{http://www.w3.org/2005/Atom}entry'):
        l=e.find('a:link',ns)
        items.append(dict(id=(e.findtext('a:id',namespaces=ns) or '').strip(),title=(e.findtext('a:title',namespaces=ns) or '').strip(),link=l.get('href') if l is not None else '',pub=(e.findtext('a:updated',namespaces=ns) or e.findtext('a:published',namespaces=ns) or '').strip()))
    return items
def to_utc(s):
    try:
        if re.match(r'\d{4}-\d\d-\d\dT',s): d=dt.datetime.fromisoformat(s.replace('Z','+00:00'))
        else: d=parsedate_to_datetime(s.replace(' UT',' GMT'))
        return d.astimezone(dt.timezone.utc).isoformat()
    except Exception: return None
for i in range(N):
    t0=dt.datetime.now(dt.timezone.utc)
    for name,(url,ua) in FEEDS.items():
        r=subprocess.run(['curl','-s','-L','--compressed','--max-time','30','-A',ua,url],capture_output=True)
        ts=t0.strftime('%Y%m%dT%H%M%SZ')
        open(f'{OUT}/snap/{name}_{ts}.xml','wb').write(r.stdout)
        items=parse(name,r.stdout)
        with open(seenf,'a') as f:
            for it in items:
                key=name+'|'+(it['id'] or it['link'] or it['title'])
                if key in seen: continue
                seen.add(key)
                pu=to_utc(it['pub'])
                lag=(t0-dt.datetime.fromisoformat(pu)).total_seconds()/60 if pu else None
                f.write(json.dumps(dict(key=key,feed=name,poll=i,first_seen_utc=t0.isoformat(),pub_raw=it['pub'],pub_utc=pu,seen_minus_pub_min=round(lag,1) if lag is not None else None,title=it['title'][:200],link=it['link']),ensure_ascii=False)+'\n')
        print(t0.isoformat(),name,len(r.stdout),len(items),flush=True)
    if i<N-1: time.sleep(IV)
