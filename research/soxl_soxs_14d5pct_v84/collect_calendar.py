"""Collect public Fed schedule notices and audit the official release index.

Only dates and publication metadata enter the event dataset. Source HTML remains
in ignored cache; current retrieval time is never historical availability.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urljoin
import hashlib, json, re, time, subprocess

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v84/sources'
HOST='https://www.federalreserve.gov'
NOTICES=['monetary20220624a.htm','monetary20230623a.htm','monetary20240809a.htm']
TIMING=HOST+'/econres/notes/feds-notes/questions-and-answers-the-information-content-of-the-post-fomc-meeting-press-conference-20211012.html'


class Parser(HTMLParser):
    def __init__(self):
        super().__init__();self.text=[];self.links=[];self.link=None;self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ['script','style']:self.skip+=1
        if tag=='a':self.link=[dict(attrs).get('href',''),[]]
        if tag in ['p','div','br','li','h1','h2','h3','h4','time']:self.text.append('\n')
    def handle_endtag(self,tag):
        if tag in ['script','style']:self.skip-=1
        if tag=='a' and self.link:
            self.links.append((self.link[0],' '.join(''.join(self.link[1]).split())));self.link=None
        if tag in ['p','div','li','time']:self.text.append('\n')
    def handle_data(self,data):
        if self.skip:return
        self.text.append(data)
        if self.link:self.link[1].append(data)


def parse(data):
    p=Parser();p.feed(data);return '\n'.join(' '.join(x.split()) for x in ''.join(p.text).splitlines() if x.strip()),p.links


def fetch(url):
    CACHE.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(url.encode()).hexdigest()[:20];path=CACHE/(key+'.html');meta=CACHE/(key+'.json')
    if not path.exists():
        request=Request(url,headers={'User-Agent':'Mozilla/5.0 historical calendar research'})
        for attempt in range(3):
            try:
                with urlopen(request,timeout=25) as r:data=r.read()
                break
            except Exception:
                if attempt==2:
                    # Some proxy responses end their HTTP/1.1 chunks early. A
                    # bounded HTTP/1.0 retry must succeed, never accept partials.
                    data=subprocess.check_output(['curl','--http1.0','--fail','--silent','--show-error','--max-time','25',url])
                    break
                time.sleep(1)
        path.write_bytes(data);meta.write_text(json.dumps(dict(url=url,fetched_at=datetime.now(timezone.utc).isoformat(),sha256=hashlib.sha256(data).hexdigest()),indent=2)+'\n')
    data=path.read_bytes();m=json.loads(meta.read_text());assert m['url']==url and m['sha256']==hashlib.sha256(data).hexdigest()
    txt,links=parse(data.decode());return dict(**m,cache_file=str(path.relative_to(ROOT)),text=txt,links=links)


def collect():
    urls=[HOST+'/newsevents/pressreleases/'+n for n in NOTICES]+[TIMING]
    urls += [HOST+f'/newsevents/pressreleases/{y}-press.htm' for y in range(2022,2027)]
    with ThreadPoolExecutor(max_workers=3) as pool:pages=list(pool.map(fetch,urls))
    candidates={};minutes=[]
    for page in pages[4:]:
        for link,title in page['links']:
            m=re.search(r'/pressreleases/\w+(\d{8})[a-z]\.htm$',link)
            if not m or not '20220624'<=m[1]<='20260430':continue
            if re.search('Federal Open Market|FOMC|meeting.*(reschedul|cancel|chang)|((reschedul|cancel|chang).*meeting)',title,re.I):
                url=urljoin(HOST,link);candidates[url]=dict(url=url,title=title,date=m[1],index=page['url'])
                if title.startswith('Minutes of the Federal Open Market'):minutes.append(url)
    with ThreadPoolExecutor(max_workers=3) as pool:press=list(pool.map(fetch,sorted(candidates)))
    minute_urls={}
    for page in press:
        if page['url'] not in minutes:continue
        for link,title in page['links']:
            if re.search(r'/monetarypolicy/fomcminutes\d{8}\.htm$',link):minute_urls[urljoin(HOST,link)]=page['url']
    with ThreadPoolExecutor(max_workers=3) as pool:detail=list(pool.map(fetch,sorted(minute_urls)))
    summary=[]
    for page in detail:
        lines=[x for x in page['text'].splitlines() if re.search(r'next meeting.*(would|will).*held',x,re.I)]
        summary.append(dict(url=page['url'],release_url=minute_urls[page['url']],next_meeting_lines=lines))
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),audit_end='2026-04-30',
        pages=[{k:v for k,v in p.items() if k not in ['text','links']} for p in pages+press+detail],
        index_candidates=list(candidates.values()),minutes=summary,
        scope='Official annual release indexes and linked FOMC minutes through development end; not a proof of every historical web revision. All historical availability is document publication, never current fetched_at.')
    target=SOURCE/'evidence/source_audit.json';target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(pages=len(report['pages']),index_candidates=len(candidates),minutes=len(summary),
        nonroutine=[c for c in candidates.values() if not re.search('Minutes of|issues FOMC statement|releases FOMC statement',c['title'])],
        missing_next=[r for r in summary if not r['next_meeting_lines']]),indent=2))


if __name__=='__main__':collect()
