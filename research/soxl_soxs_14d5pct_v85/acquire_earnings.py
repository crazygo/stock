"""Bounded public-issuer document acquisition; raw content stays in ignored cache."""
from pathlib import Path
from datetime import datetime,timezone
from urllib.request import Request,urlopen
from urllib.parse import urlparse,urljoin
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess,sys,time,shutil,re
ROOT=Path(__file__).resolve().parents[2];SOURCE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v85';RAW=CACHE/'sources';RAW.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v84'))
from collect_calendar import parse
HOSTS={'NVDA':'https://investor.nvidia.com','AMD':'https://ir.amd.com','AVGO':'https://investors.broadcom.com','MU':'https://investors.micron.com',
       'AMAT':'https://ir.appliedmaterials.com','LRCX':'https://newsroom.lamresearch.com','KLAC':'https://ir.kla.com','INTC':'https://www.intc.com'}
INDEX={'NVDA':'https://nvidianews.nvidia.com/news/','AMD':HOSTS['AMD']+'/news-events/press-releases',
       'AVGO':HOSTS['AVGO']+'/financial-information/financial-news-releases','MU':HOSTS['MU']+'/news/default.aspx','AMAT':HOSTS['AMAT']+'/news-releases',
       'LRCX':HOSTS['LRCX']+'/press-releases?l=100','KLAC':HOSTS['KLAC']+'/news-events/press-releases','INTC':HOSTS['INTC']+'/news-events/press-releases'}
START=time.monotonic();BUDGET=1800
ALLOWED={urlparse(v).hostname for v in HOSTS.values()}|{'www.amd.com','nvidianews.nvidia.com','investor.lamresearch.com','newsroom.intel.com','www.intel.com','s201.q4cdn.com','s205.q4cdn.com','s204.q4cdn.com'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def validate_html(data):
    text=data.decode('utf-8',errors='replace')
    if len(data)<=100 or not re.search(r'</html\s*>',text,re.I):
        raise ValueError('Incomplete HTML: closing document element missing')
    return text

def preserve_rejected(key,data,reason):
    folder=RAW/'rejected';folder.mkdir(exist_ok=True)
    digest=hashlib.sha256(data).hexdigest();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    p=folder/(key+'-'+digest[:12]+'-'+stamp+'.body')
    p.write_bytes(data)
    dump(p.with_suffix('.json'),dict(reason=reason,sha256=digest,bytes=len(data),recorded_at=datetime.now(timezone.utc).isoformat()))

def fetch(url):
    assert urlparse(url).scheme=='https' and urlparse(url).hostname in ALLOWED,url
    key=hashlib.sha256(url.encode()).hexdigest()[:24];path=RAW/(key+'.body');meta=RAW/(key+'.json')
    if path.exists():
        m=json.loads(meta.read_text());assert m['sha256']==sha(path) and m['url']==url
        try:validate_html(path.read_bytes())
        except ValueError:
            rejected=RAW/'rejected';rejected.mkdir(exist_ok=True)
            shutil.move(path,rejected/(key+'-'+m['sha256'][:12]+'.body'))
            shutil.move(meta,rejected/(key+'-'+m['sha256'][:12]+'.json'))
    if not path.exists():
        assert time.monotonic()-START<BUDGET,'Acquisition total budget exceeded'
        now=datetime.now(timezone.utc).isoformat()
        try:
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0 public investor document research'}),timeout=20) as response:
                data=response.read();final=response.url;content_type=response.headers.get('Content-Type','')
            try:validate_html(data)
            except ValueError as e:
                preserve_rejected(key,data,str(e));raise
        except Exception as e:
            with (CACHE/'fetch_failures.jsonl').open('a') as f:f.write(json.dumps(dict(url=url,at=now,error=repr(e)))+'\n')
            # Ordinary HTTP/1.0 retry for incomplete chunked transport; no auth or WAF bypass.
            result=subprocess.check_output(['curl','--http1.0','--location','--fail','--silent','--show-error','--max-time','20','--write-out','\n%{url_effective}',url])
            data,effective=result.rsplit(b'\n',1);final=effective.decode();content_type='unknown_retry'
        try:validate_html(data)
        except ValueError as e:
            preserve_rejected(key,data,str(e));raise
        path.write_bytes(data);m=dict(url=url,final_url=final,received_at=datetime.now(timezone.utc).isoformat(),sha256=sha(path),bytes=len(data),content_type=content_type,cache_file=str(path.relative_to(ROOT)))
        dump(meta,m)
    data=validate_html(path.read_bytes());text,links=parse(data)
    return dict(**m,html=data,text=text,links=links)


def probe():
    def one(item):
        s,url=item
        try:
            p=fetch(url);return s,dict(status='downloaded',**{k:v for k,v in p.items() if k not in ['html','text','links']},
                links=p['links'],text=p['text'])
        except Exception as e:return s,dict(status='failed',url=url,error=repr(e))
    with ThreadPoolExecutor(max_workers=3) as pool:out=dict(pool.map(one,INDEX.items()))
    dump(CACHE/'index_probe.json',out)
    for s,v in out.items():
        print(s,v['status'],v.get('final_url'),v.get('bytes'),v.get('text','')[:100])
        print([x for x in v.get('links',[]) if any(y in x[0] for y in ['page=','Page=','news','press','Press','PressRelease','financial','earnings'])][-12:])

if __name__=='__main__':probe()
