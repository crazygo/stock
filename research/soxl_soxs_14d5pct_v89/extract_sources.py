"""Extract auditable text/metadata from preserved issuer documents; no model features."""
from html.parser import HTMLParser
from datetime import datetime,timezone
import json,re
from catalogue import *

class Document(HTMLParser):
    def __init__(self):
        super().__init__();self.parts=[];self.skip=0;self.title=[];self.in_title=False;self.meta=[];self.times=[];self.time=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag in ['script','style','noscript']:self.skip+=1
        if tag=='title':self.in_title=True
        if tag=='meta':self.meta.append(a)
        if tag=='time':self.time={'datetime':a.get('datetime'),'text':[]}
        if tag in ['p','div','li','h1','h2','h3','h4','tr','table','time','br']:self.parts.append('\n')
        if tag in ['td','th']:self.parts.append('\t')
    def handle_endtag(self,tag):
        if tag in ['script','style','noscript']:self.skip=max(0,self.skip-1)
        if tag=='title':self.in_title=False
        if tag=='time' and self.time:
            self.time['text']=' '.join(''.join(self.time['text']).split());self.times.append(self.time);self.time=None
        if tag in ['p','div','li','h1','h2','h3','h4','tr','table','time']:self.parts.append('\n')
    def handle_data(self,text):
        if self.skip:return
        self.parts.append(text)
        if self.in_title:self.title.append(text)
        if self.time:self.time['text'].append(text)
    def result(self):return dict(text='\n'.join(' '.join(x.split()) for x in ''.join(self.parts).splitlines() if x.strip()),title=''.join(self.title),metadata=self.meta,times=self.times)
def normalize(x):return re.sub(r'[^a-z0-9]','',x.lower())
def extract():
    rows=json.loads((SOURCE/'catalogue.json').read_text())['records'];http=json.loads((CACHE/'http_manifest.json').read_text());H={r['url']:r for r in http['records']};B={}
    for file in (CACHE/'browser_captures').glob('*.json'):
        d=json.loads(file.read_text());B.setdefault(d['url'],[]).append((file,d))
    records=[];fail=[]
    for r in rows:
        h=H[r['url']]
        if h['status']=='html_body_review_pending':
            file=ROOT/h['cache_file'];assert sha(file)==h['sha256'];parser=Document();parser.feed(file.read_text());d=parser.result();d.update(received_at=h['received_at'],capture_method='ordinary_http')
        elif h['status']=='browser_body_review_pending':
            file=ROOT/h['cache_file'];assert sha(file)==h['sha256'];d=json.loads(file.read_text());d['received_at']=d['observed_at']
        elif r['url'] in B:
            file,d=max(B[r['url']],key=lambda x:x[1]['observed_at']);d=dict(d,received_at=d['observed_at'])
        else:
            fail.append(dict(r,reason='Body unavailable'));continue
        # Browser observations retain original DOM text; independent HTML-derived
        # text is stored as another reading path, not a second source.
        if 'html' in d:
            parser=Document();parser.feed(d['html']);html_reading=parser.result();d['html_text']=html_reading['text']
        identity=normalize(r['title']) in normalize(d['text']) or normalize(r['title']) in normalize(d.get('title',''))
        substantive=len(d['text'])>2000 and bool(re.search(r'revenue|reaffirms current .* quarter guidance',d['text'],re.I))
        if not identity or not substantive:fail.append(dict(r,reason='Title/body identity or revenue content check failed',identity=identity,substantive=substantive));continue
        result=dict(r,source_file=str(file.relative_to(ROOT)),source_sha256=sha(file),received_at=d['received_at'],capture_method=d.get('capture_method'),document_title=d.get('title'),text=d['text'],metadata=d.get('metadata',[]),times=d.get('times',[]),html_text=d.get('html_text'),
            extraction_state='source_identity_checked_numeric_review_pending')
        dump(CACHE/'documents'/(r['id']+'.json'),result)
        lines=d['text'].splitlines();snippets=[]
        for i,line in enumerate(lines):
            if re.search(r'outlook|guidance|revenue (?:is|of|was)|revenues? for|net sales|quarter ended',line,re.I):
                snippets.append({'line':i+1,'text':' '.join(lines[max(0,i-1):min(len(lines),i+5)])[:1200]})
        dump(CACHE/'review'/(r['symbol']+'_'+r['published_date']+'_'+r['id']+'.json'),dict(symbol=r['symbol'],published_date=r['published_date'],url=r['url'],snippets=snippets))
        records.append({k:v for k,v in result.items() if k not in ['text','html_text','metadata','times']})
    out=dict(created_at=datetime.now(timezone.utc).isoformat(),requested=len(rows),body_identity_checked=len(records),records=records,failures=fail,
        no_numeric_model_features=True,no_guidance_history_completeness_claim=True)
    dump(SOURCE/'evidence/source_inventory.json',out);print({'requested':len(rows),'body_identity_checked':len(records),'failures':fail})
if __name__=='__main__':extract()
