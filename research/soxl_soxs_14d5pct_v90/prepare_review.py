"""Build local audit packets from raw HTML tables, independent of v89 text regex.

All full excerpts stay in ignored cache. This is a review aid, not approval.
"""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v90'
V89=ROOT/'research/soxl_soxs_14d5pct_v89'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def flat(s):return ' '.join(s.split())


class Tables(HTMLParser):
    def __init__(self):
        super().__init__();self.tables=[];self.stack=[]
    def handle_starttag(self,tag,attrs):
        if tag=='table':
            table={'id':len(self.tables),'rows':[],'row':None,'cell':None}
            self.tables.append(table);self.stack.append(table)
        elif self.stack and tag=='tr':self.stack[-1]['row']=[]
        elif self.stack and tag in ['td','th']:self.stack[-1]['cell']=[]
    def handle_data(self,data):
        for t in self.stack:
            if t['cell'] is not None:t['cell'].append(data)
    def handle_endtag(self,tag):
        if not self.stack:return
        t=self.stack[-1]
        if tag in ['td','th'] and t['cell'] is not None:
            if t['row'] is not None:t['row'].append(flat(' '.join(t['cell'])))
            t['cell']=None
        elif tag=='tr' and t['row'] is not None:
            t['rows'].append(t['row']);t['row']=None
        elif tag=='table':self.stack.pop()


def build():
    rows=read(V89/'evidence/revenue_candidates.json')['records'];records=[]
    for row in rows:
        doc=read(ROOT/row['document_file']);raw=ROOT/row['source_file']
        assert sha(raw)==row['source_sha256']
        html=read(raw)['html'] if raw.suffix=='.json' else raw.read_text()
        parser=Tables();parser.feed(html);revenue=[]
        for table in parser.tables:
            for i,cells in enumerate(table['rows']):
                if cells and re.fullmatch(r'(?:Net )?(?:sales|revenue)s?\s*:?|Total revenues?',cells[0],re.I):
                    # Every cell and header is retained: choose the current
                    # quarter only after reading headers, never largest value.
                    revenue.append(dict(table=table['id'],row=i,cells=cells,
                        header=table['rows'][max(0,i-9):i]))
        text=flat(doc['text']);day=datetime.fromisoformat(row['published_date'])
        short='Sept?' if day.month==9 else day.strftime('%b')
        pattern=rf'(?:{day.strftime("%B")}|{short}\.?)\s+0?{day.day},?\s+{day.year}'
        dates=[dict(start=m.start(),text=text[max(0,m.start()-50):m.end()+60])
               for m in re.finditer(pattern,text,re.I)]
        g=row['guidance_candidate'];a=row['actual_candidate']
        packet=dict(id=row['id'],symbol=row['symbol'],published_date=row['published_date'],title=doc['title'],
            source_file=row['source_file'],source_sha256=row['source_sha256'],url=row['url'],
            revenue_tables=revenue,publication_matches=dates,metadata=doc['metadata'],times=doc['times'],
            guidance_context=text[max(0,g['evidence_start']-50):g['evidence_end']+180] if g else None,
            actual_context=doc['text'][max(0,a['evidence_start']-450):a['evidence_end']+100] if a else None,
            candidate=row)
        dump(CACHE/'review_packets'/(row['id']+'.json'),packet)
        records.append(dict(id=row['id'],symbol=row['symbol'],published_date=row['published_date'],
            table_matches=len(revenue),date_matches=len(dates),candidate_kind=row['candidate_kind']))
    dump(CACHE/'review_index.json',records)
    print('packets',len(records),'missing tables',[r for r in records if not r['table_matches'] and r['candidate_kind']=='quarterly_results'])
    print('no visible date match',[r for r in records if not r['date_matches']])


if __name__=='__main__':build()
