"""Assemble official dated rows from preserved HTTP and browser observations."""
from pathlib import Path
import hashlib, json, sys, re
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'research/soxl_soxs_14d5pct_v85'
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v86'
sys.path.insert(0,str(OLD))
from crawl_static import article_rows
from acquire_earnings import parse

PATTERN=re.compile(r'earnings|financial results|financial conference|fiscal.*results|results.*fiscal|webcast',re.I)
ANNOUNCEMENT=re.compile(r'to report|to announce|sets conference call|announces.*(?:earnings|financial conference call)',re.I)


def assemble():
    out={s:dict(symbol=s,pages=[],records={},all_records={}) for s in ['NVDA','AMD','AVGO','MU','AMAT','LRCX','KLAC','INTC']}
    rejected=[]
    def add(s,rows,meta):
        out[s]['pages'].append(dict(meta,rows=len(rows),first=rows[0]['published_date'] if rows else None,last=rows[-1]['published_date'] if rows else None))
        for r in rows:
            out[s]['all_records'][r['url']]=dict(r,index_url=meta['url'],index_evidence=meta['cache_file'])
            if not '2022-09-01'<=r['published_date']<='2026-04-30' or not PATTERN.search(r['title']):continue
            old=out[s]['records'].get(r['url'])
            if old and old['published_date']!=r['published_date']:raise ValueError(('Conflicting index date',old,r))
            out[s]['records'][r['url']]=dict(r,index_url=meta['url'],index_evidence=meta['cache_file'])
    for s in ['NVDA','AMD','LRCX','KLAC']:
        index=json.loads((OLD/'evidence/indexes'/(s+'.json')).read_text())
        for p in index['pages']:
            file=ROOT/p['cache_file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==p['sha256']
            add(s,article_rows(file.read_text(),p['url'],s),p)
    for file in sorted((CACHE/'browser_captures').glob('*.json')):
        d=json.loads(file.read_text());s=d['symbol']
        if d['kind']=='annual_index':
            rows=[]
            for r in d['rows']:
                for fmt in ['%B %d, %Y','%b %d, %Y','%m/%d/%y']:
                    try:day=datetime.strptime(r['date_text'],fmt).date().isoformat();break
                    except ValueError:continue
                else:raise ValueError(('Unknown directory date',r))
                rows.append(dict(symbol=s,url=r['url'],title=r['title'],published_date=day,index_date_text=r['date_text']))
            if not rows:
                rejected.append(dict(path=str(file.relative_to(ROOT)),url=d['url'],reason='Empty annual rows during asynchronous loading'))
                continue
            add(s,rows,dict(url=d['url'],sha256=hashlib.sha256(file.read_bytes()).hexdigest(),cache_file=str(file.relative_to(ROOT)),observed_at=d['observed_at'],method=d['capture_method'],selects=d['selects']))
            continue
        if d['kind']!='index' or s not in ['NVDA','AMD','INTC']:continue
        rows=article_rows(d['html'],d['url'],s)
        if not rows:
            rejected.append(dict(path=str(file.relative_to(ROOT)),url=d['url'],reason='Document observation lacks complete article nodes'))
            continue
        if len(rows)!=({'NVDA':10,'AMD':10,'INTC':12}[s]):raise ValueError(('Truncated index rows',str(file),len(rows)))
        add(s,rows,dict(url=d['url'],sha256=hashlib.sha256(file.read_bytes()).hexdigest(),cache_file=str(file.relative_to(ROOT)),observed_at=d['observed_at'],method=d['capture_method']))
    for s in ['AVGO','MU','AMAT']:
        # Directory transcription is a source pointer, not a body-verified event.
        d=json.loads((OLD/'evidence/indexes'/(s+'.json')).read_text())
        out[s]['directory_transcription']=d
        for r in d['records']:
            if r['kind']=='announcement_title':
                title=r['url'].split('/')[-1] if s!='MU' else r['url'].split('/')[-2]
                out[s]['records'].setdefault(r['url'],dict(r,symbol=s,title=title.replace('-',' '),index_url=d['index_url'],index_evidence=str((OLD/'evidence/indexes'/(s+'.json')).relative_to(ROOT))))
    for s,d in out.items():
        d['records']=sorted(d['records'].values(),key=lambda r:(r['published_date'],r['url']))
        d['all_records']=sorted(d['all_records'].values(),key=lambda r:(r['published_date'],r['url']))
        print(s,len(d['pages']),len(d['records']),sum(ANNOUNCEMENT.search(r['title']) is not None or r.get('kind')=='announcement_title' for r in d['records']))
    (CACHE/'indexes.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    (CACHE/'rejected_index_captures.json').write_text(json.dumps(rejected,indent=2)+'\n')
    return out


if __name__=='__main__':assemble()
