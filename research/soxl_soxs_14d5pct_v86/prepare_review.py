"""Offline source reconciliation and candidate extraction; never marks facts verified."""
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import sys
from assemble_indexes import ANNOUNCEMENT, ROOT, CACHE

MONTHS = {m[:3].lower(): n for n, m in enumerate(
    ['January','February','March','April','May','June','July','August','September','October','November','December'],1)}
DATE = re.compile(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+(\d{1,2})(?:,?\s+(20\d{2}))?\b', re.I)
STARTS = {
 'NVDA': r'will host a conference call',
 'AMD': r'AMD will hold a conference call|(?:it|AMD) will report',
 'AVGO': r'will report its',
 'MU': r'will hold its',
 'AMAT': r'will hold a webcast|will hold its|will report',
 'LRCX': r'will host its quarterly financial conference call',
 'KLAC': r'will review',
 'INTC': r'will report',
}
EMBEDDED = {'1093': '2022-10-06', '1112': '2023-01-11', '1260': '2025-10-06'}
LRCX_BAD = 'https://newsroom.lamresearch.com/2022-09-28-Lam-Research-Corporation-Announces-September-2022-Quarter-Financial-Conference-Call'
LRCX_GOOD = 'https://www.globenewswire.com/news-release/2022/09/28/2524713/0/en/lam-research-corporation-announces-september-2022-quarter-financial-conference-call.html'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_bodies():
    corpus = {}
    for path in sorted((ROOT/'.cache/soxl_soxs_14d5pct_v85/bodies').glob('*.json')):
        d = json.loads(path.read_text())
        if d['status'] == 'failed': continue
        source = d['source']; raw = ROOT/source['cache_file']
        if digest(raw) != source['sha256']: raise ValueError('Raw source changed')
        corpus[d['url']] = dict(text=d['text'], observed_at=source['received_at'],
            body_evidence=str(path.relative_to(ROOT)), body_sha256=digest(path),
            original_evidence=source['cache_file'], original_sha256=source['sha256'])
    for path in sorted((CACHE/'browser_captures').glob('*.json')):
        d = json.loads(path.read_text())
        if d['kind'] != 'announcement_body': continue
        if path.stem != digest(path): raise ValueError('Capture changed')
        corpus[d['url']] = dict(text=d['text'], observed_at=d['observed_at'],
            body_evidence=str(path.relative_to(ROOT)), body_sha256=digest(path),
            original_evidence=str(path.relative_to(ROOT)), original_sha256=digest(path))
    return corpus


def parse_publication(text, published_date):
    """Only an explicit body timestamp may increase publication precision."""
    for m in DATE.finditer(text):
        if not m[3]: continue
        day = date(int(m[3]), MONTHS[m[1][:3].lower()], int(m[2]))
        if day.isoformat() != published_date: continue
        tail = text[m.end():m.end()+45].strip()
        tm = re.match(r'(?:(?:at|•)\s+)?(\d{1,2}):(\d{2})\s*(AM|PM)?\s*(EDT|EST|ET)\b', tail, re.I)
        if tm:
            h, minute = int(tm[1]), int(tm[2])
            if tm[3]: h = h%12 + (12 if tm[3].upper()=='PM' else 0)
            zone = tm[4].upper()
            tz = ZoneInfo('America/New_York') if zone=='ET' else timezone(timedelta(hours=-4 if zone=='EDT' else -5))
            return datetime.combine(day, datetime.min.time(),tz).replace(hour=h,minute=minute).isoformat()
    return None


def candidate(record, body):
    text = body['text'].replace('\xa0',' ')
    start = re.search(STARTS[record['symbol']], text, re.I)
    if record['symbol']=='AMD':
        start = re.search(r'(?:it|AMD) will report',text,re.I) or start
    if not start: raise ValueError('No supported future earnings statement')
    # Parsing is limited to the future statement, excluding headers, recommendations,
    # navigation, result tables and current quote widgets.
    passage = text[start.start():start.start()+900]
    m = DATE.search(passage)
    if not m: raise ValueError('Future statement has no date')
    pub = date.fromisoformat(record['published_date'])
    year = int(m[3]) if m[3] else pub.year
    event = date(year, MONTHS[m[1][:3].lower()], int(m[2]))
    if not m[3] and event < pub: event = event.replace(year=year+1)
    if not 0 <= (event-pub).days <= 100: raise ValueError('Implausible announcement lead time')
    # This is candidate evidence only. Human review determines kinds and semantics.
    return dict(symbol=record['symbol'], source_url=record['url'],
        published_date=pub.isoformat(), published_at=parse_publication(text,pub.isoformat()),
        candidate_event_date=event.isoformat(), candidate_statement=passage,
        body_verified=False, **{k:v for k,v in body.items() if k!='text'})


def prepare():
    indexes = json.loads((CACHE/'indexes.json').read_text())
    records = [dict(r,symbol=s) for s,d in indexes.items() for r in d['records']
               if ANNOUNCEMENT.search(r['title']) or r.get('kind')=='announcement_title']
    corpus = read_bodies()
    for ident, day in EMBEDDED.items():
        matches = [u for u in corpus if '/detail/'+ident+'/' in u]
        if len(matches)!=1: raise ValueError('Missing embedded AMD announcement')
        records.append(dict(symbol='AMD',url=matches[0],published_date=day))
    output=[];errors=[]
    for r in records:
        original = r['url']
        if original==LRCX_BAD:r=dict(r,url=LRCX_GOOD)
        try:
            result=candidate(r,corpus[r['url']])
            if original!=r['url']:
                result['conflicting_source_url']=original
                result['conflict_reason']='Issuer archive body describes a different August investor event; use issuer-issued original distribution.'
            output.append(result)
        except (ValueError,KeyError) as exc: errors.append(dict(record=r,error=str(exc)))
    output.sort(key=lambda r:(r['symbol'],r['published_date'],r['source_url']))
    (CACHE/'candidate_review.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    (CACHE/'candidate_errors.json').write_text(json.dumps(errors,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(candidates=len(output),errors=errors),indent=2))
    return output


if __name__=='__main__':prepare()
