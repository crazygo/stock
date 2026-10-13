"""Publication-versioned regular FOMC schedule. No decision text or market label."""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import calendar, hashlib, json, re
from collect_calendar import SOURCE, ROOT, HOST, NOTICES, TIMING, fetch

ET=ZoneInfo('America/New_York')
MONTHS={m:i for i,m in enumerate(calendar.month_name) if m}
MONTH_RE='('+'|'.join(MONTHS)+')'
CATALOG=SOURCE/'calendar.json'


def dt(value):return datetime.fromisoformat(str(value))


def dates(text,year):
    """Parse only the two meeting dates; preserve cross-month and Thu meetings."""
    text=text.replace('–','-').replace('—','-')
    text=re.sub(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b,?','',text)
    matches=list(re.finditer(MONTH_RE+r'\s+(\d{1,2})',text))
    assert matches,text
    first=matches[0];a=datetime(year,MONTHS[first[1]],int(first[2])).date()
    if len(matches)==2:
        last=matches[-1];b=datetime(year,MONTHS[last[1]],int(last[2])).date()
    else:
        m=re.search(r'-\s*(\d{1,2})',text[first.end():]);assert m,text
        b=datetime(year,a.month,int(m[1])).date()
    assert 0<=(b-a).days<=2,(a,b,text)
    return str(a),str(b)


def next_midnight(published):
    t=dt(published).astimezone(ET)
    return datetime.combine(t.date()+timedelta(days=1),datetime.min.time(),ET).isoformat()


def build():
    audit=json.loads((SOURCE/'evidence/source_audit.json').read_text());versions=[];coverage=[]
    published=['2022-06-24T14:00:00-04:00','2023-06-23T11:30:00-04:00','2024-08-09T13:30:00-04:00']
    for name,pub,years in zip(NOTICES,published,[[2023],[2024],[2025,2026]]):
        page=fetch(HOST+'/newsevents/pressreleases/'+name);year=years[0];count={}
        body=page['text'].split('For release at',1)[1].split('For media inquiries',1)[0]
        for line in body.splitlines():
            m=re.fullmatch(r'For (\d{4}):',line)
            if m:year=int(m[1]);continue
            if not (re.match(MONTH_RE+r'\s+\d',line) or re.match(r'(Tuesday|Wednesday),',line)):continue
            explicit=re.search(r', (\d{4})',line);y=int(explicit[1]) if explicit else year
            start,end=dates(line,y);count[y]=count.get(y,0)+1
            versions.append(dict(event_id=f'{y}-{count[y]:02d}',start_date=start,end_date=end,
                statement_at=datetime.fromisoformat(end+'T14:00:00').replace(tzinfo=ET).isoformat(),
                published_at=pub,available_at=next_midnight(pub),source_url=page['url'],source_sha256=page['sha256'],
                fetched_at=page['fetched_at'],tentative=True,kind='annual_notice'))
        for y in years:
            assert count[y]==8,(y,count)
            coverage.append(dict(year=y,available_at=next_midnight(pub),source_url=page['url'],complete_regular_schedule=True))
    by_end={e['end_date']:e for e in versions};confirmations=[];outside=[]
    for m in audit['minutes']:
        assert len(m['next_meeting_lines'])==1,m
        line=m['next_meeting_lines'][0];fragment=line.split('would be held on ',1)[-1].split('.',1)[0]
        year=int(re.findall(r'\b20\d{2}\b',fragment)[0]);start,end=dates(fragment,year)
        if year<2023:outside.append(dict(url=m['url'],reason='2022 outside experiment calendar scope'));continue
        assert end in by_end and start==by_end[end]['start_date'],('REQUIRES AMENDMENT REVIEW',m,start,end)
        press=fetch(m['release_url']);match=re.search(r'For release at ([0-9:]+) p.m. (E[DS]T)',press['text']);assert match,m
        day=re.search(r'(\d{8})[a-z].htm$',m['release_url'])[1]
        hhmm=match[1].split(':');hh=int(hhmm[0])+12;minute=int(hhmm[1]) if len(hhmm)>1 else 0
        pub=datetime.strptime(day,'%Y%m%d').replace(hour=hh,minute=minute,tzinfo=ET).isoformat()
        page=fetch(m['url']);base=by_end[end]
        versions.append({**base,'published_at':pub,'available_at':next_midnight(pub),'source_url':m['url'],
                         'release_url':m['release_url'],'source_sha256':page['sha256'],'fetched_at':page['fetched_at'],
                         'tentative':False,'kind':'minutes_confirmation'})
        confirmations.append(dict(event_id=base['event_id'],url=m['url'],available_at=next_midnight(pub),matches_annual=True))
    timing=fetch(TIMING);assert 'Since March 2013' in timing['text'] and '2 pm' in timing['text']
    result=dict(schema=1,scope='Planned regular FOMC statement timing only; unscheduled policy actions excluded.',
        availability='Conservative next ET calendar day 00:00 after source publication; fetched_at is current retrieval and never historical receipt proof.',
        timing_source=dict(url=TIMING,published_date='2021-10-12',available_at='2021-10-13T00:00:00-04:00',sha256=timing['sha256']),
        versions=sorted(versions,key=lambda e:(e['available_at'],e['event_id'],e['kind'])),coverage=coverage,
        audit=dict(end=audit['audit_end'],annual_unique_events=len(by_end),confirmations=confirmations,
                   outside_scope=outside,amendments_found=0,universal_historical_revision_proof=False))
    CATALOG.write_text(json.dumps(result,indent=2)+'\n')
    print(dict(versions=len(versions),coverage_years=[x['year'] for x in coverage],confirmations=len(confirmations)))


def asof(catalog,at):
    at=dt(at);known={}
    for v in catalog['versions']:
        if dt(v['available_at'])<=at:
            old=known.get(v['event_id'])
            if old is None or dt(v['available_at'])>dt(old['available_at']):known[v['event_id']]=v
    return known


def features(catalog,at,expiry):
    at=dt(at);expiry=dt(expiry);left=(at.astimezone(ET)-timedelta(days=14)).date()
    years=set(range(left.year,expiry.astimezone(ET).year+1))
    covered={c['year'] for c in catalog['coverage'] if c['complete_regular_schedule'] and dt(c['available_at'])<=at}
    known=asof(catalog,at)
    if not years<=covered or dt(catalog['timing_source']['available_at'])>at:
        return dict(known=False,values=None,missing_years=sorted(years-covered),asof=at.isoformat(),expiry=expiry.isoformat())
    events=sorted(known.values(),key=lambda e:dt(e['statement_at']))
    future=[e for e in events if at<dt(e['statement_at'])<=expiry]
    past=[e for e in events if dt(e['statement_at'])<=at]
    before=future[0] if future else None;after=past[-1] if past else None
    def seconds(a,b):return (a.astimezone(ZoneInfo('UTC'))-b.astimezone(ZoneInfo('UTC'))).total_seconds()
    x=max(0.,1-seconds(dt(before['statement_at']),at)/(14*86400)) if before else 0.
    z=max(0.,1-seconds(at,dt(after['statement_at']))/(14*86400)) if after else 0.
    return dict(known=True,values=[x,z],missing_years=[],asof=at.isoformat(),expiry=expiry.isoformat(),
        next_event=before,previous_event=after,coverage_years=sorted(years))


if __name__=='__main__':build()
