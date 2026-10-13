"""Independent reconstruction of source facts and every historical event feature."""
import json, re
from collections import Counter
from state_calibration import *
from collect_calendar import fetch, NOTICES, HOST, TIMING


def reference(catalog,at,expiry):
    at=pd.Timestamp(at);expiry=pd.Timestamp(expiry)
    years=range((at.tz_convert(core.ET).date()-pd.Timedelta(days=14)).year,expiry.tz_convert(core.ET).year+1)
    assert all(any(c['year']==y and pd.Timestamp(c['available_at'])<=at for c in catalog['coverage']) for y in years)
    known={}
    for key in {v['event_id'] for v in catalog['versions']}:
        prior=[v for v in catalog['versions'] if v['event_id']==key and pd.Timestamp(v['available_at'])<=at]
        if prior:known[key]=max(prior,key=lambda v:pd.Timestamp(v['available_at']))
    before=[];after=[]
    for e in known.values():
        t=pd.Timestamp(e['statement_at'])
        if at<t<=expiry:before.append((t,e))
        if t<=at:after.append((t,e))
    n=min(before,key=lambda e:e[0]) if before else None;p=max(after,key=lambda e:e[0]) if after else None
    x=max(0,1-float((n[0]-at)/pd.Timedelta(days=14))) if n else 0
    y=max(0,1-float((at-p[0])/pd.Timedelta(days=14))) if p else 0
    return [x,y],n[1] if n else None,p[1] if p else None


def audit():
    catalog=read(CATALOG);sourceaudit=read(SOURCE/'evidence/source_audit.json')
    for page in sourceaudit['pages']:
        assert sha(ROOT/page['cache_file'])==page['sha256']
    # Dates independently transcribed from the three original dated notices.
    expected={2023:['01-31/02-01','03-21/03-22','05-02/05-03','06-13/06-14','07-25/07-26','09-19/09-20','10-31/11-01','12-12/12-13'],
              2024:['01-30/01-31','03-19/03-20','04-30/05-01','06-11/06-12','07-30/07-31','09-17/09-18','11-06/11-07','12-17/12-18'],
              2025:['01-28/01-29','03-18/03-19','05-06/05-07','06-17/06-18','07-29/07-30','09-16/09-17','10-28/10-29','12-09/12-10'],
              2026:['01-27/01-28','03-17/03-18','04-28/04-29','06-16/06-17','07-28/07-29','09-15/09-16','10-27/10-28','12-08/12-09']}
    factchecks=0
    for y,pairs in expected.items():
        for i,pair in enumerate(pairs):
            starts,ends=pair.split('/');versions=[v for v in catalog['versions'] if v['event_id']==f'{y}-{i+1:02d}']
            assert versions
            for v in versions:
                assert v['start_date']==f'{y}-{starts}' and v['end_date']==f'{y}-{ends}'
                assert pd.Timestamp(v['statement_at'])==pd.Timestamp(f'{y}-{ends} 14:00',tz=core.ET)
                pub=pd.Timestamp(v['published_at']);available=pd.Timestamp(v['available_at'])
                assert available==pd.Timestamp(str((pub.tz_convert(core.ET)+pd.DateOffset(days=1)).date()),tz=core.ET)
                assert v['source_sha256']==fetch(v['source_url'])['sha256']
                if v['kind']=='annual_notice':
                    text=fetch(v['source_url'])['text'];assert pub.strftime('%B %d, %Y') in text
                    hh=pub.hour%12 or 12;mm=':'+str(pub.minute).zfill(2) if pub.minute else ':00'
                    # Original release prose may use 2:00 or 1:30, all three verified.
                    assert f'For release at {hh}{mm} '+('a.m.' if pub.hour<12 else 'p.m.') in text
                else:
                    release=fetch(v['release_url'])['text'];assert pub.strftime('%B %d, %Y') in release
                assert pub<pd.Timestamp(v['statement_at']);factchecks+=1
    assert catalog['audit']['amendments_found']==0 and len(catalog['audit']['confirmations'])==27
    panel=pd.read_parquet(PARENT/'semiconductor_panel.parquet');market=extended_market();prepared=prepare_panel(panel,market)
    rows=[];sources=Counter()
    for at,r in prepared.iterrows():
        days=[d for d in market.sessions if r.date<=d<=r.date+pd.Timedelta(days=13)]
        expiry=pd.Timestamp(market.calendar[max(days)]['close_at']);values,n,p=reference(catalog,at,expiry)
        np.testing.assert_allclose([r.event_before,r.event_after],values,atol=1e-14,rtol=0)
        event=event_features(catalog,at.isoformat(),expiry.isoformat())
        assert event['next_event']==n and event['previous_event']==p and event['known']
        for e in [n,p]:
            if e:sources[e['source_url']]+=1
        rows.append(dict(at=at,date=r.date,before=values[0],after=values[1]))
    t=pd.DataFrame(rows);development=t[t.date.between('2026-01-01','2026-04-30')]
    out=dict(passed=True,calendar_sha256=sha(CATALOG),source_audit_sha256=sha(SOURCE/'evidence/source_audit.json'),
        verified_source_pages=len(sourceaudit['pages']),source_fact_versions_checked=factchecks,query_rows=len(rows),
        dates=panel.date.nunique(),missing_queries=0,before_positive=int((t.before>0).sum()),after_positive=int((t.after>0).sum()),
        development=dict(rows=len(development),before_positive=int((development.before>0).sum()),after_positive=int((development.after>0).sum())),
        used_event_sources=dict(sources),source_sha256=sha(Path(__file__)),new_capital_outcomes_seen=False,
        scope='Official published calendar+minutes confirmation reconstruction; not actual historical receipt or universal web revision proof.')
    dump(SOURCE/'evidence/calendar_verification.json',out);print({k:v for k,v in out.items() if k!='used_event_sources'})


if __name__=='__main__':audit()
