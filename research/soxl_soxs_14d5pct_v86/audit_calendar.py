"""Offline audit of source hashes, candidate coverage and quarterly directory census."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import json
import re
from urllib.parse import urlparse,parse_qs
from prepare_review import digest, read_bodies, candidate, DATE, MONTHS, ROOT, CACHE
from earnings_calendar import load_records, historical_availability

SOURCE=Path(__file__).resolve().parent
SYMBOLS=('NVDA','AMD','AVGO','MU','AMAT','LRCX','KLAC','INTC')


def results_title(title):
    return (bool(re.search(r'reports|announces',title,re.I))
            and bool(re.search(r'quarter|fiscal',title,re.I))
            and bool(re.search(r'results',title,re.I))
            and not re.search(r'preliminary|conference call|to (?:report|announce)',title,re.I))


def audit():
    ix=json.loads((SOURCE/'evidence/indexes.json').read_text())
    reviewed=json.loads((SOURCE/'reviewed_announcements.json').read_text())['announcements']
    corpus=read_bodies(); records=load_records(); all_hashes={}
    for a in reviewed:
        raw=ROOT/a['body_evidence']
        if digest(raw)!=a['body_sha256']:raise ValueError('Reviewed body changed')
        if digest(ROOT/a['original_evidence'])!=a['original_sha256']:raise ValueError('Original body changed')
        all_hashes[a['body_evidence']]=a['body_sha256']
        all_hashes[a['original_evidence']]=a['original_sha256']
        probe=candidate(dict(symbol=a['symbol'],url=a['source_url'],published_date=a['published_date']),corpus[a['source_url']])
        call=next(e for e in a['events'] if e['kind']=='earnings_call')
        if probe['candidate_event_date']!=call['event_date'] or probe['published_at']!=a['published_at']:
            raise ValueError('Reviewed date does not reproduce from body')
        if digest_text(probe['candidate_statement'])!=a['statement_sha256']:
            raise ValueError('Reviewed future statement changed')
    details={};census=[]
    for s in SYMBOLS:
        d=ix[s]
        for p in d['pages']:
            if digest(ROOT/p['cache_file'])!=p['sha256']:raise ValueError('Index changed')
            all_hashes[p['cache_file']]=p['sha256']
        entries=[a for a in reviewed if a['symbol']==s]
        planned={e['event_date'] for a in entries for e in a['events'] if e['kind']!='market_summary'}
        reported=[r for r in d['all_records'] if '2022-10-01'<=r['published_date']<='2026-04-30' and results_title(r['title'])]
        orphan=[r for r in reported if r['published_date'] not in planned]
        pending=sorted(x for x in planned if x>'2026-04-30')
        early=[r for r in d['all_records'] if r['published_date']<'2022-10-01']
        if not early or orphan:raise ValueError(('Incomplete quarterly directory census',s,orphan))
        if len(reported)!=len([x for x in planned if x<='2026-04-30']):raise ValueError(('Unmatched planned activity',s))
        if s in ['AVGO','MU','AMAT']:
            years={datetime.strptime(r['published_date'],'%Y-%m-%d').year for r in d['all_records']}
            if years!={2022,2023,2024,2025,2026}:raise ValueError('Missing annual index')
        elif s in ['NVDA','AMD','INTC']:
            pages={int(parse_qs(urlparse(p['url']).query).get('page',['1'])[0]) for p in d['pages']}
            if pages!=set(range(1,max(pages)+1)):raise ValueError('Missing numbered index page')
        details[s]=dict(index_pages=len(d['pages']),announcements=len(entries),planned_activity_days=len(planned),
            result_directory_titles=len(reported),unmatched_result_titles=len(orphan),future_planned_dates=pending,
            first_announcement=min(a['published_date'] for a in entries),last_announcement=max(a['published_date'] for a in entries))
        for r in reported:census.append(dict(symbol=s,date=r['published_date'],url=r['url'],title=r['title'],
            matched_prior_announcement_urls=[a['source_url'] for a in entries if any(e['event_date']==r['published_date'] for e in a['events'])],
            role='directory_coverage_audit_only_not_predictor_or_actual_occurrence_confirmation'))
    return dict(announcements=len(reviewed),event_versions=len(records),activity_days=sum(d['planned_activity_days'] for d in details.values()),
        result_directory_census=len(census),issuers=details,source_hashes=all_hashes,census=census,
        catalogue_review_passed=True,all_corporate_news_reviewed=False,all_unarchived_revisions_proven=False,
        prospective_received_at_available=False,price_data_added=False,new_model=False,capital_backtest=False,
        declaration='This validates the scoped retrospective quarterly catalogue and conservative publication convention, not complete historical news receipt or predictive performance.')


def digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args();value=audit()
    if args.output:
        with args.output.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ['source_hashes','census']},indent=2))
