"""Versioned, day-resolution semiconductor earnings features; no trading side effects."""
from pathlib import Path
from datetime import date, timedelta
import hashlib
import json
import sys

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
sys.path.insert(0, str(ROOT/'research/soxl_soxs_14d5pct_v85'))
from announcement_calendar import historical_availability, snapshot, timestamp, validate, ET


def load_records(path=SOURCE/'reviewed_announcements.json'):
    data = json.loads(Path(path).read_text())
    if data['schema'] != 'reviewed_semiconductor_earnings_announcements_v86':
        raise ValueError('Unregistered calendar schema')
    records = []
    for a in data['announcements']:
        if not a['body_verified']: raise ValueError('Unreviewed announcement')
        for event in a['events']:
            kind = event['kind']
            records.append(dict(record_id=hashlib.sha256((a['source_url']+'#'+kind).encode()).hexdigest(),
                event_key=a['activity_key']+'_'+kind, symbol=a['symbol'], kind=kind,
                published_date=a['published_date'], published_at=a['published_at'],
                event_date=event['event_date'], planned_time_et=event['planned_time_et'],
                time_precision=event['time_precision'], source_url=a['source_url'],
                body_verified=True, notes=a['notes'], observed_at=a['observed_at'],
                body_evidence=a['body_evidence'], body_sha256=a['body_sha256']))
    validate(records)
    return records


def query(cutoff, records=None, coverage=None):
    use_frozen = records is None
    records = load_records() if records is None else records
    if coverage is None:
        contract = json.loads((SOURCE/'coverage.json').read_text())
        if use_frozen and hashlib.sha256((SOURCE/'reviewed_announcements.json').read_bytes()).hexdigest() != contract['reviewed_file_sha256']:
            raise ValueError('Calendar differs from the reviewed coverage contract')
        coverage = contract['issuers']
    day = timestamp(cutoff).astimezone(ET).date()
    result = snapshot(records, cutoff, (day+timedelta(days=13)).isoformat(), coverage)
    # Query the latest visible version, not an eventual result/revision.
    earnings = [r for r in result['events'] if r['kind'] in {'earnings_call','earnings_release'}]
    future = sorted({(r['symbol'],r['event_date']) for r in earnings
                     if 0 <= (date.fromisoformat(r['event_date'])-day).days <= 13})
    recent = sorted({(r['symbol'],r['event_date']) for r in earnings
                     if 1 <= (day-date.fromisoformat(r['event_date'])).days <= 14})
    complete = not result['missing_coverage']
    result['features'] = dict(
        planned_activity_days_next14=len(future) if complete else None,
        next14_proximity=max([1-(date.fromisoformat(d)-day).days/14 for _,d in future],default=0.) if complete else None,
        planned_activity_days_previous14=len(recent) if complete else None,
        previous14_proximity=max([1-(day-date.fromisoformat(d)).days/15 for _,d in recent],default=0.) if complete else None,
    )
    result['feature_resolution']='ET calendar day; the full planned day remains in window at every clock'
    result['scope']='Reviewed regular quarterly financial activity catalogue for a fixed eight-company panel; not all sector news or unscheduled events'
    result['actual_event_occurrence_verified']=False
    result['model_or_trade_eligibility']=False
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--cutoff',required=True)
    args=parser.parse_args()
    print(json.dumps(query(args.cutoff),indent=2,ensure_ascii=False))
