"""As-announced quarterly activity context, with explicit day-resolution limits."""
from datetime import date, timedelta
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import sys

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parents[1]
CATALOG=SOURCE/'calendar.json'
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v86'))
from earnings_calendar import load_records, snapshot, timestamp, ET


@lru_cache(maxsize=4)
def inputs(serialized):
    catalog=json.loads(serialized)
    if catalog['schema']!='known_quarterly_activity_inputs_v87':
        raise ValueError('Unregistered calendar')
    for name,digest in catalog['source_files_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen calendar source changed: '+name)
    facts=ROOT/catalog['facts_path'];coverage=json.loads((ROOT/catalog['coverage_path']).read_text())
    if hashlib.sha256(facts.read_bytes()).hexdigest()!=coverage['reviewed_file_sha256']:
        raise ValueError('Calendar differs from coverage contract')
    return load_records(facts),coverage['issuers']


def from_records(records,coverage,at,expiry):
    now=timestamp(str(at)).astimezone(ET);end=timestamp(str(expiry)).astimezone(ET)
    if end<now or (end.date()-now.date()).days>13:
        raise ValueError('Expiry outside original fourteen-day contract')
    view=snapshot(records,now.isoformat(),end.date().isoformat(),coverage)
    earnings=[r for r in view['events'] if r['kind'] in {'earnings_call','earnings_release'}]
    future=sorted({(r['symbol'],r['event_date']) for r in earnings
                   if now.date()<=date.fromisoformat(r['event_date'])<=end.date()})
    recent=sorted({(r['symbol'],r['event_date']) for r in earnings
                   if 1<=(now.date()-date.fromisoformat(r['event_date'])).days<=14})
    complete=not view['missing_coverage']
    values=[len(future),max([1-(date.fromisoformat(d)-now.date()).days/14 for _,d in future],default=0.),
            len(recent),max([1-(now.date()-date.fromisoformat(d)).days/15 for _,d in recent],default=0.)] if complete else None
    return dict(known=complete,values=values,missing_coverage=view['missing_coverage'],
        future_days=[list(x) for x in future],previous_days=[list(x) for x in recent],
        visible_record_ids=sorted(r['record_id'] for r in view['events']),
        actual_event_occurrence_verified=False,exact_intraday_event_window=False)


def features(catalog,at,expiry):
    records,coverage=inputs(json.dumps(catalog,sort_keys=True))
    return from_records(records,coverage,at,expiry)
