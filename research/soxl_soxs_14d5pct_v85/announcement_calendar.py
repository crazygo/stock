"""Publication-aware event evidence. This module neither trains nor recommends trades."""
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo('America/New_York')
PANEL = ('NVDA', 'AMD', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC', 'INTC')
EARNINGS_KINDS = {'earnings_release', 'earnings_call'}


def timestamp(value):
    t = datetime.fromisoformat(value)
    if t.tzinfo is None:
        raise ValueError('Timestamp requires an explicit UTC offset')
    return t


def historical_availability(record):
    """Declared simulation convention; never relabel this as historical receipt.

    Exact publication: next ET midnight. Date without a verified timezone:
    midnight ET two calendar dates later, preserving published_at=None.
    This delay covers the full stated date even at UTC-12. It does not prove
    that an operational system actually collected the publication then.
    """
    if record.get('published_at'):
        day = timestamp(record['published_at']).astimezone(ET).date() + timedelta(days=1)
    else:
        day = date.fromisoformat(record['published_date']) + timedelta(days=2)
    return datetime.combine(day, time(), ET)


def validate(records):
    ids = set()
    versions = set()
    event_types = {}
    for r in records:
        if r['record_id'] in ids:
            raise ValueError('Duplicate record id')
        ids.add(r['record_id'])
        if r['symbol'] not in PANEL or not r['source_url'].startswith('https://'):
            raise ValueError('Unregistered issuer or missing source')
        if not r['body_verified']:
            raise ValueError('Title-only evidence cannot be a calendar fact')
        if r['kind'] not in EARNINGS_KINDS | {'market_summary'}:
            raise ValueError('Unregistered event kind')
        identity = (r['symbol'], r['kind'])
        if event_types.setdefault(r['event_key'], identity) != identity:
            raise ValueError('A revision cannot change issuer or event kind')
        precision = r['time_precision']
        if precision not in {'minute', 'approximate_minute', 'after_regular_close', 'date_only', 'local_time_unresolved'}:
            raise ValueError('Unregistered precision')
        if bool(r.get('planned_time_et')) != (precision in {'minute', 'approximate_minute'}):
            raise ValueError('Time precision and ET time disagree')
        day = date.fromisoformat(r['event_date'])
        if day < date.fromisoformat(r['published_date']):
            raise ValueError('Retrospective result cannot masquerade as preannouncement')
        if r.get('published_at'):
            # published_date is the source-local date, not a converted ET date.
            if timestamp(r['published_at']).date().isoformat() != r['published_date']:
                raise ValueError('Publication date and timestamp disagree')
        if r.get('planned_time_et'):
            time.fromisoformat(r['planned_time_et'])
        version = (r['event_key'], historical_availability(r))
        if version in versions:
            raise ValueError('Ambiguous event versions at the same availability cutoff')
        versions.add(version)


def snapshot(records, cutoff, deadline, coverage):
    """Select only visible announcement versions; missing coverage stays unknown.

    coverage contains issuer-specific, body-reviewed publication catalogues.
    An expired planned date is not proof that results were actually released.
    count_in_window is distinct from observed_events_in_window when incomplete.
    """
    validate(records)
    now = timestamp(cutoff).astimezone(ET)
    end = date.fromisoformat(deadline)
    if end < now.date():
        raise ValueError('Window ends before the query date')
    missing = []
    for symbol in PANEL:
        c = coverage.get(symbol, {})
        if not (c.get('body_catalogue_verified') is True
                and c.get('publication_from', '9999') <= now.date().isoformat()
                <= c.get('publication_through', '0000')
                and c.get('pending_announcements_at_start_resolved') is True):
            missing.append(symbol)
    latest = {}
    for r in records:
        available = historical_availability(r)
        if available > now:
            continue
        old = latest.get(r['event_key'])
        if old is None or available > historical_availability(old):
            latest[r['event_key']] = r
    visible = []
    for key, r in sorted(latest.items()):
        day = date.fromisoformat(r['event_date'])
        visible.append(dict(r, historical_available_at=historical_availability(r).isoformat(),
                            temporal_state='planned_date_elapsed' if day < now.date() else 'planned',
                            in_window=now.date() <= day <= end))
    # A call and a results release on the same issuer/day are one earnings day.
    days = sorted({(r['symbol'], r['event_date']) for r in visible
                   if r['in_window'] and r['kind'] in EARNINGS_KINDS})
    return dict(cutoff=now.isoformat(), deadline=end.isoformat(),
                missing_coverage=missing, events=visible, observed_earnings_days=days,
                count_in_window=None if missing else len(days),
                coverage_status='incomplete' if missing else 'reviewed_catalogue',
                availability_basis='historical_simulation_assumption',
                actual_event_occurrence_verified=False, model_or_trade_eligibility=False)
