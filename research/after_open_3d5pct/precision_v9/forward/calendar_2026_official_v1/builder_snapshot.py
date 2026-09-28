"""Pure local replay of the captured official 2026 US equity calendars.

Source capture is separate. This module never performs an HTTP request.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from ...timeaxis import add_regular_minutes
from .ledger import calendar_sessions, canonical, file_sha, timestamp

ET = ZoneInfo('America/New_York')
UTC = timezone.utc
START = date(2026, 1, 1)
END = date(2026, 12, 31)
SOURCE_URLS = {'nasdaq': 'https://nasdaqtrader.com/Trader.aspx?id=Calendar',
               'nyse': 'https://www.nyse.com/trade/hours-calendars'}
CALENDAR_ID = 'nasdaq_nyse_us_equities_2026_v1'


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.parts = [], []
        self.table = self.row = self.cell = None
        self.in_script = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.in_script += 1
        if self.in_script:
            return
        if tag == 'table':
            self.table = []
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.in_script:
            return
        clean = ' '.join(data.split())
        if clean:
            self.parts.append(clean)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.in_script = max(0, self.in_script - 1)
            return
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def _table(raw: bytes, header: list[str]) -> tuple[list[list[str]], list[str]]:
    parsed = _Tables()
    parsed.feed(raw.decode('utf-8'))
    matches = [rows for rows in parsed.tables if rows and rows[0] == header]
    if len(matches) != 1:
        raise ValueError(f'official source lacks one expected calendar table: {header}')
    return matches[0][1:], parsed.parts


def _nasdaq_dates(raw: bytes) -> tuple[set[str], set[str]]:
    rows, parts = _table(raw, ['2026', 'Holiday', 'Status'])
    if 'U.S. Equity and Options Markets Holiday Schedule 2026' not in parts:
        raise ValueError('Nasdaq page is not the 2026 equity calendar')
    closed, early = set(), set()
    for row in rows:
        if len(row) != 3:
            raise ValueError('Nasdaq holiday row malformed')
        day = datetime.strptime(row[0], '%B %d, %Y').date()
        if not START <= day <= END or day.weekday() >= 5:
            raise ValueError('Nasdaq holiday date outside 2026 trading weekdays')
        target = closed if row[2] == 'Closed' else early if row[2] == '1:00 p.m.' else None
        if target is None or day.isoformat() in closed | early:
            raise ValueError('Nasdaq holiday status or duplicate changed')
        target.add(day.isoformat())
    if not closed or not early:
        raise ValueError('Nasdaq closure/early-close source incomplete')
    return closed, early


def _nyse_dates(raw: bytes) -> tuple[set[str], set[str]]:
    rows, parts = _table(raw, ['Holiday', '2026', '2027', '2028'])
    closed = set()
    for row in rows:
        if len(row) != 4:
            raise ValueError('NYSE holiday row malformed')
        match = re.match(r'^(Monday|Tuesday|Wednesday|Thursday|Friday), '
                         r'([A-Za-z]+ \d{1,2})(?:\b|$)', row[1])
        if not match:
            raise ValueError('NYSE 2026 holiday date malformed')
        day = datetime.strptime(match.group(2) + ', 2026', '%B %d, %Y').date()
        if day.strftime('%A') != match.group(1) or not START <= day <= END:
            raise ValueError('NYSE holiday weekday/year inconsistent')
        if day.isoformat() in closed:
            raise ValueError('NYSE duplicate holiday')
        closed.add(day.isoformat())
    early = set()
    for description, day in (('Friday, November 27, 2026', '2026-11-27'),
                             ('Thursday, December 24, 2026', '2026-12-24')):
        matches = [part for part in parts
                   if 'Each market will close early at 1:00 p.m.' in part
                   and description in part and 'All times are Eastern Time' in part]
        if len(matches) != 1:
            raise ValueError(f'NYSE 13:00 ET early-close proof absent: {day}')
        early.add(day)
    if not closed or len(early) != 2:
        raise ValueError('NYSE source closure/early-close set incomplete')
    return closed, early


def _load_sources(run_dir: Path):
    envelope = json.loads((run_dir / 'sources.json').read_text())
    records = envelope.get('sources')
    if not isinstance(records, list) or {r.get('name') for r in records} != set(SOURCE_URLS):
        raise ValueError('exactly Nasdaq and NYSE source records required')
    found = {}
    for record in records:
        name = record['name']
        response = urlparse(record.get('response_url', ''))
        if (record.get('requested_url') != SOURCE_URLS[name] or
                response.scheme != 'https' or response.hostname not in
                ({'nasdaqtrader.com', 'www.nasdaqtrader.com'} if name == 'nasdaq'
                 else {'nyse.com', 'www.nyse.com'}) or
                record.get('status') != 200 or
                not str(record.get('content_type', '')).lower().startswith('text/html')):
            raise ValueError(f'{name}: official URL or HTTP status changed')
        retrieved = timestamp(record['retrieved_at_utc'])
        if retrieved > datetime.now(UTC) + timedelta(seconds=5):
            raise ValueError(f'{name}: source retrieval lies in the future')
        path = run_dir / record['raw_path']
        if path.resolve().parent != run_dir.resolve():
            raise ValueError('source path escapes calendar run')
        raw = path.read_bytes()
        if len(raw) != record['raw_size'] or _sha_bytes(raw) != record['raw_sha256']:
            raise ValueError(f'{name}: official raw response bytes changed')
        found[name] = (raw, record)
    return found


def normalize_sessions(closed: set[str], early: set[str]) -> list[dict]:
    if closed & early:
        raise ValueError('a date cannot be closed and early-close')
    for name, values in (('closed', closed), ('early', early)):
        if any(not START <= date.fromisoformat(value) <= END or
               date.fromisoformat(value).weekday() >= 5 for value in values):
            raise ValueError(f'{name} official date outside 2026 trading weekdays')
    sessions = []
    day = START
    while day <= END:
        iso = day.isoformat()
        if day.weekday() < 5 and iso not in closed:
            opening = datetime.combine(day, time(9, 30), ET)
            closing = datetime.combine(day, time(13 if iso in early else 16), ET)
            sessions.append({'session_date': iso,
                             'open_at': opening.isoformat(),
                             'close_at': closing.isoformat(),
                             'duration_minutes': int((closing-opening).total_seconds()/60),
                             'session_type': 'early_close' if iso in early else 'regular'})
        day += timedelta(days=1)
    return sessions


def validate_calendar(document: dict) -> None:
    if (document.get('calendar_id') != CALENDAR_ID or
            document.get('timezone') != 'America/New_York' or
            document.get('coverage_start') != START.isoformat() or
            document.get('coverage_end') != END.isoformat()):
        raise ValueError('calendar identity or 2026 boundary changed')
    rows = document.get('sessions')
    if not isinstance(rows, list) or not rows:
        raise ValueError('normalized session table empty')
    dates = [row.get('session_date') for row in rows]
    if dates != sorted(set(dates)) or not all(START <= date.fromisoformat(d) <= END for d in dates):
        raise ValueError('normalized dates are duplicated, unordered, or outside 2026')
    closed = set(document.get('closed_dates', []))
    early = set(document.get('early_close_dates', []))
    if (document.get('closed_dates') != sorted(closed) or
            document.get('early_close_dates') != sorted(early)):
        raise ValueError('closed or early-close dates duplicated or unordered')
    if rows != normalize_sessions(closed, early):
        raise ValueError('normalized session clock differs from frozen holidays')
    parsed = calendar_sessions({'calendar_sessions': rows})
    if len(parsed) != len(rows):
        raise ValueError('ledger session parser changed row count')
    for row in rows:
        day = date.fromisoformat(row['session_date'])
        start = datetime.combine(day, time(9, 30), ET)
        close = datetime.combine(day, time(13 if row['session_type'] == 'early_close' else 16), ET)
        if (timestamp(row['open_at']) != start.astimezone(UTC) or
                timestamp(row['close_at']) != close.astimezone(UTC) or
                row['duration_minutes'] != (210 if row['session_type'] == 'early_close' else 390)):
            raise ValueError('ET/UTC or DST session conversion changed')
    if document.get('sessions_sha256') != _sha_bytes(canonical(rows).encode()):
        raise ValueError('canonical session hash mismatch')


def build_from_sources(run_dir: Path) -> dict:
    """Offline-only creation. Refuses overwrite and source/code drift."""
    run_dir = Path(run_dir)
    builder_bytes = Path(__file__).read_bytes()
    code_before = _sha_bytes(builder_bytes)
    sources = _load_sources(run_dir)
    nasdaq_closed, nasdaq_early = _nasdaq_dates(sources['nasdaq'][0])
    nyse_closed, nyse_early = _nyse_dates(sources['nyse'][0])
    if nasdaq_closed != nyse_closed or nasdaq_early != nyse_early:
        raise ValueError('Nasdaq and NYSE official 2026 dates disagree')
    rows = normalize_sessions(nasdaq_closed, nasdaq_early)
    document = {'calendar_id': CALENDAR_ID, 'timezone': 'America/New_York',
                'coverage_start': START.isoformat(), 'coverage_end': END.isoformat(),
                'closed_dates': sorted(nasdaq_closed),
                'early_close_dates': sorted(nasdaq_early),
                'sessions_sha256': _sha_bytes(canonical(rows).encode()),
                'sessions': rows}
    validate_calendar(document)
    code_after = file_sha(Path(__file__))
    if code_before != code_after:
        raise ValueError('calendar builder code changed during generation')
    manifest = {'calendar_id': CALENDAR_ID,
                'generated_at_utc': datetime.now(UTC).isoformat(),
                'sources': [sources[name][1] for name in ('nasdaq', 'nyse')],
                'sources_manifest_sha256': file_sha(run_dir / 'sources.json'),
                'normalized_sessions_sha256': document['sessions_sha256'],
                'session_count': len(rows), 'first_session': rows[0]['session_date'],
                'last_session': rows[-1]['session_date'],
                'builder_path': str(Path(__file__).resolve()),
                'builder_snapshot_path': 'builder_snapshot.py',
                'builder_snapshot_sha256': code_before,
                'builder_sha256_before': code_before,
                'builder_sha256_after': code_after}
    with (run_dir / 'builder_snapshot.py').open('xb') as stream:
        stream.write(builder_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    for name, value in (('sessions.json', document), ('manifest.json', manifest)):
        with (run_dir / name).open('x') as stream:
            stream.write(canonical(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return manifest


def verify_artifact(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    document = json.loads((run_dir / 'sessions.json').read_text())
    sources = _load_sources(run_dir)
    closed, early = _nasdaq_dates(sources['nasdaq'][0])
    nyse_closed, nyse_early = _nyse_dates(sources['nyse'][0])
    if closed != nyse_closed or early != nyse_early or \
            closed != set(document['closed_dates']) or early != set(document['early_close_dates']):
        raise ValueError('official sources and normalized dates disagree')
    validate_calendar(document)
    if (manifest['normalized_sessions_sha256'] != document['sessions_sha256'] or
            manifest['session_count'] != len(document['sessions']) or
            manifest['sources_manifest_sha256'] != file_sha(run_dir / 'sources.json') or
            manifest['sources'] != [sources[name][1] for name in ('nasdaq', 'nyse')] or
            manifest['builder_sha256_before'] != manifest['builder_sha256_after'] or
            manifest['builder_sha256_after'] != file_sha(Path(__file__)) or
            manifest['builder_snapshot_sha256'] != manifest['builder_sha256_after'] or
            manifest['builder_snapshot_path'] != 'builder_snapshot.py' or
            file_sha(run_dir / manifest['builder_snapshot_path']) !=
            manifest['builder_snapshot_sha256']):
        raise ValueError('calendar manifest/source/code hash changed')
    return document


def require_coverage(document: dict, cohort_dates: list[str], final_entry_at,
                     horizon_minutes: int = 1950) -> dict:
    """Check coverage only; never creates a cohort or extrapolates into 2027."""
    validate_calendar(document)
    sessions = document['sessions']
    dates = [row['session_date'] for row in sessions]
    if (len(cohort_dates) < 60 or cohort_dates != sorted(set(cohort_dates)) or
            not cohort_dates or cohort_dates[0] not in dates):
        raise ValueError('coverage request needs >=60 unique official sessions')
    first = dates.index(cohort_dates[0])
    if dates[first:first + len(cohort_dates)] != cohort_dates:
        raise ValueError('cohort skips a frozen official session or exceeds 2026')
    entry = timestamp(final_entry_at)
    last = cohort_dates[-1]
    if entry != datetime.combine(date.fromisoformat(last), time(11, 35), ET).astimezone(UTC):
        raise ValueError('last entry is not the frozen 11:35 ET anchor')
    if horizon_minutes != 1950:
        raise ValueError('only the frozen maximum 1950 RTH minutes are supported')
    try:
        label_end = add_regular_minutes(entry, horizon_minutes,
                                        calendar_sessions({'calendar_sessions': sessions}))
    except ValueError as exc:
        raise ValueError('calendar ends before final 1950 RTH minutes') from exc
    if label_end.astimezone(ET).date() > END:
        raise ValueError('1950 RTH minutes extend beyond official 2026 source')
    return {'cohort_sessions': len(cohort_dates), 'first_session': cohort_dates[0],
            'last_session': last, 'final_label_end_at': label_end.isoformat(),
            'calendar_id': CALENDAR_ID, 'coverage_sufficient': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline replay of captured 2026 official calendar')
    parser.add_argument('action', choices=('build', 'verify'))
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = (build_from_sources(args.run_dir) if args.action == 'build'
              else verify_artifact(args.run_dir))
    print(canonical({'calendar_id': result['calendar_id'],
                     'session_count': result.get('session_count', len(result.get('sessions', [])))}))


if __name__ == '__main__':
    main()
