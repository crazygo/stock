"""Append-only all-candidate label store; separate from the BUY ledger."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from .labels import evaluate_candidate
from .ledger import ET, TARGETS, _at, canonical, digest, file_sha, timestamp


def _source_proof(record: dict, *, live: bool) -> bool:
    expected = record.get('source_sha256')
    if not expected:
        return False
    path = record.get('path')
    if live and not path:
        return False
    if path:
        try:
            source = Path(path)
            content = {k: v for k, v in record.items()
                       if k not in ('source_sha256', 'path')}
            return file_sha(source) == expected and json.loads(source.read_text()) == content
        except (OSError, ValueError):
            return False
    return expected == digest({k: v for k, v in record.items()
                               if k not in ('source_sha256', 'path')})


def _future_eligibility_key(value) -> bool:
    forbidden = ('label', 'outcome', 'score', 'threshold', 'prediction',
                 'future', 'buy', 'signal', 'route', 'model')
    if isinstance(value, dict):
        return any((str(key).lower() in ('y', 'raw', 'processed') or
                    any(word in str(key).lower() for word in forbidden) or
                    _future_eligibility_key(child)) for key, child in value.items())
    if isinstance(value, list):
        return any(_future_eligibility_key(child) for child in value)
    return False


class PriorStore:
    def __init__(self, path: Path, *, mode='fixture', clock=None):
        if mode not in ('fixture', 'historical_fixture', 'live'):
            raise ValueError('invalid prior-store mode')
        if mode != 'fixture' and clock is not None:
            raise ValueError('live/history import cannot inject a past clock')
        self.path, self.mode, self.clock = Path(path), mode, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,candidate_id TEXT NOT NULL,
                symbol TEXT NOT NULL,session_date TEXT NOT NULL,recorded_at TEXT NOT NULL,
                payload_json TEXT NOT NULL,payload_sha TEXT NOT NULL,
                prev_hash TEXT NOT NULL,event_hash TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS one_candidate_registration
                ON events(candidate_id,event_type) WHERE event_type='candidate_registered';
            CREATE UNIQUE INDEX IF NOT EXISTS one_session_manifest
                ON events(candidate_id,event_type) WHERE event_type='session_registered';
            CREATE INDEX IF NOT EXISTS candidate_outcomes
                ON events(candidate_id,event_type,seq);
            CREATE TRIGGER IF NOT EXISTS no_prior_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT,'append-only prior events'); END;
            CREATE TRIGGER IF NOT EXISTS no_prior_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT,'append-only prior events'); END;
        ''')
        old = self.db.execute("SELECT value FROM metadata WHERE key='mode'").fetchone()
        if old and old['value'] != mode:
            self.db.close()
            raise ValueError('prior-store fixture/live mode mismatch')
        if not old:
            self.db.execute("INSERT INTO metadata(key,value) VALUES('mode',?)", (mode,))

    def close(self):
        self.db.close()

    def now(self):
        return timestamp(self.clock() if self.clock else datetime.now(timezone.utc))

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def append(self, event_type, candidate_id, symbol, session_date, payload):
        old = self.db.execute('SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        row = {'event_id': str(uuid4()), 'event_type': event_type,
               'candidate_id': candidate_id, 'symbol': symbol,
               'session_date': session_date, 'recorded_at': self.now().isoformat(),
               'payload_sha': digest(payload),
               'prev_hash': old['event_hash'] if old else '0' * 64}
        row['event_hash'] = digest(row)
        row['payload_json'] = canonical(payload)
        fields = list(row)
        cursor = self.db.execute(f"INSERT INTO events({','.join(fields)}) VALUES({','.join('?' for _ in fields)})",
                                 tuple(row.values()))
        return dict(row, seq=cursor.lastrowid, payload=payload)

    def verify_chain(self, cut=None):
        rows = self.db.execute('SELECT * FROM events WHERE seq<=? ORDER BY seq',
                               (cut if cut is not None else 2**63 - 1,)).fetchall()
        prior = '0' * 64
        for number, row in enumerate(rows, 1):
            fields = dict(row)
            payload = json.loads(fields.pop('payload_json'))
            recorded_hash = fields.pop('event_hash')
            if fields.pop('seq') != number or fields['prev_hash'] != prior or \
                    digest(payload) != fields['payload_sha'] or digest(fields) != recorded_hash:
                raise ValueError(f'prior-store hash chain damaged at {number}')
            prior = recorded_hash
        return len(rows)

    def events(self, cut=None, event_type=None):
        if cut is not None and (not isinstance(cut, int) or cut < 0 or cut > self.verify_chain()):
            raise ValueError('future or invalid prior-store cut')
        else:
            self.verify_chain()
        sql = 'SELECT * FROM events WHERE seq<=?'
        args = [cut if cut is not None else 2**63 - 1]
        if event_type:
            sql += ' AND event_type=?'
            args.append(event_type)
        return [dict(row, payload=json.loads(row['payload_json'])) for row in
                self.db.execute(sql + ' ORDER BY seq', args)]


def register_candidates(universe_version: dict, official_session: dict,
                        eligibility_manifest: dict, store: PriorStore) -> dict:
    """Freeze one candidate row for every universe member, independent of route."""
    universe_id = universe_version.get('universe_version') or universe_version.get('id')
    day = official_session.get('session_date')
    label_contract = eligibility_manifest.get('label_contract', 'v9_11_35_390_rth_none')
    members = universe_version.get('members')
    if not universe_id or not day or not isinstance(members, list) or not members:
        raise ValueError('frozen universe/session missing')
    candidate_symbols = [m['symbol'] if isinstance(m, dict) else m for m in members
                         if not isinstance(m, dict) or m.get('role') == 'candidate']
    if len(set(candidate_symbols)) != len(candidate_symbols) or 'QQQ' in candidate_symbols:
        raise ValueError('duplicate or benchmark candidate in frozen universe')
    rows = eligibility_manifest.get('rows')
    if not isinstance(rows, list) or len(rows) != len(candidate_symbols) or \
            len({r.get('symbol') for r in rows}) != len(rows) or \
            {r.get('symbol') for r in rows} != set(candidate_symbols):
        raise ValueError('eligibility manifest omits or adds frozen candidates')
    if any(_future_eligibility_key(r) for r in rows):
        raise ValueError('candidate eligibility may not inspect outcomes or action scores')
    if any(type(r.get('eligible')) is not bool or not r.get('sample_id') for r in rows):
        raise ValueError('each candidate needs stable sample_id and boolean eligibility')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate candidate sample IDs')
    open_at, close_at = timestamp(official_session['open_at']), timestamp(official_session['close_at'])
    if not open_at < _at(day, 11, 35) < close_at or \
            open_at.astimezone(ET).date().isoformat() != day:
        raise ValueError('official session cannot support frozen decision/entry')
    if not _source_proof(universe_version, live=store.mode == 'live') or \
            not _source_proof(eligibility_manifest, live=store.mode == 'live'):
        raise ValueError('universe or eligibility source proof missing or changed')
    info_deadline = _at(day, 11, 30, 30)
    if store.mode == 'live':
        for record in (universe_version, eligibility_manifest):
            received = timestamp(record['received_at'])
            if received > info_deadline or received > store.now():
                raise ValueError('live candidate source was not received by decision')
    candidate_ids = []
    for row in sorted(rows, key=lambda item: item['symbol']):
        symbol = row['symbol']
        candidate_ids.append(f'{universe_id}|{symbol}|{day}|11:30|{label_contract}')
    manifest_payload = {'universe_version': universe_id, 'session_date': day,
                        'candidate_ids': candidate_ids,
                        'universe_sha256': universe_version['source_sha256'],
                        'universe_path': universe_version.get('path'),
                        'eligibility_sha256': eligibility_manifest['source_sha256'],
                        'eligibility_path': eligibility_manifest.get('path'),
                        'official_session': official_session,
                        'label_contract': label_contract}
    manifest_id = f'{universe_id}|{day}|{label_contract}'
    if store.mode == 'live' and store.now() > info_deadline:
        raise ValueError('live candidate registration missed decision deadline')
    store.verify_chain()
    with store.transaction():
        old = store.db.execute("SELECT * FROM events WHERE event_type='session_registered' AND candidate_id=?",
                               (manifest_id,)).fetchone()
        if old:
            if json.loads(old['payload_json']) != manifest_payload:
                raise ValueError('frozen candidate session changed')
            return {'candidate_ids': candidate_ids,
                    'coverage': {'registered': len(rows),
                                 'eligible': sum(r['eligible'] for r in rows)},
                    'manifest_event_id': old['event_id']}
        manifest_event = store.append('session_registered', manifest_id, '', day, manifest_payload)
        for candidate_id, row in zip(candidate_ids, sorted(rows, key=lambda item: item['symbol'])):
            existing_sample = store.db.execute("SELECT candidate_id FROM events WHERE event_type='candidate_registered' AND json_extract(payload_json,'$.sample_id')=?",
                                               (row['sample_id'],)).fetchone()
            if existing_sample:
                raise ValueError('sample_id already registered in another candidate')
            store.append('candidate_registered', candidate_id, row['symbol'], day,
                         {'universe_version': universe_id, 'sample_id': row['sample_id'],
                          'eligible': row['eligible'], 'reason': row.get('reason'),
                          'eligibility': row,
                          'cutoff_at': _at(day, 11, 30).isoformat(),
                          'decision_at': _at(day, 11, 30).isoformat(),
                          'information_deadline_at': info_deadline.isoformat(),
                          'planned_entry_at': _at(day, 11, 35).isoformat(),
                          'label_contract': label_contract,
                          'universe_sha256': universe_version['source_sha256'],
                          'eligibility_sha256': eligibility_manifest['source_sha256']})
    return {'candidate_ids': candidate_ids,
            'coverage': {'registered': len(rows), 'eligible': sum(r['eligible'] for r in rows)},
            'manifest_event_id': manifest_event['event_id']}


def update_candidate_outcomes(candidate_ids, local_outcome_snapshot, calendar,
                              asof, store: PriorStore):
    """Append one full nine-target version per candidate, including non-BUYs."""
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError('unique registered candidate IDs required')
    observed_at = timestamp(asof)
    if store.mode != 'fixture' and abs((observed_at - store.now()).total_seconds()) > 5:
        raise ValueError('live/history import uses the actual current clock')
    known = {e['candidate_id']: e for e in store.events(event_type='candidate_registered')}
    if set(candidate_ids) - known.keys():
        raise ValueError('outcome update refers to unregistered candidate')
    evaluated = []
    for candidate_id in sorted(candidate_ids):
        registered = known[candidate_id]
        r = registered['payload']
        candidate = {'sample_id': r['sample_id'], 'symbol': registered['symbol'],
                     'decision_at': r['decision_at'],
                     'planned_entry_at': r['planned_entry_at']}
        result = evaluate_candidate(candidate, local_outcome_snapshot, calendar, observed_at)
        snapshot = (local_outcome_snapshot if 'bars' in local_outcome_snapshot else
                    local_outcome_snapshot[registered['symbol']])
        actions = snapshot.get('corporate_actions') or {}
        if store.mode == 'live' and (not snapshot.get('path') or not snapshot.get('received_at') or
                                      not actions.get('path') or not actions.get('received_at')):
            raise ValueError('live label source/action receipt proof missing')
        if store.mode == 'live' and (timestamp(snapshot['received_at']) > store.now() or
                                      timestamp(actions['received_at']) > store.now()):
            raise ValueError('live label source/action receipt lies in the future')
        payload = {'candidate_id': candidate_id, 'sample_id': r['sample_id'],
                   'evaluated_at': observed_at.isoformat(),
                   'result': result, 'source_sha256': snapshot['source_sha256'],
                   'source_path': snapshot.get('path'),
                   'source_received_at': snapshot.get('received_at'),
                   'action_sha256': actions.get('source_sha256'),
                   'action_path': actions.get('path'),
                   'action_received_at': actions.get('received_at'),
                   'calendar_sha256': digest(calendar)}
        evaluated.append((registered, payload))
    appended = []
    store.verify_chain()
    with store.transaction():
        for registered, payload in evaluated:
            candidate_id = registered['candidate_id']
            previous = store.db.execute("SELECT * FROM events WHERE event_type='outcome_version' AND candidate_id=? ORDER BY seq DESC LIMIT 1",
                                        (candidate_id,)).fetchone()
            if previous:
                before = json.loads(previous['payload_json'])
                without_revision = dict(before)
                without_revision.pop('supersedes', None)
                if without_revision == payload:
                    continue
            payload['supersedes'] = previous['event_id'] if previous else None
            appended.append(store.append('outcome_version', candidate_id,
                                          registered['symbol'], registered['session_date'], payload))
    return appended


def read_prior_asof(symbols, information_deadline, store_cut, store: PriorStore):
    """Seal a view through the actual capture clock, never a future deadline."""
    deadline = timestamp(information_deadline)
    captured_at = store.now()
    observed_through = min(deadline, captured_at)
    if type(store_cut) is not int or store_cut < 0:
        raise ValueError('explicit nonnegative integer prior-store cut required')
    if isinstance(symbols, str):
        raise ValueError('prior view symbols must be an iterable of symbol IDs')
    symbols = sorted(set(symbols))
    if not symbols:
        raise ValueError('prior view needs requested symbols')
    events = store.events(cut=store_cut)
    for event in events:
        if event['event_type'] == 'session_registered':
            payload = event['payload']
            for role in ('universe', 'eligibility'):
                path = payload.get(f'{role}_path')
                if path and file_sha(path) != payload[f'{role}_sha256']:
                    raise ValueError(f'{role} candidate source bytes changed')
    registrations = {e['candidate_id']: e for e in events if e['event_type'] == 'candidate_registered'}
    versions = {}
    coverage = {'registered': len(registrations), 'eligible': 0, 'ineligible': 0,
                'mature_selected': 0, 'pending': 0, 'missing': 0,
                'invalid': 0, 'evidence_unknown': 0, 'without_outcome': 0}
    for e in events:
        if e['event_type'] == 'outcome_version':
            versions.setdefault(e['candidate_id'], []).append(e)
    selected, rows = [], []
    for candidate_id, registered in sorted(registrations.items()):
        registration = registered['payload']
        if not registration['eligible']:
            coverage['ineligible'] += 1
            continue
        coverage['eligible'] += 1
        candidates = versions.get(candidate_id, [])
        if not candidates:
            coverage['without_outcome'] += 1
            continue
        chosen = None
        for event in candidates:
            result = event['payload']['result']
            source_received = event['payload'].get('source_received_at')
            action_received = event['payload'].get('action_received_at')
            known = [timestamp(event['recorded_at'])]
            if result.get('label_available_at'):
                known.append(timestamp(result['label_available_at']))
            if source_received:
                known.append(timestamp(source_received))
            if action_received:
                known.append(timestamp(action_received))
            known_at = max(known)
            if known_at <= observed_through:
                chosen = event, known_at
        if chosen is None:
            coverage['pending'] += 1
            continue
        event, known_at = chosen
        for role in ('source', 'action'):
            path = event['payload'].get(f'{role}_path')
            if path and file_sha(path) != event['payload'][f'{role}_sha256']:
                raise ValueError(f'{role} outcome source bytes changed')
        result = event['payload']['result']
        if not result['all_nine_mature']:
            statuses = {x['status'] for x in result['targets'].values()}
            category = ('evidence_unknown' if 'evidence_unknown' in statuses else
                        'invalid' if 'invalid' in statuses or 'halts' in statuses else
                        'missing' if 'missing' in statuses else 'pending')
            coverage[category] += 1
            continue
        y = result['y']
        if not isinstance(y, list) or len(y) != len(TARGETS) or any(type(v) is not int or v not in (0, 1) for v in y):
            raise ValueError('mature prior row has nonbinary or incomplete nine labels')
        if result['sample_id'] != registration['sample_id']:
            raise ValueError('sample ID changed in outcome version')
        coverage['mature_selected'] += 1
        selected.append({'candidate_id': candidate_id, 'event_id': event['event_id'],
                         'event_seq': event['seq'],
                         'source_sha256': event['payload']['source_sha256'],
                         'action_sha256': event['payload']['action_sha256'],
                         'calendar_sha256': event['payload']['calendar_sha256'],
                         'raw_label_available_at': result['label_available_at'],
                         'effective_known_at': known_at.isoformat()})
        if registered['symbol'] in symbols:
            row = {key: result[key] for key in ('sample_id', 'symbol',
                   'decision_at', 'label_end_at', 'y')}
            row['label_available_at'] = known_at.isoformat()
            rows.append(row)
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate selected prior sample IDs')
    rows.sort(key=lambda r: (r['symbol'], r['decision_at'], r['sample_id']))
    body = {'version': 'all_candidate_prior_view_v1', 'mode': store.mode,
            'store_cut': store_cut, 'information_deadline_at': deadline.isoformat(),
            'captured_at': captured_at.isoformat(),
            'observed_through_at': observed_through.isoformat(),
            'symbols': symbols, 'rows': rows, 'selected_versions': selected,
            'coverage': coverage,
            'max_known_at': max((x['effective_known_at'] for x in selected), default=None)}
    data = canonical(body) + '\n'
    content_sha = hashlib.sha256(data.encode()).hexdigest()
    path = store.path.parent / 'prior_views' / f'{content_sha}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if file_sha(path) != content_sha:
            raise ValueError('existing PriorView bytes changed')
    else:
        with path.open('x') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    with store.transaction():
        existing = next((row for row in store.db.execute("SELECT * FROM events WHERE event_type='view_created'")
                         if json.loads(row['payload_json'])['path'] == str(path)), None)
        if existing:
            if json.loads(existing['payload_json'])['sha256'] != content_sha:
                raise ValueError('PriorView seal changed')
        else:
            store.append('view_created', '', '', '',
                         {'path': str(path), 'sha256': content_sha,
                          'store_cut': store_cut, 'body_sha256': digest(body)})
    return dict(body, path=str(path), view_sha256=content_sha)


def verify_prior_view(path: Path, store: PriorStore) -> dict:
    path = Path(path).resolve()
    events = store.events(event_type='view_created')
    seals = [e for e in events if Path(e['payload']['path']).resolve() == path]
    if len(seals) != 1 or file_sha(path) != seals[0]['payload']['sha256']:
        raise ValueError('PriorView absent, duplicated, or bytes changed')
    body = json.loads(path.read_text())
    if digest(body) != seals[0]['payload']['body_sha256']:
        raise ValueError('PriorView content differs from sealed hash')
    return dict(body, path=str(path), view_sha256=seals[0]['payload']['sha256'])
