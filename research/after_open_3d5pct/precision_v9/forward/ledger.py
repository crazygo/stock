"""Local append-only v9 research publication ledger. No network or order API."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4
from zoneinfo import ZoneInfo

from ...contracts import Session
from ...timeaxis import add_regular_minutes

UTC = timezone.utc
ET = ZoneInfo('America/New_York')
TARGETS = ('1d_3pct', '1d_5pct', '1d_8pct', '3d_3pct', '3d_5pct', '3d_8pct',
           '5d_3pct', '5d_5pct', '5d_8pct')
REQUIRED_ARTIFACTS = ('model_sha256', 'schema_sha256', 'calibration_sha256',
                      'source_code_sha256')


def _get(value, key, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _mapping(value):
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    raise TypeError('expected frozen dictionary or dataclass')


def _reject_registration_wrapper(cohort) -> None:
    if (isinstance(cohort, dict) and 'registration' in cohort) or \
            (not isinstance(cohort, dict) and hasattr(cohort, 'registration')):
        raise ValueError('cohort registration wrapper is forbidden; freeze the full CohortSpec')


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def timestamp(value) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('timestamp lacks timezone')
    return result.astimezone(UTC)


def calendar_sessions(spec) -> list[Session]:
    raw = _get(spec, 'calendar_sessions', _get(spec, 'sessions'))
    if not raw:
        raise ValueError('missing frozen official calendar')
    out = [Session(timestamp(_get(s, 'open_at')), timestamp(_get(s, 'close_at')))
           for s in raw]
    if any(a.close_at >= b.open_at for a, b in zip(out, out[1:])):
        raise ValueError('calendar is not sorted and nonoverlapping')
    return out


def _at(session_date: str, hour: int, minute: int, second=0) -> datetime:
    return datetime.combine(datetime.fromisoformat(session_date).date(),
                            time(hour, minute, second), ET).astimezone(UTC)


def _prediction_hash(prediction: dict) -> str:
    without = dict(prediction)
    quality = dict(without.get('quality', {}))
    quality.pop('prediction_sha256', None)
    without['quality'] = quality
    return digest(without)


def _artifact_valid(quality: dict, route, snapshot_sha=None) -> bool:
    artifacts = quality.get('artifacts') or {}
    if not all(artifacts.get(name) for name in REQUIRED_ARTIFACTS):
        return False
    frozen = _get(route, 'artifacts', {}) or {}
    if any(frozen.get(name) and frozen[name] != artifacts[name] for name in REQUIRED_ARTIFACTS):
        return False
    files = artifacts.get('files') or {}
    if not {'model', 'source', 'schema', 'calibration', 'snapshot'} <= set(files):
        return False
    try:
        for role, aggregate_key in (('model', 'model_sha256'),
                                    ('source', 'source_code_sha256')):
            refs = files[role]
            if not isinstance(refs, list) or not refs:
                return False
            hashes = []
            for ref in refs:
                expected = ref['sha256']
                if not expected or file_sha(Path(ref['path'])) != expected:
                    return False
                hashes.append(expected)
            if digest(hashes) != artifacts[aggregate_key]:
                return False
        for role, key in (('schema', 'schema_sha256'),
                          ('calibration', 'calibration_sha256')):
            ref = files[role]
            if file_sha(Path(ref['path'])) != ref['sha256'] or ref['sha256'] != artifacts[key]:
                return False
        ref = files['snapshot']
        if file_sha(Path(ref['path'])) != ref['sha256'] or \
                (snapshot_sha and ref['sha256'] != snapshot_sha):
            return False
    except (KeyError, OSError, TypeError):
        return False
    return True


def route_contract(route) -> dict:
    artifacts = _get(route, 'artifacts', {}) or {}
    return {'model_version': _get(route, 'model_version'),
            'score_column': _get(route, 'score_column'),
            'thresholds': _get(route, 'thresholds'),
            'artifacts': {name: artifacts.get(name) for name in REQUIRED_ARTIFACTS}}


class Ledger:
    """Fixture clocks cannot open a live DB; live time comes only from the OS."""

    def __init__(self, path: Path, *, mode='fixture', clock=None):
        if mode not in ('fixture', 'live'):
            raise ValueError('invalid ledger mode')
        if mode == 'live' and clock is not None:
            raise ValueError('live ledger cannot use an injected clock')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode, self.clock = mode, clock
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                route_id TEXT NOT NULL, model_version TEXT NOT NULL,
                cohort_id TEXT NOT NULL, run_id TEXT NOT NULL,
                sample_id TEXT NOT NULL, symbol TEXT NOT NULL,
                session_date TEXT NOT NULL, cutoff_at TEXT NOT NULL,
                decision_at TEXT NOT NULL, recorded_at TEXT NOT NULL,
                event_type TEXT NOT NULL, payload_json TEXT NOT NULL,
                payload_sha TEXT NOT NULL, prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS one_decision_event
                ON events(route_id,cohort_id,cutoff_at,symbol,event_type)
                WHERE event_type IN ('prediction','buy','refusal');
            CREATE UNIQUE INDEX IF NOT EXISTS one_session_reconciliation
                ON events(route_id,cohort_id,session_date,event_type)
                WHERE event_type='session_status';
            CREATE INDEX IF NOT EXISTS route_symbol_events
                ON events(route_id,symbol,event_type,seq);
            CREATE TRIGGER IF NOT EXISTS no_event_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT,'append-only events'); END;
            CREATE TRIGGER IF NOT EXISTS no_event_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT,'append-only events'); END;
        ''')
        row = self.db.execute("SELECT value FROM metadata WHERE key='mode'").fetchone()
        if row and row['value'] != mode:
            self.db.close()
            raise ValueError('fixture/live ledger mode mismatch')
        if not row:
            self.db.execute("INSERT INTO metadata(key,value) VALUES('mode',?)", (mode,))

    def close(self):
        self.db.close()

    def now(self) -> datetime:
        return timestamp(self.clock() if self.clock else datetime.now(UTC))

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def append(self, fields: dict, payload: dict) -> dict:
        """Caller holds BEGIN IMMEDIATE; hash covers every stored field."""
        prior = self.db.execute('SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1').fetchone()
        prev = prior['event_hash'] if prior else '0' * 64
        row = dict(fields, event_id=str(uuid4()), recorded_at=self.now().isoformat(),
                   payload_sha=digest(payload), prev_hash=prev)
        row['event_hash'] = digest(row)
        row['payload_json'] = canonical(payload)
        columns = tuple(row)
        self.db.execute(f"INSERT INTO events({','.join(columns)}) VALUES({','.join('?' for _ in columns)})",
                        tuple(row.values()))
        row['payload'] = payload
        return row

    def verify_chain(self, *, cut=None) -> int:
        rows = self.db.execute('SELECT * FROM events WHERE seq<=? ORDER BY seq',
                               (cut if cut is not None else 2**63 - 1,)).fetchall()
        prev = '0' * 64
        for expected_seq, row in enumerate(rows, 1):
            raw = dict(row)
            payload = json.loads(raw.pop('payload_json'))
            event_hash = raw.pop('event_hash')
            if raw.pop('seq') != expected_seq or raw['prev_hash'] != prev or \
                    digest(payload) != raw['payload_sha'] or digest(raw) != event_hash:
                raise ValueError(f'ledger hash chain damaged at {expected_seq}')
            prev = event_hash
        return len(rows)

    def events(self, *, cut=None, event_type=None) -> list[dict]:
        self.verify_chain(cut=cut)
        query = 'SELECT * FROM events WHERE seq<=?'
        params = [cut if cut is not None else 2**63 - 1]
        if event_type:
            query += ' AND event_type=?'
            params.append(event_type)
        query += ' ORDER BY seq'
        out = []
        for row in self.db.execute(query, params):
            item = dict(row)
            item['payload'] = json.loads(item.pop('payload_json'))
            out.append(item)
        return out


def register_cohort(cohort, ledger: Ledger) -> dict:
    """Freeze full official-date denominator before the first cohort session."""
    _reject_registration_wrapper(cohort)
    cohort_id = _get(cohort, 'cohort_id')
    dates = list(_get(cohort, 'session_dates', []))
    if not cohort_id or len(dates) < 60 or dates != sorted(set(dates)):
        raise ValueError('cohort requires >=60 ordered unique official sessions')
    routes = list(_get(cohort, 'routes', []))
    contracts = _get(cohort, 'route_contracts', {}) or {}
    if len(routes) != 5 or len(set(routes)) != 5 or set(contracts) != set(routes):
        raise ValueError('cohort requires five frozen route contracts')
    for route_id, contract in contracts.items():
        if not contract.get('model_version') or not contract.get('score_column') or \
                set(contract.get('thresholds') or {}) != set(_get(cohort, 'groups', [])) or \
                any(not contract.get('artifacts', {}).get(name) for name in REQUIRED_ARTIFACTS):
            raise ValueError(f'incomplete frozen route contract: {route_id}')
    frozen_calendar = _get(cohort, 'calendar_sessions')
    if frozen_calendar:
        sessions = calendar_sessions({'calendar_sessions': frozen_calendar})
        official = [s.open_at.astimezone(ET).date().isoformat() for s in sessions]
        try:
            first = official.index(dates[0])
        except ValueError as exc:
            raise ValueError('cohort starts outside frozen calendar') from exc
        if official[first:first + len(dates)] != dates or \
                timestamp(_get(cohort, 'last_session_close_at')) != sessions[first + len(dates) - 1].close_at:
            raise ValueError('cohort skips official sessions or changes final close')
    elif ledger.mode == 'live':
        raise ValueError('live cohort lacks frozen official calendar')
    if ledger.mode == 'live' and (not _get(cohort, 'calendar_verified') or
                                  not _get(cohort, 'calendar_sha256') or
                                  not _get(cohort, 'spec_sha256') or
                                  _get(cohort, 'evidence_mode') != 'live'):
        raise ValueError('live cohort lacks verified frozen calendar/spec')
    frozen = _mapping(cohort)
    if ledger.now() >= _at(dates[0], 9, 30):
        raise ValueError('cohort registration is not before first session')
    ledger.verify_chain()
    with ledger.transaction():
        old = ledger.db.execute("SELECT * FROM events WHERE event_type='cohort_registered' AND cohort_id=?",
                                (cohort_id,)).fetchone()
        if old:
            if json.loads(old['payload_json'])['registration_sha256'] != digest(frozen):
                raise ValueError('cohort registration changed')
            return dict(old)
        fields = {'route_id': '_system', 'model_version': '', 'cohort_id': cohort_id,
                  'run_id': '', 'sample_id': '', 'symbol': '', 'session_date': '',
                  'cutoff_at': '', 'decision_at': '', 'event_type': 'cohort_registered'}
        return ledger.append(fields, {'registration_sha256': digest(frozen),
                                      'session_dates': dates, 'registration': frozen})


def _eligible(pred: dict, spec, route, cohort, now: datetime,
              ledger_mode: str) -> tuple[str | None, dict]:
    lineage = pred.get('lineage') or {}
    q = pred.get('quality') or {}
    symbol, date = pred['symbol'], pred['session_date']
    decision = _at(date, 11, 30)
    deadline = _at(date, 11, 30, 30)
    entry = _at(date, 11, 35)
    evidence = {'published_at': now.isoformat(), 'planned_entry_at': entry.isoformat(),
                'decision_at': decision.isoformat(), 'deadline_at': deadline.isoformat()}
    if any(key in pred for key in ('y', 'label', 'labels', 'outcome', 'outcomes',
                                   'entry_price', 'first_touch')):
        return 'data_unverified', evidence
    if now < decision:
        return 'data_unverified', dict(evidence, reason_detail='published_before_decision')
    if now > deadline or now >= entry:
        return 'expired', evidence
    if pred.get('cutoff_at') != decision.isoformat() or pred.get('decision_at') != decision.isoformat():
        return 'data_unverified', evidence
    if pred.get('information_deadline_at', deadline.isoformat()) != deadline.isoformat():
        return 'data_unverified', evidence
    if pred.get('route_id') != _get(route, 'route_id') or \
            pred.get('model_version') != _get(route, 'model_version'):
        return 'model_unavailable', evidence
    if not pred.get('sample_id') or not lineage.get('source_ids') or \
            not lineage.get('snapshot_sha256'):
        return 'data_unverified', evidence
    if ledger_mode == 'live' and lineage.get('mode') != 'live':
        return 'data_unverified', dict(evidence, reason_detail='nonlive_lineage')
    if q.get('prediction_sha256') != _prediction_hash(pred):
        return 'data_unverified', evidence
    if not all(q.get(k) is True for k in ('g1_ready', 'g2_eligible', 'publication_eligible')) \
            or q.get('rejections') or lineage.get('receipt_verified') is not True:
        return 'data_unverified', evidence
    try:
        latest_available = timestamp(lineage['latest_available_at'])
        latest_received = timestamp(lineage['latest_received_at'])
        if latest_available > deadline or latest_received > deadline or \
                latest_available > now or latest_received > now:
            return 'data_unverified', dict(evidence, reason_detail='input_not_observed_at_publication')
    except (ValueError, KeyError, TypeError):
        return 'data_unverified', evidence
    if not _artifact_valid(q, route, lineage.get('snapshot_sha256')):
        return 'data_unverified', evidence
    raw, processed = pred.get('raw'), pred.get('processed')
    if not isinstance(raw, dict) or not isinstance(processed, dict) or \
            set(raw) != set(TARGETS) or set(processed) != set(TARGETS):
        return 'model_unavailable', evidence
    for scores in (raw, processed):
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
               for v in scores.values()):
            return 'model_unavailable', evidence
    if pred.get('score_column') != _get(route, 'score_column', 'raw_3d_5pct'):
        return 'model_unavailable', evidence
    return None, evidence


def issue_once(spec, route, cohort, run_id, predictions, quality, ledger: Ledger) -> list[dict]:
    """Append all decisions atomically. A returned buy has committed to SQLite."""
    _reject_registration_wrapper(cohort)
    if not run_id or not isinstance(predictions, list):
        raise ValueError('run_id and prediction list required')
    cohort_id = _get(cohort, 'cohort_id')
    route_id = _get(route, 'route_id')
    version = _get(route, 'model_version')
    if not all((cohort_id, route_id, version)):
        raise ValueError('frozen cohort/route identity required')
    if route_id not in _get(cohort, 'routes', []):
        raise ValueError('route outside frozen cohort')
    if route_contract(route) != (_get(cohort, 'route_contracts', {}) or {}).get(route_id):
        raise ValueError('route differs from frozen cohort contract')
    dates = list(_get(cohort, 'session_dates', []))
    groups = _get(spec, 'groups') or {}
    thresholds = _get(route, 'thresholds') or {}
    sessions = calendar_sessions(spec)
    cohort_calendar = _get(cohort, 'calendar_sessions')
    if cohort_calendar and digest(cohort_calendar) != digest(
            _get(spec, 'calendar_sessions', _get(spec, 'sessions'))):
        raise ValueError('cohort calendar and issue calendar differ')
    if ledger.mode == 'live' and (_get(cohort, 'spec_sha256') != digest(_mapping(spec)) or
                                  _get(cohort, 'calendar_sha256') != digest(
                                      _get(spec, 'calendar_sessions', _get(spec, 'sessions')))):
        raise ValueError('live calendar/spec changed after cohort freeze')
    if not dates or not groups or set(thresholds) != set(groups):
        raise ValueError('missing frozen dates/groups/thresholds')
    if any(value is not None and (not isinstance(value, (int, float)) or
                                  not math.isfinite(value) or not 0 <= value <= 1)
           for value in thresholds.values()):
        raise ValueError('invalid frozen threshold')
    ledger.verify_chain()
    output = []
    if len({p['session_date'] for p in predictions}) > 1:
        raise ValueError('issue_once accepts one official session per transaction')
    with ledger.transaction():
        reg = ledger.db.execute("SELECT payload_json FROM events WHERE cohort_id=? AND event_type='cohort_registered'",
                                (cohort_id,)).fetchone()
        if not reg or json.loads(reg['payload_json'])['registration_sha256'] != digest(
                _mapping(cohort)):
            raise ValueError('cohort is not identically preregistered')
        if not predictions:
            date = _get(quality, 'session_date')
            if date not in dates:
                raise ValueError('empty run requires a frozen official session_date')
            old = ledger.db.execute("SELECT * FROM events WHERE event_type='run_empty' AND cohort_id=? AND route_id=? AND run_id=?",
                                    (cohort_id, route_id, run_id)).fetchone()
            if old:
                return [dict(old)]
            return [ledger.append({'route_id': route_id, 'model_version': version,
                'cohort_id': cohort_id, 'run_id': run_id, 'sample_id': '',
                'symbol': '', 'session_date': date,
                'cutoff_at': _at(date, 11, 30).isoformat(),
                'decision_at': _at(date, 11, 30).isoformat(),
                'event_type': 'run_empty'},
                {'reason': 'no_candidates', 'run_quality_sha256': digest(_mapping(quality))})]
        for pred in sorted(predictions, key=lambda p: (p['decision_at'], p['symbol'])):
            symbol, date = pred['symbol'], pred['session_date']
            if date not in dates or not any(s.open_at.astimezone(ET).date().isoformat() == date for s in sessions):
                raise ValueError('prediction outside frozen official cohort/calendar')
            cutoff = _at(date, 11, 30).isoformat()
            old = ledger.db.execute('''SELECT * FROM events WHERE route_id=? AND cohort_id=?
                AND cutoff_at=? AND symbol=? AND event_type IN ('prediction','refusal') ORDER BY seq LIMIT 1''',
                (route_id, cohort_id, cutoff, symbol)).fetchone()
            pred_hash = _prediction_hash(pred)
            if old:
                saved = json.loads(old['payload_json'])
                if saved['prediction_sha256'] != pred_hash:
                    raise ValueError('immutable prediction differs on retry')
                receipt = ledger.db.execute('''SELECT * FROM events WHERE route_id=? AND cohort_id=?
                    AND cutoff_at=? AND symbol=? AND event_type IN ('buy','refusal') ORDER BY seq LIMIT 1''',
                    (route_id, cohort_id, cutoff, symbol)).fetchone()
                output.append(dict(receipt))
                continue
            now = ledger.now()
            reason, evidence = _eligible(pred, spec, route, cohort, now, ledger.mode)
            if _get(quality, 'g2_eligible') is not True and reason is None:
                reason = 'data_unverified'
            score_column = pred.get('score_column', 'raw_3d_5pct')
            score_family, target = score_column.split('_', 1) if '_' in score_column else ('', '')
            scores = pred.get('raw' if score_family == 'raw' else 'processed', {})
            score = scores.get(target)
            members = [g for g, names in groups.items() if symbol in names]
            triggers = [g for g in members if thresholds[g] is not None and
                        isinstance(score, (int, float)) and score >= thresholds[g]]
            if reason is None and not triggers:
                reason = 'no_trigger'
            if reason is None:
                prior = ledger.db.execute("SELECT payload_json FROM events WHERE route_id=? AND symbol=? AND event_type='buy' ORDER BY seq DESC LIMIT 1",
                                          (route_id, symbol)).fetchone()
                if prior and timestamp(json.loads(prior['payload_json'])['cooldown_end']) > timestamp(pred['decision_at']):
                    reason = 'cooldown'
            if reason is None:
                try:
                    cooldown_end = add_regular_minutes(timestamp(evidence['planned_entry_at']), 1170, sessions)
                    evidence['cooldown_end'] = cooldown_end.isoformat()
                except ValueError:
                    reason = 'calendar_unavailable'
            fields = {'route_id': route_id, 'model_version': version, 'cohort_id': cohort_id,
                      'run_id': run_id, 'sample_id': pred.get('sample_id', ''),
                      'symbol': symbol, 'session_date': date, 'cutoff_at': cutoff,
                      'decision_at': pred.get('decision_at', ''), 'event_type': 'prediction'}
            common = {'prediction_sha256': pred_hash, 'prediction': pred,
                      'quality': pred.get('quality', {}),
                      'lineage': pred.get('lineage', {}), 'raw': pred.get('raw'),
                      'processed': pred.get('processed'), 'score_column': score_column}
            if reason in ('data_unverified', 'model_unavailable', 'expired'):
                fields['event_type'] = 'refusal'
                receipt = ledger.append(fields, {'prediction_sha256': pred_hash,
                                                 'reason': reason, **evidence})
            else:
                ledger.append(fields, common)
                fields['event_type'] = 'refusal' if reason else 'buy'
                receipt = ledger.append(fields, {'prediction_sha256': pred_hash,
                    'reason': reason, 'score': score, 'thresholds': thresholds,
                    'member_groups': members, 'trigger_groups': triggers if not reason else [],
                    **evidence})
            output.append(receipt)
        if ledger.now() > _at(predictions[0]['session_date'], 11, 30, 30) and \
                any(row['event_type'] == 'buy' for row in output):
            raise ValueError('publication deadline elapsed before transaction commit')
    return output


def reconcile_session(spec, cohort, route_id: str, session_date: str, ledger: Ledger) -> dict:
    """Append actual post-close discovery; no run is invented for a stopped day."""
    if session_date not in _get(cohort, 'session_dates', []):
        raise ValueError('session outside frozen cohort')
    if route_id not in _get(cohort, 'routes', []):
        raise ValueError('unknown frozen route')
    session = next((s for s in calendar_sessions(spec)
                    if s.open_at.astimezone(ET).date().isoformat() == session_date), None)
    if session is None or ledger.now() < session.close_at:
        raise ValueError('official session not closed or absent from calendar')
    ledger.verify_chain()
    cohort_id = _get(cohort, 'cohort_id')
    with ledger.transaction():
        old = ledger.db.execute("SELECT * FROM events WHERE event_type='session_status' AND cohort_id=? AND route_id=? AND session_date=?",
                                (cohort_id, route_id, session_date)).fetchone()
        if old:
            return dict(old)
        attempted = ledger.db.execute('''SELECT COUNT(*) FROM events WHERE cohort_id=? AND route_id=?
            AND session_date=? AND event_type IN ('prediction','refusal','run_empty')''',
            (cohort_id, route_id, session_date)).fetchone()[0]
        buys = ledger.db.execute('''SELECT COUNT(*) FROM events WHERE cohort_id=? AND route_id=?
            AND session_date=? AND event_type='buy' ''',
            (cohort_id, route_id, session_date)).fetchone()[0]
        status = 'ran_with_buy' if buys else 'ran_no_buy' if attempted else 'missing_run'
        return ledger.append({'route_id': route_id, 'model_version': '',
                              'cohort_id': cohort_id, 'run_id': '', 'sample_id': '',
                              'symbol': '', 'session_date': session_date,
                              'cutoff_at': '', 'decision_at': '',
                              'event_type': 'session_status'},
                             {'status': status, 'attempted_candidates': attempted,
                              'issued_buys': buys, 'discovered_at': ledger.now().isoformat()})


def record_protocol_violation(signal_id: str, reason: str, ledger: Ledger) -> dict:
    """Flag an already published BUY without changing its prediction or denominator."""
    if not reason:
        raise ValueError('violation reason required')
    ledger.verify_chain()
    with ledger.transaction():
        buy = ledger.db.execute("SELECT * FROM events WHERE event_id=? AND event_type='buy'",
                                (signal_id,)).fetchone()
        if buy is None:
            raise ValueError('protocol violation must refer to an issued BUY')
        old = next((row for row in ledger.db.execute(
            "SELECT * FROM events WHERE event_type='protocol_violation' AND sample_id=?",
            (signal_id,)) if json.loads(row['payload_json'])['reason'] == reason), None)
        if old:
            return dict(old)
        fields = {key: buy[key] for key in ('route_id', 'model_version', 'cohort_id',
            'run_id', 'symbol', 'session_date', 'cutoff_at', 'decision_at')}
        fields['sample_id'] = signal_id
        fields['event_type'] = 'protocol_violation'
        return ledger.append(fields, {'signal_id': signal_id, 'reason': reason,
                                      'discovered_at': ledger.now().isoformat()})
