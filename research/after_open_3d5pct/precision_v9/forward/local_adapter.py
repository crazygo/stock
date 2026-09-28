"""Offline, fail-closed file adapter for v9 forward engineering fixtures.

This module has no downloader, account access, scheduler, or live publisher.
Local byte verification does not establish an historical receipt.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter

import pandas as pd

from .features import BAR_COLUMNS, LABEL_COLUMNS, build_features_asof
from .inference import load_route, predict_route
from ...timeaxis import add_regular_minutes
from .ledger import (_at, _prediction_hash, calendar_sessions, canonical,
                     file_sha, issue_once, record_protocol_violation, timestamp)
from .prior_store import _future_eligibility_key, verify_prior_view


def _read_ref(ref, *, kind='json'):
    if not isinstance(ref, dict) or not {'path', 'sha256'} <= set(ref):
        raise ValueError('local source path and SHA required')
    path = Path(ref['path']).expanduser().resolve(strict=True)
    if file_sha(path) != ref['sha256']:
        raise ValueError(f'local source bytes changed: {path}')
    if kind == 'parquet':
        value = pd.read_parquet(path)
    elif kind == 'json':
        value = json.loads(path.read_text())
    else:
        raise ValueError('unsupported local source format')
    if file_sha(path) != ref['sha256']:
        raise ValueError(f'local source changed during read: {path}')
    return value, {'path': str(path), 'sha256': ref['sha256'],
                   'received_at': ref.get('received_at')}


def _members(document):
    if not isinstance(document, dict) or not isinstance(document.get('members'), list):
        raise ValueError('prior universe member list missing')
    members = [m['symbol'] if isinstance(m, dict) else m for m in document['members']
               if not isinstance(m, dict) or m.get('role') == 'candidate']
    if not members or len(set(members)) != len(members) or 'QQQ' in members:
        raise ValueError('prior universe candidates invalid')
    return set(members)


def _bar_file(ref, symbol):
    data, proof = _read_ref(ref, kind=ref.get('format', 'parquet'))
    if isinstance(data, list):
        data = pd.DataFrame(data)
    if not isinstance(data, pd.DataFrame) or not BAR_COLUMNS <= set(data):
        raise ValueError(f'{symbol}: source bar content/schema missing')
    if LABEL_COLUMNS & set(data) or any(str(c).startswith('y_') for c in data):
        raise ValueError(f'{symbol}: source contains future labels or entry')
    if 'symbol' in data and not data['symbol'].eq(symbol).all():
        raise ValueError(f'{symbol}: source contains another symbol')
    return data, proof


def _actions_file(ref, symbol):
    content, proof = _read_ref(ref)
    required = {'coverage_start', 'coverage_end', 'split_dates', 'halt_dates',
                'coverage_verified'}
    if not isinstance(content, dict) or not required <= set(content) or \
            not isinstance(content['split_dates'], list) or \
            not isinstance(content['halt_dates'], list):
        raise ValueError(f'{symbol}: corporate-action coverage content unknown')
    return content, proof


def _write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return file_sha(path)


def _checked_input_hashes(bundle, route_proof, loaded, prior_store, *, strict=True):
    """Rehash every successfully parsed reference, including model dependencies."""
    evidence = bundle['evidence']
    refs = {name: evidence[name] for name in
            ('manifest', 'prior_universe', 'forecast_candidates',
             'required_market_dependencies', 'calendar')}
    refs.update(evidence['sources'])
    refs.update({f'groups:{key}': value for key, value in evidence['groups'].items()})
    if route_proof:
        refs['route_manifest'] = route_proof
    if loaded:
        refs.update({f"artifact:{item['role']}:{index}": item
                     for index, item in enumerate(loaded['artifacts'])})
    after, mismatches = {}, []
    for name, ref in refs.items():
        try:
            actual = file_sha(ref['path'])
        except OSError:
            actual = None
        after[name] = {'path': ref['path'], 'sha256': actual}
        if actual != ref['sha256']:
            mismatches.append(f'{name}: source bytes changed after read')
    try:
        prior = verify_prior_view(bundle['prior_view']['path'], prior_store)
        prior_hash = prior['view_sha256']
        prior_error = None
    except (ValueError, OSError) as exc:
        prior_hash = (file_sha(bundle['prior_view']['path'])
                      if Path(bundle['prior_view']['path']).exists() else None)
        prior_error = f'{type(exc).__name__}: {exc}'
    after['prior_view'] = {'path': bundle['prior_view']['path'], 'sha256': prior_hash}
    if prior_error is not None:
        after['prior_view']['seal_error'] = prior_error
        mismatches.append('PriorView seal or hash chain verification failed')
    if prior_hash != bundle['prior_view']['view_sha256']:
        mismatches.append('PriorView changed after read')
    if strict and mismatches:
        raise ValueError('; '.join(mismatches))
    return after


def load_local_bundle(input_manifest_ref, prior_view_path, mode, prior_store):
    """Verify three distinct frozen lists and every referenced local file.

    Returns all forecast candidates, including individually rejected ones.
    `live` means an input format only; verified bytes alone never set G2 true.
    """
    if mode not in ('fixture', 'historical_fixture', 'live'):
        raise ValueError('unknown local evidence mode')
    manifest, manifest_proof = _read_ref(input_manifest_ref)
    if not isinstance(manifest, dict) or manifest.get('mode') != mode:
        raise ValueError('input manifest mode differs')
    required = {'session_date', 'prior_universe', 'forecast_candidates',
                'required_market_dependencies', 'calendar', 'bars',
                'corporate_actions'}
    if not required <= set(manifest):
        raise ValueError(f'input manifest missing {sorted(required-set(manifest))}')
    day = manifest['session_date']
    prior_view = verify_prior_view(prior_view_path, prior_store)
    if (mode == 'live' and prior_view['mode'] != 'live') or \
            (mode == 'historical_fixture' and prior_view['mode'] != 'historical_fixture') or \
            (mode == 'fixture' and prior_view['mode'] != 'fixture'):
        raise ValueError('prior view crosses fixture/live namespace')
    deadline = _at(day, 11, 30, 30)
    if timestamp(prior_view['observed_through_at']) > deadline or \
            timestamp(prior_view['information_deadline_at']) > deadline:
        raise ValueError('prior view extends beyond frozen information deadline')
    if mode == 'live' and timestamp(prior_view['captured_at']) > datetime.now(timezone.utc):
        raise ValueError('live prior view capture is in the future')
    universe, universe_proof = _read_ref(manifest['prior_universe'])
    full_members = _members(universe)
    forecast, forecast_proof = _read_ref(manifest['forecast_candidates'])
    if not isinstance(forecast, dict) or not isinstance(forecast.get('rows'), list):
        raise ValueError('forecast candidate rows missing')
    rows = forecast['rows']
    if len({r.get('symbol') for r in rows}) != len(rows) or \
            not {r.get('symbol') for r in rows} <= full_members or \
            any(type(r.get('eligible')) is not bool or not r.get('sample_id') or
                _future_eligibility_key(r) for r in rows):
        raise ValueError('forecast candidates are duplicated or outside prior universe')
    dependencies, dependency_proof = _read_ref(manifest['required_market_dependencies'])
    route_deps = dependencies.get('routes') if isinstance(dependencies, dict) else None
    if not isinstance(route_deps, dict):
        raise ValueError('route market dependency matrix missing')
    for route_id, by_symbol in route_deps.items():
        if set(by_symbol) != {r['symbol'] for r in rows}:
            raise ValueError(f'{route_id}: dependency matrix omits forecast candidates')
        for symbol, deps in by_symbol.items():
            if not isinstance(deps, list) or symbol not in deps or len(set(deps)) != len(deps):
                raise ValueError(f'{route_id}:{symbol}: required dependencies invalid')
            if route_id in ('B_group', 'C_group', 'C_no_daily') and 'QQQ' not in deps:
                raise ValueError(f'{route_id}:{symbol}: grouped dependency lacks QQQ')
    registrations = {e['symbol']: e for e in prior_store.events(
        cut=prior_view['store_cut'], event_type='candidate_registered')
        if e['session_date'] == day}
    sessions_registered = [e for e in prior_store.events(
        cut=prior_view['store_cut'], event_type='session_registered')
        if e['session_date'] == day]
    if len(sessions_registered) != 1 or set(registrations) != full_members or \
            sessions_registered[0]['payload']['universe_sha256'] != universe_proof['sha256']:
        raise ValueError('prior universe is not fully registered at the selected cut')
    for row in rows:
        registration = registrations[row['symbol']]['payload']
        if row['sample_id'] != registration['sample_id'] or \
                row['eligible'] != registration['eligible']:
            raise ValueError('forecast candidate differs from frozen all-candidate row')
    calendar_doc, calendar_proof = _read_ref(manifest['calendar'])
    sessions = calendar_doc.get('sessions') if isinstance(calendar_doc, dict) else calendar_doc
    if not isinstance(sessions, list) or not sessions:
        raise ValueError('official calendar content missing')
    parsed = calendar_sessions({'calendar_sessions': sessions})
    if len(parsed) != len(sessions) or any(not {'session_date', 'duration_minutes'} <= set(s)
                                                for s in sessions):
        raise ValueError('official calendar feature fields missing')
    if day not in {s['session_date'] for s in sessions}:
        raise ValueError('forecast day outside official calendar')
    bars, actions, source_failures, proofs = {}, {}, {}, {}
    for symbol, ref in manifest['bars'].items():
        try:
            bars[symbol], proofs[f'bars:{symbol}'] = _bar_file(ref, symbol)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            source_failures[f'bars:{symbol}'] = str(exc)
    for symbol, ref in manifest['corporate_actions'].items():
        try:
            actions[symbol], proofs[f'actions:{symbol}'] = _actions_file(ref, symbol)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            source_failures[f'actions:{symbol}'] = str(exc)
    groups = manifest.get('groups') or {}
    group_data, group_proofs = {}, {}
    for name in ('versions', 'memberships', 'universe_symbols'):
        if name in groups:
            group_data[name], group_proofs[name] = _read_ref(groups[name])
    if group_data and (not isinstance(group_data.get('versions'), list) or
                       not isinstance(group_data.get('memberships'), list) or
                       not isinstance(group_data.get('universe_symbols'), list)):
        raise ValueError('group version/member content malformed')
    if group_data:
        group_candidates = group_data['universe_symbols']
        if (any(not isinstance(s, str) or not s or s.strip() != s
                for s in group_candidates) or
                len(group_candidates) != len(set(group_candidates)) or
                set(group_candidates) != full_members or 'QQQ' in group_candidates):
            raise ValueError('group universe differs from registered full candidate universe')
        version_ids = [v.get('version_id') for v in group_data['versions']]
        if any(not v for v in version_ids) or len(set(version_ids)) != len(version_ids):
            raise ValueError('group version IDs missing or duplicated')
        for member in group_data['memberships']:
            if (member.get('version_id') not in version_ids or
                    not member.get('symbol') or
                    not isinstance(member.get('group_ids'), list)):
                raise ValueError('group membership is not bound to a frozen version')
    feature_start = manifest.get('feature_calendar_start', sessions[0]['session_date'])
    if feature_start > day:
        raise ValueError('feature calendar begins after forecast session')
    snapshot = {'bars': bars,
                'sessions': [session for session in sessions
                             if feature_start <= session['session_date'] <= day],
                'mode': 'historical_fixture' if mode == 'fixture' else mode,
                'snapshot_file': manifest_proof['path'],
                'snapshot_sha256': manifest_proof['sha256'],
                'snapshot_hash_verified': True}
    metadata = {'split_days': {s: a['split_dates'] for s, a in actions.items()},
                'versions': group_data.get('versions', []),
                'memberships': group_data.get('memberships', []),
                'universe_symbols': group_data.get('universe_symbols', []),
                'membership_snapshot_sha256': group_proofs.get('memberships', {}).get('sha256'),
                'group_peer_semantics': manifest.get('group_peer_semantics'),
                # Byte checks are not contemporaneous receipt attestations.
                'group_versions_complete': False,
                'corporate_actions_verified': False,
                'mature_prior_scope': 'unverified',
                'mature_prior_snapshot_sha256': prior_view['view_sha256']}
    evidence = {'mode': mode, 'manifest': manifest_proof,
                'prior_view': {'path': prior_view['path'], 'sha256': prior_view['view_sha256'],
                               'captured_at': prior_view['captured_at'],
                               'observed_through_at': prior_view['observed_through_at'],
                               'store_cut': prior_view['store_cut']},
                'prior_universe': universe_proof,
                'forecast_candidates': forecast_proof,
                'required_market_dependencies': dependency_proof,
                'calendar': calendar_proof,
                'sources': proofs, 'groups': group_proofs,
                'local_bytes_verified': not source_failures,
                'real_receipt_verified': False,
                'source_failures': source_failures}
    return {'snapshot': snapshot, 'metadata': metadata,
            'forecast_rows': rows, 'prior_view': prior_view,
            'dependencies': route_deps, 'evidence': evidence}


def run_local_e2e(plan, new_output, fixture_ledger, prior_store):
    """Run one offline research bundle and persist every forecast-candidate decision."""
    if fixture_ledger.mode != 'fixture':
        raise ValueError('local adapter cannot publish to a live ledger')
    destination = Path(new_output)
    destination.mkdir(parents=True, exist_ok=False)
    plan_sha = _write_new(destination / 'plan.json', plan)
    started = perf_counter()
    audit = []
    receipts = []
    bundle = None
    route_proof = None
    loaded = None
    before_hashes = {}
    try:
        mode = plan['mode']
        bundle = load_local_bundle(plan['input_manifest'], plan['prior_view_path'],
                                   mode, prior_store)
        load_seconds = perf_counter() - started
        session_day = bundle['snapshot']['sessions'][-1]['session_date']
        if session_day != plan['session_date']:
            raise ValueError('plan day differs from latest feature calendar session')
        route_doc, route_proof, load_error = None, None, None
        route_id = plan['route']['route_id']
        try:
            route_doc, route_proof = _read_ref(plan['route_manifest'])
            if route_doc['route_id'] != route_id:
                raise ValueError('route manifest differs from frozen plan route')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            load_error = str(exc)
        if route_id not in bundle['dependencies']:
            raise ValueError('route omitted from required dependency matrix')
        model_started = perf_counter()
        loaded = None
        if load_error is None:
            try:
                loaded = load_route(route_doc)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                load_error = str(exc)
        model_seconds = perf_counter() - model_started
        predictions = []
        for row in bundle['forecast_rows']:
            symbol = row['symbol']
            record = {'symbol': symbol, 'sample_id': row['sample_id'],
                      'session_date': plan['session_date'], 'route_id': route_id,
                      'status': None, 'rejections': [], 'stage_seconds': {}}
            audit.append(record)
            if not row['eligible']:
                record.update(status='rejected', rejections=['frozen_ineligible'])
                continue
            if load_error is not None:
                record.update(status='rejected', rejections=['model_unavailable'],
                              error=load_error)
                continue
            required = bundle['dependencies'][route_id][symbol]
            absent = [s for s in required if s not in bundle['snapshot']['bars']]
            missing_actions = [s for s in required if s not in bundle['metadata']['split_days']]
            if absent or missing_actions:
                record.update(status='rejected', rejections=[
                    'required_market_source_missing' if absent else 'corporate_actions_missing'])
                record['missing_symbols'] = sorted(set(absent + missing_actions))
                continue
            if route_id in ('B_group', 'C_group', 'C_no_daily') and (
                    not bundle['metadata']['versions'] or
                    not bundle['metadata']['memberships'] or
                    bundle['metadata']['group_peer_semantics'] !=
                    'v9_causal_history_end_v1'):
                record.update(status='rejected', rejections=['group_metadata_incompatible'])
                continue
            decision = {'symbol': symbol, 'session_date': plan['session_date'],
                        'cutoff_at': _at(plan['session_date'], 11, 30).isoformat(),
                        'decision_at': _at(plan['session_date'], 11, 30).isoformat(),
                        'information_deadline_at': _at(plan['session_date'], 11, 30, 30).isoformat()}
            try:
                began = perf_counter()
                built = build_features_asof(bundle['snapshot'], decision,
                                            route_doc['schema'], bundle['metadata'],
                                            bundle['prior_view']['rows'])
                record['stage_seconds']['features'] = perf_counter() - began
                if built['rejections'] or built['X'] is None:
                    record.update(status='rejected', rejections=built['rejections'])
                    record['feature_lineage'] = built.get('lineage', {})
                    continue
                used_symbols = built['lineage'].get('used_market_symbols')
                if (not isinstance(used_symbols, list) or
                        set(used_symbols) != set(required)):
                    record.update(status='rejected',
                                  rejections=['required_market_dependency_mismatch'],
                                  used_market_symbols=used_symbols,
                                  declared_market_symbols=required)
                    continue
                began = perf_counter()
                prediction = predict_route(loaded, built)
                record['stage_seconds']['inference'] = perf_counter() - began
                # Include non-bar dependencies in the ledger's publication-time
                # lineage. Local bytes alone never upgrade G2.
                source_proofs = [bundle['evidence'][name] for name in
                                 ('manifest', 'prior_universe', 'forecast_candidates',
                                  'required_market_dependencies', 'calendar')]
                source_proofs += [bundle['evidence']['sources'].get(f'{role}:{s}', {})
                                  for s in required for role in ('bars', 'actions')]
                if route_id in ('B_group', 'C_group', 'C_no_daily'):
                    source_proofs += list(bundle['evidence']['groups'].values())
                received = [p.get('received_at') for p in source_proofs]
                known = [timestamp(bundle['prior_view']['captured_at'])]
                known.extend(timestamp(value) for value in received if value)
                lineage = prediction['lineage']
                for key in ('latest_available_at', 'latest_received_at'):
                    if lineage.get(key):
                        known.append(timestamp(lineage[key]))
                    lineage[key] = max(known).isoformat()
                lineage['prior_view_sha256'] = bundle['prior_view']['view_sha256']
                lineage['prior_view_store_cut'] = bundle['prior_view']['store_cut']
                lineage['prior_view_observed_through_at'] = (
                    bundle['prior_view']['observed_through_at'])
                lineage['dependency_receipt_fields_present'] = bool(all(received))
                lineage['latest_dependency_known_at'] = max(known).isoformat()
                prediction['quality']['prediction_sha256'] = _prediction_hash(prediction)
                record['dependency_receipt_fields_present'] = (
                    lineage['dependency_receipt_fields_present'])
                record['latest_dependency_known_at'] = lineage['latest_dependency_known_at']
                record['prediction_sha256'] = prediction['quality']['prediction_sha256']
                record['g1_ready'] = prediction['quality']['g1_ready']
                record['g2_eligible'] = prediction['quality']['g2_eligible']
                predictions.append(prediction)
                record['status'] = 'predicted_unpublished'
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                record.update(status='rejected', rejections=[
                    'feature_or_model_exception'], error_type=type(exc).__name__,
                    error=str(exc))
        _write_new(destination / 'preledger_audit.json', audit)
        hash_started = perf_counter()
        before_hashes = _checked_input_hashes(bundle, route_proof, loaded, prior_store)
        before_hashes['plan'] = {'path': str(destination / 'plan.json'),
                                 'sha256': file_sha(destination / 'plan.json')}
        if before_hashes['plan']['sha256'] != plan_sha:
            raise ValueError('immutable plan bytes changed before ledger publication')
        hash_before_seconds = perf_counter() - hash_started
        ledger_seconds = 0.
        if predictions:
            began = perf_counter()
            receipts = issue_once(plan['spec'], plan['route'], plan['cohort'],
                                  plan['run_id'], predictions,
                                  {'g2_eligible': False}, fixture_ledger)
            ledger_seconds = perf_counter() - began
            by_symbol = {r['symbol']: r for r in receipts}
            for row in audit:
                if row['symbol'] in by_symbol:
                    row['status'] = 'ledger_' + by_symbol[row['symbol']]['event_type']
                    row['ledger_event_id'] = by_symbol[row['symbol']]['event_id']
                    receipt = by_symbol[row['symbol']]
                    payload = receipt.get('payload') or json.loads(receipt['payload_json'])
                    row['ledger_reason'] = payload['reason']
            # Persist the committed ledger result in a second immutable artifact.
            _write_new(destination / 'ledger_audit.json', audit)
        hash_started = perf_counter()
        after_hashes = _checked_input_hashes(bundle, route_proof, loaded, prior_store)
        after_hashes['plan'] = {'path': str(destination / 'plan.json'),
                                'sha256': file_sha(destination / 'plan.json')}
        if before_hashes != after_hashes:
            raise ValueError('input hashes changed across ledger publication')
        hash_after_seconds = perf_counter() - hash_started
        counts = {name: sum(r['status'] == name for r in audit)
                  for name in sorted({r['status'] for r in audit})}
        calendar = calendar_sessions({'calendar_sessions': plan['spec']['calendar_sessions']})
        try:
            add_regular_minutes(_at(plan['session_date'], 11, 35), 1950, calendar)
            add_regular_minutes(_at(plan['cohort']['session_dates'][-1], 11, 35),
                                1950, calendar)
            horizon_sufficient = True
        except ValueError:
            horizon_sufficient = False
        summary = {'mode': mode, 'session_date': plan['session_date'],
                   'route_id': route_id, 'forecast_denominator': len(audit),
                   'counts': counts, 'ledger_event_ids': [r['event_id'] for r in receipts],
                   'plan_sha256': plan_sha,
                   'adapter_code_sha256': file_sha(Path(__file__)),
                   'prior_store_code_sha256': file_sha(Path(__file__).with_name('prior_store.py')),
                   'input_hashes_before': before_hashes,
                   'input_hashes_after': after_hashes,
                   'evidence': bundle['evidence'], 'route_manifest': route_proof,
                   'route_artifact_hashes': {k: route_doc[k] for k in
                      ('model_sha256', 'schema_sha256', 'calibration_sha256',
                       'source_code_sha256')} if route_doc else {},
                   'stage_seconds': {'load_and_decode': load_seconds,
                                     'model_load': model_seconds,
                                     'features': sum(r['stage_seconds'].get('features', 0.)
                                                     for r in audit),
                                     'inference': sum(r['stage_seconds'].get('inference', 0.)
                                                      for r in audit),
                                     'hash_before': hash_before_seconds,
                                     'ledger': ledger_seconds,
                                     'hash_after': hash_after_seconds,
                                     'total_before_report_write': perf_counter()-started},
                   'gates': {'g1_local_path': bool(predictions) and
                                             not any(r['status'] == 'rejected' for r in audit),
                             'g1_numerical_parity': False,
                             'g2_real_receipts': False, 'g3_forward_outcomes': False,
                             'calendar_horizon_sufficient': horizon_sufficient,
                             'official_calendar_source_verified': False,
                             'action_manifest_frozen': bool(route_doc and route_doc.get('action_frozen')),
                             'runtime_30_seconds_verified': False},
                   'final_status': 'local_fixture_only' if predictions else 'local_rejected'}
        _write_new(destination / 'summary.json', summary)
        return summary
    except BaseException as exc:
        for receipt in receipts:
            if receipt['event_type'] == 'buy':
                record_protocol_violation(receipt['event_id'],
                                          'local_run_finalization_failed', fixture_ledger)
        observed_after = (_checked_input_hashes(bundle, route_proof, loaded,
                                                 prior_store, strict=False)
                          if bundle is not None else {})
        _write_new(destination / 'fatal.json',
                   {'error_type': type(exc).__name__, 'error': str(exc),
                    'candidate_audit': audit, 'input_hashes_before': before_hashes,
                    'input_hashes_after': observed_after,
                    'elapsed_seconds': perf_counter()-started})
        raise
