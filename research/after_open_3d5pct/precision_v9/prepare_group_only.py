"""R03M: rebuild frozen-2026 group facts and tensors without R03 acquisition/fit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time
import zipfile

import numpy as np
import pandas as pd

from research.group_expectation_matrix import build as original
from research.after_open_3d5pct import v6_data as vd
from research.after_open_3d5pct.focus_v8 import prepare as focus_prepare
from research.after_open_3d5pct.focus_v8 import core as focus_core
from research.after_open_3d5pct.focus_v8.core import focus_weights, mature_prior, splits
from . import data as v9


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
STOCK = PROJECT.parents[1]
V8 = PROJECT / 'runs/focus_v8_20260927'
DEFAULT_OUTPUT = PROJECT / 'runs/precision_v9_20260927/R03M_group_only_v1'
REGISTRATION = HERE / 'rounds/R03M'
ADDED = ('AAOI', 'ANET', 'AXTI', 'CIEN', 'COHR', 'CRDO', 'FN')
OLD_NON_CANDIDATE_GROUP_SYMBOLS = frozenset(
    'ARKG BBH BOTZ COPX DRAM EWH EWY FINX GLD GRID IBB IGV ITA IWO IXC IYE '
    'IYM KSTR PAVE QQQ QUAL SLV SLVP SMH SOXS SOXX VIS VLUE VT XBI'.split())
STRATEGIES = tuple(vd.STRATEGIES)
EXPECTED = {
    'config.json': '7da26f04c6c0fc3d296df932a1da153ac4541ceb821f4f1800085f53f4296057',
    'source_provenance.json': 'e2f92d6742a58c46e836cb6b30bcccfacbae57696ddfdb211e2b9ff410869d84',
    'dataset/rows.parquet': 'b6220c3645ff6477ff59763d75deaff26b6f60c7b1ad34f6d0b2b913a4fed2fe',
    'dataset/features.npz': '6bce8471011f1c2c54ce2551ba7584f18196fb5b056580e528d95532d5c0e744',
    'dataset/manifest.json': '43bbf25f88c62313b610657b77b1ae99e4b45973fa41a91cef4f8279a067d9e9',
    'inputs/market_data/universe/qqq_retrospective_v1.json': 'cd556bc08e0315d1243baa3817440ed969cdbeb17e8debb493b84867adea5218',
    'inputs/market_data/calendars/nasdaq_sessions_2026_v1.json': '1de4f9b4a472b9995fa6a7e8649fa1b39ce9e6b75e07e544f1b099569fd6989e',
    'inputs/research/group_expectation_matrix/outputs/20260925_v5/versions.json': 'fcc5c218947c666a850e1984695d1c8259b795a3514a7cbc856429a74549888b',
    'inputs/research/group_expectation_matrix/outputs/20260925_v5/memberships.json': '1a23e39efd8c0e3b2a6d793e8eb0ebc0e2c7c619a17abb77d7027cabe5c86a7d',
}
CODE_PATHS = {
    'group_build.py': Path(original.__file__),
    'focus_prepare.py': Path(focus_prepare.__file__),
    'focus_core.py': Path(focus_core.__file__),
    'v6_data.py': Path(vd.__file__),
    'v9_data.py': Path(v9.__file__),
    'r03m_prepare_group_only.py': Path(__file__),
}
DEFINITION_FILES = {
    'config.json': 'd976fa5f36eb4acce1482d45e8c29ad255e486bd37aadd8829f034518bf5ff18',
    'strategies.json': '4336add67c55c500941974ed63809b86e08461f518fb14abec6bd9dd99325700',
    'manifest.json': '83b78ff7b6c47198cbc27b9e1dc02b3118f0986c992330f03c5e134ab99c50cb',
}
DEFINITION_ROOT = STOCK / 'research/group_expectation_matrix/outputs/20260925_v5'
FOCUS_PROTOCOL = STOCK / 'research/after_open_3d5pct/focus_v8/protocol.json'
FOCUS_PROTOCOL_SHA = 'ef4e5875dac44ac550bdbe866e29c47e5d413d115c78ceaf5d4a7b66c5f4b676'
RETRY_REGISTRATION_SHA = '90a6b52e75ebf0c4a913faccfbd12e2c8dfd93830b8ac2dc0fb935e74c8f3b27'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n')


def _progress(output: Path, stage: str, **values) -> None:
    record = {'stage': stage, 'at': datetime.now(timezone.utc).isoformat(),
              'elapsed_seconds': round(time.monotonic() - _progress.started, 3), **values}
    write(output / 'progress.json', record)
    with (output / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True) + '\n')
    print(json.dumps(record, sort_keys=True), flush=True)


def _input_sources(inputs: Path, old_manifest: dict) -> dict:
    actual = {}
    for relative, expected in sorted(old_manifest['source_hashes'].items()):
        path = inputs / relative
        if not path.is_file():
            raise FileNotFoundError(f'frozen input missing: {relative}')
        found = sha(path)
        if found != expected:
            raise ValueError(f'frozen manifest source changed: {relative}')
        actual[relative] = found
    if len(actual) != 223:
        raise ValueError(f'R03M requires all 223 frozen sources, got {len(actual)}')
    return actual


def _frozen_identity() -> tuple[dict, dict, dict]:
    for relative, expected in EXPECTED.items():
        if sha(V8 / relative) != expected:
            raise ValueError(f'frozen v8 identity changed: {relative}')
    if sha(REGISTRATION / 'REGISTRATION.md') != '7cd17fbcd49b8bbe648a3015ec8fd25df5dbc96155adaabf995b6095f4db7303':
        raise ValueError('R03M registration changed')
    if sha(REGISTRATION / 'FAILURE_REVIEW_AND_RETRY_REGISTRATION.md') != RETRY_REGISTRATION_SHA:
        raise ValueError('R03M retry registration changed')
    code_expected = {
        'group_build.py': 'cdcc28163beb61ddef68a9eb9e940ff46d5963d0c25d956198cccdb65a7c9d93',
        'focus_prepare.py': 'ab99c7c8296986f2e190da76f6637a65661c049c10dab9dcc4d1f6ad966a4780',
        'focus_core.py': '093a1e399f566c437611ee14789b4c8512761f5c800cbf6a932c3f542545a590',
        'v6_data.py': 'c50fee6d0a0a4a3b4ff9103d415e4d1022646ba0425f6833b1cf63bb624b60c7',
        'v9_data.py': '06207586233fea7153094dbdb40e314cc7608bf5b7c96b269e21bbc083607a3e',
    }
    for name, expected in code_expected.items():
        if sha(CODE_PATHS[name]) != expected:
            raise ValueError(f'classification/group source changed: {name}')
    if sha(FOCUS_PROTOCOL) != FOCUS_PROTOCOL_SHA:
        raise ValueError('frozen training/prior protocol changed')
    config = json.loads((V8 / 'config.json').read_text())
    provenance = json.loads((V8 / 'source_provenance.json').read_text())
    if sorted(provenance['added']) != list(ADDED) or len(provenance['files']) != 218:
        raise ValueError('frozen added symbol/source inventory changed')
    old_manifest = json.loads((V8 / 'dataset/manifest.json').read_text())
    if old_manifest['rows'] != 14424:
        raise ValueError('old row count changed')
    return config, provenance, old_manifest


def _sessions(config: dict) -> tuple[list[dict], list[str], dict]:
    full = json.loads((V8 / 'inputs' / config['calendar']).read_text())['sessions']
    sessions = [s for s in full if '2026-01-01' <= s['session_date'] <= '2026-09-24']
    dates = [s['session_date'] for s in sessions]
    if len(sessions) != 183 or dates != sorted(set(dates)):
        raise ValueError('R03M official 2026 calendar changed')
    weeks = {}
    for i, date in enumerate(dates):
        iso = pd.Timestamp(date).isocalendar()
        weeks.setdefault(f'{iso.year}-W{iso.week:02d}', i)
    return sessions, dates, weeks


def _definitions() -> dict:
    for name, expected in DEFINITION_FILES.items():
        if sha(DEFINITION_ROOT / name) != expected:
            raise ValueError(f'original group definition changed: {name}')
    cfg = json.loads((DEFINITION_ROOT / 'config.json').read_text())
    if cfg['evaluation_as_of'] != '2026-09-24T20:00:01+00:00':
        raise ValueError('original group asof changed')
    definitions = json.loads((DEFINITION_ROOT / 'strategies.json').read_text())
    strategies = {v['strategy_id']: v for v in definitions if v['strategy_id'] in STRATEGIES}
    if set(strategies) != set(STRATEGIES):
        raise ValueError('five original strategy definitions missing')
    return {'asof': pd.Timestamp(cfg['evaluation_as_of']), 'strategies': strategies,
            'source_sha256': DEFINITION_FILES}


def _versions(sessions: list[dict], weeks: dict) -> list[dict]:
    ordered = list(weeks.items())
    created_at = datetime.now(timezone.utc).isoformat()
    official = json.loads((V8 / 'inputs/market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())['sessions']
    first_by_week = {}
    for session in official:
        iso = pd.Timestamp(session['session_date']).isocalendar()
        first_by_week.setdefault(f'{iso.year}-W{iso.week:02d}', session)
    versions = []
    for j, (week, ix) in enumerate(ordered):
        first = first_by_week[week]
        cutoff = pd.Timestamp(first['open_at']).tz_convert('UTC')
        if j + 1 < len(ordered):
            until = pd.Timestamp(first_by_week[ordered[j+1][0]]['open_at']).tz_convert('UTC')
        else:
            until = cutoff + pd.Timedelta(days=7)
        for strategy in STRATEGIES:
            versions.append({'week_id': week, 'strategy_id': strategy,
                             'version_id': f'R03M_group_only_v1:{strategy}:{week}',
                             'first_session': first['session_date'],
                             'effective_from': cutoff.isoformat(),
                             'effective_to': until.isoformat(),
                             'feature_cutoff_at': cutoff.isoformat(),
                             'membership_basis': 'causal_weekly_reconstruction',
                             'created_at': created_at,
                             'evidence_scope': 'exposed_development',
                             'source_observed_at': None,
                             'classifier_paths': ['original_float64', 'added_float32']})
    return versions


def _window(sessions: list[dict], dates: list[str], first_date: str, strategy: str) -> tuple[list[int], list[str], int]:
    n = int(strategy.split('_')[1]) if strategy.startswith('trend_') else 20
    need = n if strategy == 'liquidity' else n + 1
    ids = [i for i, date in enumerate(dates) if date < first_date][-need:]
    return ids, [dates[i] for i in ids], need


def _source_facts(symbol: str, hashes: dict) -> dict:
    return {'bars_sha256': hashes[f'market_data/us_5m/{symbol}/2026.parquet'],
            'actions_sha256': hashes[f'market_data/corporate_actions/{symbol}.parquet']}


def rebuild_original_symbol(symbol: str, inputs: Path, sessions: list[dict], dates: list[str],
                            versions: list[dict], definitions: dict, hashes: dict) -> list[dict]:
    """Use the original float64 grid/alignment/classifier, including its action rule."""
    bars = pd.read_parquet(inputs / f'market_data/us_5m/{symbol}/2026.parquet')
    action_frame = pd.read_parquet(inputs / f'market_data/corporate_actions/{symbol}.parquet')
    full_sessions = json.loads((inputs / 'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())['sessions']
    full_dates = [s['session_date'] for s in full_sessions]
    if full_dates != sorted(set(full_dates)) or not set(dates) <= set(full_dates):
        raise ValueError('original classifier official calendar changed')
    grid = original.calendar_grid(full_sessions)
    aligned, arr, valid, _, duplicate_rows, known_rows = original.prepare_bars(
        bars, grid, definitions['asof'])
    daily = original.daily_history(grid, aligned, arr, valid)
    actions = original.corporate_dates(action_frame)
    result = []
    for version in versions:
        strategy = version['strategy_id']
        n = int(strategy.split('_')[1]) if strategy.startswith('trend_') else 20
        need = n if strategy == 'liquidity' else n + 1
        prior_dates = [d for d in full_dates if d < version['first_session']]
        window_dates = prior_dates[-need:]
        cutoff = pd.Timestamp(version['feature_cutoff_at'])
        label, status, classifier_facts = original.classify_history(
            daily, prior_dates, cutoff, definitions['strategies'][strategy], actions)
        if label is not None and (status != 'classified' or len(window_dates) != need):
            raise ValueError('original classifier returned inconsistent status')
        used_end = window_dates[-1] if label is not None else None
        maximum = daily.reindex(window_dates).available.max() if label is not None else pd.NaT
        value = (classifier_facts.get('median_daily_turnover') if strategy == 'liquidity'
                 else classifier_facts.get('annualized_volatility') if strategy == 'volatility'
                 else classifier_facts.get('trend_z'))
        facts = {**classifier_facts, 'history_end': used_end,
                 'window_dates': window_dates, 'window_need': need,
                 'max_available_at': maximum.isoformat() if pd.notna(maximum) else None,
                 'classifier_value': value, 'reason': None if label is not None else status,
                 'classifier_path': 'original_float64',
                 'action_dates_in_window': [a for a in actions if window_dates and window_dates[0] < a <= window_dates[-1]],
                 'duplicate_source_rows': duplicate_rows, 'known_source_rows': known_rows,
                 **_source_facts(symbol, hashes)}
        if label is not None and (used_end >= version['first_session'] or maximum >= cutoff):
            raise ValueError('original classifier used nonprior or unavailable history')
        result.append({'symbol': symbol, 'issuer': 'ALPHABET' if symbol in ('GOOG','GOOGL') else symbol,
                       'week_id': version['week_id'], 'strategy_id': strategy,
                       'version_id': version['version_id'],
                       'group_ids': [f'{strategy}:{label}'] if label else [],
                       'status': status, 'facts': facts})
    return result


def rebuild_added_symbol(symbol: str, view: vd.SymbolData, actions: set[str],
                         sessions: list[dict], dates: list[str], versions: list[dict],
                         hashes: dict) -> list[dict]:
    """Copy the frozen focus_v8.new_memberships float32 calculation path."""
    result = []
    for version in versions:
        strategy = version['strategy_id']
        cutoff = pd.Timestamp(version['feature_cutoff_at'])
        ids, window_dates, need = _window(sessions, dates, version['first_session'], strategy)
        reason = None
        if len(ids) != need:
            reason = 'insufficient_history'
        elif not view.daily_full[ids].all():
            reason = 'incomplete_history'
        elif any(window_dates[0] < a <= window_dates[-1] for a in actions):
            reason = 'corporate_action_in_lookback'
        observed = (int(view.effective_available[ids, 66:144].view('int64').max())
                    if len(ids) == need else None)
        if reason is None and observed >= cutoff.value:
            reason = 'history_not_available'
        label = None
        value = None
        if reason is None:
            day = view.seqday[ids]
            closes = np.exp(day[:, 12].astype(float) * 5 + day[:, 0])
            returns = np.diff(np.log(closes))
            n = int(strategy.split('_')[1]) if strategy.startswith('trend_') else 20
            if strategy == 'liquidity':
                dollars = [float(view.raw5[i, 66:66+int(sessions[i]['duration_minutes'])//5, 5].sum())
                           for i in ids]
                value = float(np.median(dollars))
                label = ['low','medium','high'][int(value >= 50e6) + int(value >= 200e6)]
            elif strategy == 'volatility':
                value = float(np.std(returns, ddof=1) * np.sqrt(252))
                label = ['low','medium','high'][int(value >= .2) + int(value >= .4)]
            else:
                value = float(returns.sum() / max(np.std(returns, ddof=1) * np.sqrt(n), 1e-12))
                label = 'up' if value > 1 else 'down' if value < -1 else 'range'
            if not np.isfinite(value):
                reason, label, value = 'invalid_classifier_value', None, None
        used_end = window_dates[-1] if label is not None else None
        maximum = pd.Timestamp(observed, tz='UTC').isoformat() if label is not None else None
        if label is not None and (used_end >= version['first_session'] or observed >= cutoff.value):
            raise ValueError('added classifier used nonprior or unavailable history')
        facts = {'history_end': used_end, 'window_dates': window_dates, 'window_need': need,
                 'max_available_at': maximum, 'classifier_value': value,
                 'v8_classifier_value': value, 'reason': reason,
                 'classifier_path': 'added_float32',
                 'action_dates_in_window': [a for a in actions if window_dates and window_dates[0] < a <= window_dates[-1]],
                 **_source_facts(symbol, hashes)}
        result.append({'symbol': symbol, 'issuer': symbol, 'week_id': version['week_id'],
                       'strategy_id': strategy, 'version_id': version['version_id'],
                       'group_ids': [f'{strategy}:{label}'] if label else [],
                       'status': 'classified' if label else reason, 'facts': facts})
    return result


def _candidate_map(versions: list[dict], members: list[dict], candidates: list[str]) -> tuple[dict, dict]:
    valid = {(v['week_id'], v['strategy_id']): v for v in versions}
    own, peers = {}, defaultdict(set)
    allowed = set(candidates) - {'QQQ'}
    for member in members:
        if member['symbol'] not in allowed or len(member['group_ids']) != 1:
            continue
        version = valid.get((member['week_id'], member['strategy_id']))
        if version is None or member['version_id'] != version['version_id']:
            raise ValueError('new classified membership/version mismatch')
        key = (member['week_id'], member['strategy_id'], member['symbol'])
        if key in own:
            raise ValueError(f'duplicate classified member: {key}')
        own[key] = (member['group_ids'][0], member['facts'], version)
        peers[(member['week_id'], member['group_ids'][0])].add(member['symbol'])
    return own, peers


def _old_candidate_lookup(old_members: list[dict], candidates: list[str],
                          versions: list[dict]) -> tuple[dict, list[dict]]:
    """Compare the original group snapshot only at frozen v8 candidate roles."""
    allowed = set(candidates)
    expected_pairs = {(v['week_id'], v['strategy_id']) for v in versions}
    if len(allowed) != len(candidates) or len(expected_pairs) != len(versions):
        raise ValueError('candidate or version grid contains duplicates')
    lookup, extras = {}, []
    for member in old_members:
        if member['strategy_id'] not in STRATEGIES:
            continue
        if (member['week_id'], member['strategy_id']) not in expected_pairs:
            raise ValueError('old strategy membership outside expected weeks')
        if member['symbol'] not in allowed:
            extras.append(member)
            continue
        key = (member['symbol'], member['week_id'], member['strategy_id'])
        if key in lookup:
            raise ValueError(f'duplicate old candidate membership: {key}')
        lookup[key] = member
    required = {(symbol, week, strategy) for symbol in allowed - set(ADDED)
                for week, strategy in expected_pairs}
    missing = required - set(lookup)
    if missing:
        raise ValueError(f'original candidate membership missing: {min(missing)}; count={len(missing)}')
    return lookup, extras


def _read_old_group_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as old:
        return old['group'], old['group_seq'], old['y']


def _npy_payload_hashes(path: Path, names: tuple[str, ...]) -> dict:
    hashes = {}
    with zipfile.ZipFile(path) as archive:
        for name in names:
            h = hashlib.sha256()
            with archive.open(name + '.npy') as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    h.update(block)
            hashes[name] = h.hexdigest()
    return hashes


def _archive(old_path: Path, new_path: Path, groups: np.ndarray, seq: np.ndarray) -> dict:
    protected = ('x5', 'x60', 'xday', 'y')
    before = _npy_payload_hashes(old_path, protected)
    with zipfile.ZipFile(old_path) as source, zipfile.ZipFile(new_path, 'w', compression=zipfile.ZIP_DEFLATED,
                                                             compresslevel=6, allowZip64=True) as target:
        for name in ('x5','x60','xday','group','group_seq','y'):
            if name in protected:
                with source.open(name + '.npy') as src, target.open(name + '.npy', 'w', force_zip64=True) as dst:
                    shutil.copyfileobj(src, dst, length=1 << 20)
            else:
                with target.open(name + '.npy', 'w', force_zip64=True) as dst:
                    np.lib.format.write_array(dst, groups if name == 'group' else seq,
                                              allow_pickle=False)
    after = _npy_payload_hashes(new_path, protected)
    if before != after:
        raise ValueError('protected own/label NPY bytes changed')
    return {'old': before, 'new': after}


def _prior_and_supervision(rows: pd.DataFrame, y: np.ndarray) -> dict:
    if len(rows) != len(y) or y.shape != (14424, 3, 3):
        raise ValueError('nine-target supervision shape changed')
    y9 = y.reshape(len(rows), 9)
    prior, count = mature_prior(rows, y9)
    weight = focus_weights(rows)
    available = pd.to_datetime(rows.label_available_at, utc=True).astype('int64').to_numpy()
    end = pd.to_datetime(rows.label_end_at, utc=True).astype('int64').to_numpy()
    decisions = pd.to_datetime(rows.decision_at, utc=True).astype('int64').to_numpy()
    sample_ids = rows.sample_id.to_numpy()
    history_digest = hashlib.sha256()
    for _, ids0 in rows.groupby('symbol').indices.items():
        ids = np.array(sorted(ids0, key=lambda i: decisions[i]))
        for i in ids:
            history = ids[(decisions[ids] < decisions[i]) &
                          (available[ids] < decisions[i]) &
                          (end[ids] < decisions[i])][-63:]
            if count[i] != len(history) or not np.array_equal(
                    prior[i], ((y9[history].sum(0) + 1) / (len(history) + 2)).astype(np.float32)):
                raise ValueError(f'mature prior/history mismatch: {sample_ids[i]}')
            history_digest.update((sample_ids[i] + '\0' + '\0'.join(sample_ids[history]) + '\n').encode())
    result = {'prior_sha256': hashlib.sha256(prior.tobytes()).hexdigest(),
              'prior_count_sha256': hashlib.sha256(count.tobytes()).hexdigest(),
              'mature_history_ids_sha256': history_digest.hexdigest(),
              'weights_sha256': hashlib.sha256(weight.tobytes()).hexdigest(),
              'positive_weight_ids_sha256': hashlib.sha256('\n'.join(rows.loc[weight > 0, 'sample_id']).encode()).hexdigest(),
              'positive_weight_count': int(np.sum(weight > 0)), 'prior_count_total': int(count.sum())}
    from research.after_open_3d5pct.focus_v8.core import PROTOCOL
    for fold in PROTOCOL['folds']:
        parts = splits(rows, fold)
        result[f'fold_{fold["id"]}'] = {
            name: {'count': len(ids),
                   'sample_ids_sha256': hashlib.sha256('\n'.join(rows.iloc[ids].sample_id).encode()).hexdigest(),
                   'positive_weight_ids_sha256': hashlib.sha256('\n'.join(rows.iloc[ids][weight[ids] > 0].sample_id).encode()).hexdigest()}
            for name, ids in parts.items()}
    return result


def _audit_groups(rows: pd.DataFrame, old_group: np.ndarray, old_seq: np.ndarray,
                  new_group: np.ndarray, new_seq: np.ndarray, output: Path) -> dict:
    if old_group.shape != new_group.shape or old_seq.shape != new_seq.shape:
        raise ValueError('group tensor shape changed')
    slots = list(STRATEGIES) + ['QQQ']
    summary = {'group_changed_rows': 0, 'group_seq_changed_rows': 0,
               'group_changed_slot_channels': np.zeros((6,10),np.int64),
               'group_seq_changed_week_slot_channels': np.zeros((6,6,10),np.int64),
               'max_group_abs': 0., 'max_group_seq_abs': 0., 'first_changed': []}
    with (output / 'key_audit.jsonl').open('w') as stream:
        for i, row in enumerate(rows.itertuples(index=False)):
            a = np.abs(old_group[i] - new_group[i]); b = np.abs(old_seq[i] - new_seq[i])
            ag = a > 0; bg = b > 0
            if ag.any(): summary['group_changed_rows'] += 1
            if bg.any(): summary['group_seq_changed_rows'] += 1
            summary['group_changed_slot_channels'] += ag
            summary['group_seq_changed_week_slot_channels'] += bg
            summary['max_group_abs'] = max(summary['max_group_abs'], float(np.max(a)))
            summary['max_group_seq_abs'] = max(summary['max_group_seq_abs'], float(np.max(b)))
            if (ag.any() or bg.any()) and len(summary['first_changed']) < 20:
                summary['first_changed'].append({'sample_id': row.sample_id,
                                                 'group_max_abs': float(np.max(a)),
                                                 'group_seq_max_abs': float(np.max(b))})
            stream.write(json.dumps({'sample_id': row.sample_id, 'row_index': i,
                                     'group_max_abs': float(np.max(a)),
                                     'group_seq_max_abs': float(np.max(b)),
                                     'changed_current_slots': [slots[j] for j in range(6) if ag[j].any()],
                                     'changed_seq_week_slots': [[int(w), slots[j]] for w,j in zip(*np.where(bg.any(axis=2)))],
                                     'current_group_valid_count': int(np.sum(new_group[i,:5,9])),
                                     'old_group_valid_count': int(np.sum(old_group[i,:5,9]))},
                                    sort_keys=True) + '\n')
    summary['group_changed_slot_channels'] = summary['group_changed_slot_channels'].tolist()
    summary['group_seq_changed_week_slot_channels'] = summary['group_seq_changed_week_slot_channels'].tolist()
    return summary


def prepare(output: Path = DEFAULT_OUTPUT) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'R03M run must be new: {output}')
    config, provenance, old_manifest = _frozen_identity()
    inputs = V8 / 'inputs'
    universe = json.loads((inputs / config['universe']).read_text())
    candidates = sorted(m['symbol'] for m in universe['members'] if m['role'] == 'candidate')
    frozen_roles = {m['symbol']:m['role'] for m in universe['members']}
    if len(candidates) != 108 or len(set(candidates)) != 108 or 'QQQ' in candidates:
        raise ValueError('frozen R03M candidate pool changed')
    source_before = _input_sources(inputs, old_manifest)
    definitions = _definitions()
    output.mkdir(parents=True)
    _progress.started = time.monotonic()
    try:
        for name in ('STATE.md','BACKLOG.md','MATRIX.md','REGISTRATION.md',
                     'FAILURE_REVIEW_AND_RETRY_REGISTRATION.md'):
            shutil.copy2(REGISTRATION / name, output / name)
        for relative in EXPECTED:
            target = output / 'frozen_identity' / relative
            if relative in ('config.json','source_provenance.json','dataset/manifest.json'):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(V8 / relative, target)
        definition_dir = output / 'source_definitions'
        definition_dir.mkdir()
        for name in DEFINITION_FILES:
            shutil.copy2(DEFINITION_ROOT / name, definition_dir / name)
        shutil.copy2(FOCUS_PROTOCOL, definition_dir / 'focus_v8_protocol.json')
        code_dir = output / 'source_code'
        code_dir.mkdir()
        code_hashes = {}
        for name, path in CODE_PATHS.items():
            shutil.copy2(path, code_dir / name)
            code_hashes[name] = sha(code_dir / name)
            if code_hashes[name] != sha(path):
                raise ValueError(f'source changed during snapshot: {name}')
        write(output / 'source_code_sha256.json', code_hashes)
        write(output / 'inputs_before_sha256.json', {'frozen_identity': EXPECTED,
               'manifest_sources': source_before, 'source_code': code_hashes,
               'source_definitions': DEFINITION_FILES,
               'focus_v8_protocol_sha256': FOCUS_PROTOCOL_SHA,
               'registration_sha256': sha(REGISTRATION / 'REGISTRATION.md'),
               'retry_registration_sha256': RETRY_REGISTRATION_SHA})
        _progress(output, 'sources_verified', source_count=len(source_before), candidate_count=len(candidates))
        sessions, dates, weeks = _sessions(config)
        versions = _versions(sessions, weeks)
        old_members = json.loads((inputs / config['group_run'] / 'memberships.json').read_text())
        old_lookup, excluded_old_benchmarks = _old_candidate_lookup(old_members, candidates, versions)
        if (len(excluded_old_benchmarks) != 5850 or
                {m['symbol'] for m in excluded_old_benchmarks} != OLD_NON_CANDIDATE_GROUP_SYMBOLS):
            raise ValueError('frozen old non-candidate metadata scope changed')
        members = []
        views = {}
        old_root = vd.ROOT
        try:
            vd.ROOT = inputs
            for ix, symbol in enumerate(candidates + ['QQQ']):
                view = vd.load_symbol(symbol, sessions, config)
                if view is None:
                    raise FileNotFoundError(f'frozen 2026 view missing: {symbol}')
                views[symbol] = view
                if symbol != 'QQQ':
                    if symbol in ADDED:
                        members.extend(rebuild_added_symbol(symbol, view, vd._split_days(symbol),
                                                            sessions, dates, versions, source_before))
                    else:
                        members.extend(rebuild_original_symbol(symbol, inputs, sessions, dates,
                                                               versions, definitions, source_before))
                if ix % 10 == 9 or ix == len(candidates):
                    _progress(output, 'classifying', symbols_done=ix+1,
                              symbols_total=len(candidates)+1, memberships=len(members))
        finally:
            vd.ROOT = old_root
        if len(members) != len(candidates)*len(versions):
            raise ValueError('full candidate×week×strategy membership coverage missing')
        new_lookup = {(m['symbol'],m['week_id'],m['strategy_id']):m for m in members}
        if len(new_lookup) != len(members):
            raise ValueError('duplicate new membership key')
        classification = Counter()
        examples = []
        group_dir = output / 'group_metadata'
        group_dir.mkdir()
        with (group_dir / 'membership_audit.jsonl').open('w') as audit_stream:
            for key, member in sorted(new_lookup.items()):
                older = old_lookup.get(key)
                category = 'old_missing' if older is None else (
                    'same' if older['group_ids'] == member['group_ids'] and older['status'] == member['status']
                    else 'changed')
                classification[(member['strategy_id'],category)] += 1
                record = {'symbol':key[0], 'week_id':key[1], 'strategy_id':key[2],
                          'old_group_ids':older['group_ids'] if older else None,
                          'new_group_ids':member['group_ids'],
                          'old_status':older['status'] if older else None,
                          'new_status':member['status'],
                          'new_reason':member['facts']['reason'],
                          'new_facts':member['facts'], 'classification_result':category}
                audit_stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
                if category != 'same' and len(examples) < 100:
                    examples.append({k:v for k,v in record.items() if k != 'new_facts'})
        if set(old_lookup) - set(new_lookup):
            raise ValueError('old classified/empty membership key lost')
        write(group_dir / 'versions.json', versions)
        write(group_dir / 'memberships.json', members)
        write(group_dir / 'classification_audit.json', {
            'counts': {f'{k[0]}/{k[1]}':v for k,v in sorted(classification.items())},
            'examples': examples, 'old_keys':len(old_lookup), 'new_keys':len(new_lookup),
            'old_non_candidate_members_excluded_from_peer_comparison':len(excluded_old_benchmarks),
            'old_non_candidate_symbols_and_frozen_roles':{
                symbol:frozen_roles.get(symbol, 'absent')
                for symbol in sorted({m['symbol'] for m in excluded_old_benchmarks})}})
        _progress(output, 'memberships_rebuilt', membership_count=len(members),
                  classification_changes=sum(v for (s,c),v in classification.items() if c=='changed'))
        own, peers = _candidate_map(versions, members, candidates)
        old_rows = pd.read_parquet(V8 / 'dataset/rows.parquet')
        if (len(old_rows) != 14424 or old_rows.sample_id.duplicated().any() or
            old_rows[['symbol','session_date']].duplicated().any() or
            not set(old_rows.symbol) <= set(candidates)):
            raise ValueError('old key population invalid')
        date_index = {date:i for i,date in enumerate(dates)}
        representatives = {}
        for day in range(len(dates)):
            seen = {}
            for older in range(day,-1,-1):
                iso = pd.Timestamp(dates[older]).isocalendar()
                week = f'{iso.year}-W{iso.week:02d}'
                seen.setdefault(week,older)
                if len(seen) == 6:
                    break
            representatives[day] = sorted(seen.values())[-6:]
        groups = np.zeros((len(old_rows),6,10),np.float32)
        sequence = np.zeros((len(old_rows),6,6,10),np.float32)
        cache = {}
        for i,row in enumerate(old_rows.itertuples(index=False)):
            day = date_index.get(row.session_date)
            if day is None:
                raise ValueError(f'old key outside 2026 calendar: {row.sample_id}')
            def state(index):
                key = (row.symbol,index)
                if key not in cache:
                    cache[key] = v9._group_state(row.symbol,index,dates,sessions,views,own,peers)
                return cache[key]
            groups[i] = state(day)
            reps = representatives[day]
            for j,older in enumerate(reps):
                sequence[i,6-len(reps)+j] = state(older)
            if not np.array_equal(groups[i],sequence[i,-1]):
                raise ValueError(f'current group/sequence mismatch: {row.sample_id}')
            if i % 1000 == 999 or i+1 == len(old_rows):
                _progress(output, 'rebuilding_group_tensors', keys_done=i+1,
                          keys_total=len(old_rows), cached_states=len(cache))
        old_group, old_seq, y = _read_old_group_arrays(V8 / 'dataset/features.npz')
        group_audit = _audit_groups(old_rows,old_group,old_seq,groups,sequence,output)
        write(output / 'group_audit.json', group_audit)
        new_rows = old_rows.copy(deep=True)
        new_rows['group_valid_count'] = np.sum(groups[:,:5,9],axis=1).astype(np.int64)
        for column in old_rows.columns:
            if column == 'group_valid_count':
                continue
            if not old_rows[column].equals(new_rows[column]):
                raise ValueError(f'protected old row column changed: {column}')
        if not old_rows.sample_id.equals(new_rows.sample_id):
            raise ValueError('old row key order changed')
        dataset = output / 'dataset'; dataset.mkdir()
        new_rows.to_parquet(dataset / 'rows.parquet', index=False)
        array_hashes = _archive(V8 / 'dataset/features.npz',dataset / 'features.npz',groups,sequence)
        with np.load(dataset / 'features.npz') as archive:
            if not np.array_equal(archive['group'],groups) or not np.array_equal(archive['group_seq'],sequence):
                raise ValueError('written group tensors differ')
            if not np.array_equal(archive['y'],y):
                raise ValueError('written labels differ')
        before_supervision = _prior_and_supervision(old_rows,y)
        after_supervision = _prior_and_supervision(new_rows,y)
        if before_supervision != after_supervision:
            raise ValueError('prior/split/positive-weight supervision changed')
        new_rows_check = pd.read_parquet(dataset / 'rows.parquet')
        if (len(new_rows_check) != len(old_rows) or
            any(not old_rows[c].equals(new_rows_check[c]) for c in old_rows if c!='group_valid_count')):
            raise ValueError('written protected rows changed')
        write(output / 'supervision_audit.json', {'old':before_supervision,'new':after_supervision,
              'protected_npy_payload_sha256':array_hashes,
              'key_population':{'old_count':len(old_rows), 'new_count':len(new_rows),
                                'old_only':0,'new_only':0,'duplicate_old':0,'duplicate_new':0,
                                'same_order':True},
              'recipe':{'seed':focus_core.PROTOCOL['seed'],
                        'protocol_id':focus_core.PROTOCOL['id'],
                        'protocol_sha256':FOCUS_PROTOCOL_SHA},
              'group_valid_count_changed_rows':int(np.sum(old_rows.group_valid_count.to_numpy() !=
                                                         new_rows.group_valid_count.to_numpy()))})
        write(dataset / 'manifest.json', {'protocol':'R03M_group_only_v1',
              'source_run':str(V8),'source_manifest_sha256':sha(V8/'dataset/manifest.json'),
              'source_hashes':source_before,'group_versions_sha256':sha(group_dir/'versions.json'),
              'group_memberships_sha256':sha(group_dir/'memberships.json'),
              'builder_sha256':code_hashes['r03m_prepare_group_only.py'],
              'rows':len(new_rows),'candidate_count':len(candidates),
              'benchmark':'QQQ','group_semantics':'v9_causal_history_end_v1',
              'evidence_scope':'exposed_development',
              'availability_quality':'assumed_archived_end_plus_1s_no_received_at',
              'no_training':True})
        after_sources = _input_sources(inputs,old_manifest)
        if source_before != after_sources:
            raise ValueError('frozen sources changed during R03M run')
        write(output / 'inputs_after_sha256.json', {'manifest_sources':after_sources,
              'frozen_identity':{key:sha(V8/key) for key in EXPECTED},
              'source_code':{name:sha(path) for name,path in CODE_PATHS.items()},
              'source_definitions':{name:sha(DEFINITION_ROOT/name) for name in DEFINITION_FILES},
              'focus_v8_protocol_sha256':sha(FOCUS_PROTOCOL),
              'retry_registration_sha256':sha(REGISTRATION/'FAILURE_REVIEW_AND_RETRY_REGISTRATION.md')})
        if {name:sha(path) for name,path in CODE_PATHS.items()} != code_hashes:
            raise ValueError('source code changed during R03M run')
        if {name:sha(DEFINITION_ROOT/name) for name in DEFINITION_FILES} != DEFINITION_FILES:
            raise ValueError('group definition changed during R03M run')
        if sha(FOCUS_PROTOCOL) != FOCUS_PROTOCOL_SHA:
            raise ValueError('focus training/prior protocol changed during R03M run')
        if sha(REGISTRATION/'FAILURE_REVIEW_AND_RETRY_REGISTRATION.md') != RETRY_REGISTRATION_SHA:
            raise ValueError('retry registration changed during R03M run')
        result = {'status':'built_group_only_parity_unreviewed',
                  'row_count':len(new_rows),'candidate_count':len(candidates),
                  'source_count':len(source_before),'membership_count':len(members),
                  'old_membership_count':len(old_lookup),
                  'group_audit':group_audit,
                  'classification_counts':{f'{k[0]}/{k[1]}':v for k,v in sorted(classification.items())},
                  'dataset_sha256':{p:sha(dataset/p) for p in ('rows.parquet','features.npz','manifest.json')},
                  'old_dataset_sha256':{p:sha(V8/'dataset'/p) for p in ('rows.parquet','features.npz','manifest.json')},
                  'source_before_sha256':sha(output/'inputs_before_sha256.json'),
                  'source_after_sha256':sha(output/'inputs_after_sha256.json'),
                  'elapsed_seconds':round(time.monotonic()-_progress.started,3),
                  'max_rss_raw':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'g2_eligible':False,'trained':False,'certificate':None}
        write(output / 'result.json',result)
        _progress(output,'complete',rows=len(new_rows),group_changed=group_audit['group_changed_rows'],
                  group_seq_changed=group_audit['group_seq_changed_rows'])
        return result
    except Exception as exc:
        _progress(output,'failed',error_type=type(exc).__name__,message=str(exc))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    prepare(args.output)
