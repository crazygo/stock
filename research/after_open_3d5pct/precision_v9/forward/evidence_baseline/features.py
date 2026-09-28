"""Feature-only, version-filtered 11:30 snapshot encoder.

The input is already local/in memory.  This module never opens market-data paths,
computes an outcome, or reads a future entry price.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as d
from ...focus_v8.core import TARGETS, path_summary
from ...train_multiscale_v6 import tabular
from ...iterations_v7.b_group.experiment import curve_features, group_relative_features

SHAPES = {'x5': (8, 192, 14), 'x60': (31, 17, 14),
          'xday': (126, 14), 'group_seq': (6, 6, 10), 'prior': (9,)}
GROUP_ROUTES = {'B_group', 'C_group', 'C_no_daily'}
ROUTES = GROUP_ROUTES | {'B_no_group', 'C_no_group'}
BAR_COLUMNS = {'session_date', 'start_at_et', 'end_at', 'available_at', 'session_type',
               'price_basis', *d.PRICE_COLUMNS}
LABEL_COLUMNS = {'y', 'entry_price', 'label_end_at', 'label_available_at',
                 'terminal_1d', 'terminal_3d', 'terminal_5d'}


def _ts(value):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError('timestamp must be present and timezone-aware')
    return timestamp.tz_convert('UTC')


def _select(symbol, frame, cutoff, deadline, mode):
    """Select the last eligible revision before encoding any rolling denominator."""
    if not isinstance(frame, pd.DataFrame) or not BAR_COLUMNS <= set(frame):
        raise ValueError(f'{symbol}: missing source bar columns')
    if LABEL_COLUMNS & set(frame) or any(c.startswith('y_') for c in frame):
        raise ValueError('current/future outcome or entry columns are forbidden in snapshot')
    if mode == 'live' and 'received_at' not in frame:
        raise ValueError(f'{symbol}: live source has no received_at')
    if frame.empty:
        raise ValueError(f'{symbol}: empty source')
    if set(frame.price_basis.dropna()) != {'NONE'}:
        raise ValueError(f'{symbol}: price_basis is not frozen NONE')
    chosen = []
    ignored = {'future_bar': 0, 'unavailable': 0, 'late_or_missing_receipt': 0,
               'unverified_keys': 0}
    visible_keys = set()
    rejected_keys = set()
    for row in frame.to_dict('records'):
        end = _ts(row['end_at'])
        if end > cutoff:
            ignored['future_bar'] += 1
            continue
        key = (row['session_date'],row['start_at_et'])
        available = _ts(row['available_at'])
        if available < end:
            raise ValueError(f'{symbol}: available_at precedes bar_end')
        if available > deadline:
            ignored['unavailable'] += 1
            rejected_keys.add(key)
            continue
        received = row.get('received_at')
        if mode == 'live' and (received is None or pd.isna(received) or _ts(received) > deadline):
            ignored['late_or_missing_receipt'] += 1
            rejected_keys.add(key)
            continue
        if received is not None and not pd.isna(received):
            received = _ts(received)
            if received > deadline:
                ignored['late_or_missing_receipt'] += 1
                rejected_keys.add(key)
                continue
        else:
            received = None
        row['_end'] = end
        row['_available'] = available
        row['_received'] = received
        identity = row.get('source_id')
        if mode == 'live' and (identity is None or pd.isna(identity) or not str(identity)):
            raise ValueError(f'{symbol}: live bar version lacks observed source_id')
        row['_source_id'] = (str(identity) if identity is not None and pd.notna(identity) else
                             f"{symbol}:{row['session_date']}:{row['start_at_et']}:{available.isoformat()}")
        chosen.append(row)
        visible_keys.add(key)
    versions = {}
    for row in chosen:
        key = (row['session_date'], row['start_at_et'])
        rank = (row['_received'] or row['_available'], row['_available'], row['_source_id'])
        if key not in versions or rank > versions[key][0]:
            versions[key] = (rank, row)
    ignored['unverified_keys'] = len(rejected_keys-visible_keys)
    return [v[1] for v in versions.values()], ignored


def _view(symbol, bars, sessions):
    dates = [s['session_date'] for s in sessions]
    index = {date: i for i, date in enumerate(dates)}
    raw = np.full((len(dates), d.SLOTS, 6), np.nan, np.float32)
    availability = np.full((len(dates), d.SLOTS), np.datetime64('NaT', 'ns'))
    source_ids = []
    for row in bars:
        day = index.get(row['session_date'])
        if day is None:
            continue
        start = _ts(row['start_at_et'])
        et = start.tz_convert('America/New_York')
        slot = (et.hour-4)*12 + et.minute//5
        if et.strftime('%Y-%m-%d') != row['session_date'] or et.minute % 5 or not 0 <= slot < d.SLOTS:
            raise ValueError(f'{symbol}: invalid bar clock')
        if row['_end'] != start + pd.Timedelta(minutes=5):
            raise ValueError(f'{symbol}: invalid 5m duration')
        regular_end = 66 + int(sessions[day]['duration_minutes'])//5
        expected = 'pre_market' if slot < 66 else 'regular' if slot < regular_end else 'post_market'
        if row['session_type'] != expected:
            raise ValueError(f'{symbol}: incorrect session type')
        values = np.array([row[c] for c in d.PRICE_COLUMNS], np.float32)
        if (not np.isfinite(values).all() or values[0] <= 0 or values[2] <= 0 or
            values[1] < max(values[0], values[2], values[3]) or
            values[2] > min(values[0], values[1], values[3]) or values[4] < 0):
            raise ValueError(f'{symbol}: invalid OHLCV')
        raw[day, slot] = values
        availability[day, slot] = np.datetime64(row['_available'].tz_localize(None))
        source_ids.append(row['_source_id'])
    starts = np.empty((len(dates), d.SLOTS), np.float64)
    for day, session in enumerate(sessions):
        base = _ts(session['open_at']).tz_convert('America/New_York').normalize() + pd.Timedelta(hours=4)
        starts[day] = pd.date_range(base, periods=d.SLOTS, freq='5min').tz_convert('UTC').asi8
    durations = np.full((len(dates), d.SLOTS), 5, np.float32)
    types = np.broadcast_to(np.r_[np.zeros(66), np.ones(78), np.full(48, 2)],
                            (len(dates), d.SLOTS)).copy()
    regular_ends = np.array([66+int(s['duration_minutes'])//5 for s in sessions])
    for day, end in enumerate(regular_ends):
        types[day, end:] = 2
    positions = np.broadcast_to(np.arange(d.SLOTS)/d.SLOTS, (len(dates), d.SLOTS))
    seq5 = d._encode(raw, durations, types, positions, starts, d._prior_median(raw[:, :, 5]))
    raw60 = np.full((len(dates), len(d.HOUR_BINS), 6), np.nan, np.float32)
    dur60 = np.zeros((len(dates), len(d.HOUR_BINS)), np.float32)
    type60 = np.zeros_like(dur60)
    pos60 = np.zeros_like(dur60)
    start60 = np.zeros_like(dur60, np.float64)
    for j, (a, b, kind) in enumerate(d.HOUR_BINS):
        block = raw[:, a:b]
        good = np.isfinite(block[:, :, :4]).all(axis=(1, 2))
        good &= ~((a < regular_ends) & (regular_ends < b))
        for day in np.flatnonzero(good):
            raw60[day, j] = [block[day, 0, 0], np.max(block[day, :, 1]),
                             np.min(block[day, :, 2]), block[day, -1, 3],
                             np.sum(block[day, :, 4]), np.sum(block[day, :, 5])]
        dur60[:, j] = (b-a)*5
        type60[:, j] = np.where(a >= regular_ends, 2, kind)
        pos60[:, j] = a/d.SLOTS
        start60[:, j] = starts[:, a]
    seq60 = d._encode(raw60, dur60, type60, pos60, start60, d._prior_median(raw60[:, :, 5]))
    daily = np.full((len(dates), 6), np.nan, np.float32)
    full = np.zeros(len(dates), bool)
    cutoff_return = np.full(len(dates), np.nan, np.float32)
    cutoff_rvol = np.full(len(dates), np.nan, np.float32)
    prefix_vol = np.full(len(dates), np.nan, np.float32)
    for day, session in enumerate(sessions):
        n = int(session['duration_minutes'])//5
        reg = raw[day, 66:66+n]
        full[day] = len(reg) == n and np.isfinite(reg[:, :4]).all()
        if full[day]:
            daily[day] = [reg[0, 0], np.max(reg[:, 1]), np.min(reg[:, 2]),
                          reg[-1, 3], np.sum(reg[:, 4]), np.sum(reg[:, 5])]
        pre = raw[day, 66:90]
        if np.isfinite(pre[:, :4]).all():
            cutoff_return[day] = np.log(pre[-1, 3]/pre[0, 0])
            prefix_vol[day] = np.sum(pre[:, 5])
        past = prefix_vol[max(0, day-20):day]
        median = np.nanmedian(past) if np.isfinite(past).sum() >= 5 else np.nan
        if np.isfinite(median) and median > 0 and np.isfinite(prefix_vol[day]):
            cutoff_rvol[day] = prefix_vol[day]/median
    seqday = d._encode(daily[:, None, :],
                       np.array([[s['duration_minutes'] for s in sessions]], np.float32).T,
                       np.full((len(dates), 1), 1), np.zeros((len(dates), 1)),
                       np.array([[d._iso(s['open_at']).value] for s in sessions], np.float64),
                       d._prior_median(daily[:, None, 5]))[:, 0]
    observed_daily = availability.view('int64').max(axis=1)
    for day, session in enumerate(sessions):
        decision = _ts(session['open_at']) + pd.Timedelta(hours=2, seconds=30)
        past_max = observed_daily[max(0, day-146):day].max(initial=np.iinfo('int64').min)
        today_max = availability[day, :90].view('int64').max(initial=np.iinfo('int64').min)
        if past_max > decision.value or today_max > decision.value:
            cutoff_return[day] = np.nan
            cutoff_rvol[day] = np.nan
    return d.SymbolData(raw, availability, seq5, seq60, seqday, full,
                        cutoff_return, cutoff_rvol, source_ids, 0)


def _prior(symbol, decision, store):
    if not isinstance(store, (list, tuple)):
        raise ValueError('mature_prior_store must list complete historical rows')
    eligible = []
    for row in store:
        if row.get('symbol') != symbol:
            continue
        if not {'sample_id', 'decision_at', 'label_end_at', 'label_available_at', 'y'} <= set(row):
            raise ValueError('mature prior row lacks complete nine-target lineage')
        if (_ts(row['decision_at']) < decision and _ts(row['label_end_at']) < decision and
            _ts(row['label_available_at']) < decision):
            y = np.asarray(row['y'], dtype=np.float32)
            if y.shape != (9,) or not np.isfinite(y).all():
                raise ValueError('mature prior has incomplete nine labels')
            eligible.append((_ts(row['decision_at']), str(row['sample_id']), y))
    eligible.sort(key=lambda item: (item[0], item[1]))
    old = eligible[-63:]
    value = ((np.sum([x[2] for x in old], axis=0) + 1)/(len(old)+2)).astype(np.float32) if old else np.full(9, .5, np.float32)
    return value, [x[1] for x in old]


def build_features_asof(snapshot, decision, schema, metadata, mature_prior_store):
    """Return X, lineage, rejections; no label or future-entry input is accepted."""
    required = {'route_id', 'targets', 'feature_columns', 'input_shapes', 'price_basis',
                'tensor_order', 'tabular_width', 'path_width', 'curve_width',
                'relative_width', 'prior_rule', 'group_mode', 'normalizer',
                'support_rule', 'tabular_columns', 'path_columns',
                'curve_columns', 'relative_columns'}
    missing = required-set(schema)
    if missing:
        raise ValueError(f'inference schema missing: {sorted(missing)}')
    route = schema['route_id']
    if (route not in ROUTES or tuple(schema['targets']) != tuple(TARGETS) or
        tuple(schema['feature_columns']) != d.FEATURE_COLUMNS or
        schema['input_shapes'] != {k:list(v) for k,v in SHAPES.items()} or
        schema['price_basis'] != 'NONE' or
        schema['tensor_order'] != ['x5','x60','xday','group_seq','prior'] or
        schema['prior_rule'] != 'own_complete_nine_beta_1_1_last_63' or
        schema['group_mode'] != ('causal_weekly' if route in GROUP_ROUTES else 'excluded') or
        schema['normalizer'] != 'fixed_no_fitted_scaler' or
        schema['support_rule'] not in ('none','checkpoint_buffers')):
        raise ValueError('inference schema differs from frozen route contract')
    if not isinstance(snapshot, dict) or not {'bars','sessions','mode'} <= set(snapshot):
        raise ValueError('snapshot requires in-memory bars, sessions, mode')
    if {'y','labels','entry_price'} & set(snapshot):
        raise ValueError('outcome or future entry is forbidden in feature snapshot')
    mode = snapshot['mode']
    if mode not in ('historical_fixture','live'):
        raise ValueError('unknown snapshot mode')
    if not {'symbol','session_date','cutoff_at','decision_at','information_deadline_at'} <= set(decision):
        raise ValueError('decision clock or candidate missing')
    symbol, date = decision['symbol'], decision['session_date']
    cutoff, anchor = _ts(decision['cutoff_at']), _ts(decision['decision_at'])
    deadline = _ts(decision['information_deadline_at'])
    if (cutoff.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M') != f'{date} 11:30' or
        anchor != cutoff or deadline != cutoff + pd.Timedelta(seconds=30)):
        raise ValueError('decision is not the frozen 11:30/11:30:30 ET clock')
    sessions = [s for s in snapshot['sessions'] if s['session_date'] <= date]
    dates = [s['session_date'] for s in sessions]
    if not dates or dates != sorted(set(dates)) or dates[-1] != date:
        raise ValueError('official session chronology or candidate missing')
    day = len(dates)-1
    if _ts(sessions[day]['close_at']) <= cutoff:
        return {'X':None,'lineage':{},'rejections':['no_1130_session']}
    if 'split_days' not in metadata or symbol not in metadata['split_days']:
        raise ValueError('corporate action input-window status missing')
    if any(x in set(metadata['split_days'][symbol]) for x in dates[max(0,day-146):day+1]):
        return {'X':None,'lineage':{},'rejections':['split_in_input_window']}
    representative = {}
    for older in range(day,-1,-1):
        iso = pd.Timestamp(dates[older]).isocalendar()
        week = f'{iso.year}-W{iso.week:02d}'
        representative.setdefault(week,older)
        if len(representative) >= 6:
            break
    versions, memberships, group_version_ids = [], [], []
    group_verified = route not in GROUP_ROUTES
    symbols = [symbol]
    if route in GROUP_ROUTES:
        if not {'versions','memberships','universe_symbols'} <= set(metadata):
            raise ValueError('group route lacks frozen member/version metadata')
        for version in metadata['versions']:
            if version.get('strategy_id') not in d.STRATEGIES:
                continue
            if not {'version_id','week_id','strategy_id','effective_from','effective_to',
                    'feature_cutoff_at','membership_basis'} <= set(version):
                raise ValueError('group version metadata incomplete')
            if (_ts(version['feature_cutoff_at']) > deadline or
                _ts(version['effective_from']) > deadline):
                continue
            if mode == 'live' and any(k not in version or _ts(version[k]) > deadline
                                      for k in ('created_at','published_at','received_at')):
                continue
            versions.append(version)
            group_version_ids.append(str(version['version_id']))
        allowed_ids = set(group_version_ids)
        for member in metadata['memberships']:
            if member['version_id'] not in allowed_ids:
                continue
            if mode == 'live' and ('received_at' not in member or _ts(member['received_at']) > deadline):
                continue
            memberships.append(member)
        relevant = [m for m in memberships if m['symbol']==symbol and
                    m['week_id'] in representative and m['strategy_id'] in d.STRATEGIES and
                    len(m['group_ids'])==1]
        group_ids = {(m['week_id'],m['group_ids'][0]) for m in relevant}
        members = {m['symbol'] for m in memberships
                   if m['strategy_id'] in d.STRATEGIES and
                   (m['week_id'],next(iter(m['group_ids']),None)) in group_ids}
        symbols = sorted((members & set(metadata['universe_symbols'])) | {'QQQ',symbol})
        # A missing real receipt must not silently turn a previously classified
        # current peer into an absent feature and then an eligible live buy.
        expected = [m for m in metadata['memberships'] if m['symbol']==symbol and
                    m['week_id'] in representative and m['strategy_id'] in d.STRATEGIES and
                    len(m['group_ids'])==1 and
                    any(v['version_id']==m['version_id'] and
                        _ts(v['feature_cutoff_at'])<=deadline and
                        _ts(v['effective_from'])<=deadline for v in metadata['versions'])]
        group_verified = (mode == 'live' and bool(metadata.get('group_versions_complete')) and
                          bool(metadata.get('membership_snapshot_sha256')) and
                          all(m in memberships for m in expected))
    if any(s not in snapshot['bars'] for s in symbols):
        return {'X':None,'lineage':{},'rejections':['missing_symbol_source']}
    views, used, ignored = {}, [], {}
    for s in symbols:
        selected, counts = _select(s, snapshot['bars'][s], cutoff, deadline, mode)
        views[s] = _view(s, selected, sessions)
        used.extend((s,row) for row in selected)
        ignored[s] = counts
    own = views[symbol]
    if not np.isfinite(own.cutoff_return[day]):
        return {'X':None,'lineage':{},'rejections':['missing_opening_prefix']}
    if route in GROUP_ROUTES:
        # A frozen universe requires every listed peer source, even if a peer has no prefix.
        valid = {(v['week_id'],v['strategy_id']):v for v in versions if v['strategy_id'] in d.STRATEGIES}
        own_map, peers = {}, defaultdict(set)
        for member in memberships:
            v = valid.get((member['week_id'],member['strategy_id']))
            if (member['symbol'] not in symbols or v is None or
                member['version_id'] != v['version_id'] or len(member['group_ids']) != 1):
                continue
            own_map[(member['week_id'],member['strategy_id'],member['symbol'])] = (
                member['group_ids'][0],member['facts'],v)
            peers[(member['week_id'],member['group_ids'][0])].add(member['symbol'])
    else:
        own_map, peers = {}, {}
    group_seq = np.zeros(SHAPES['group_seq'],np.float32)
    if route in GROUP_ROUTES:
        for j, older in enumerate(sorted(representative.values())[-6:]):
            group_seq[6-len(representative)+j] = d._group_state(
                symbol,older,dates,sessions,views,own_map,peers)
    x5 = np.zeros(SHAPES['x5'],np.float32)
    lo=max(0,day-7)
    x5[7-(day-lo):7]=own.seq5[lo:day]
    x5[7,:90]=own.seq5[day,:90]
    x60=np.zeros(SHAPES['x60'],np.float32)
    lo=max(0,day-30)
    x60[30-(day-lo):30]=own.seq60[lo:day]
    for j,(_,end,_) in enumerate(d.HOUR_BINS):
        if end<=90:
            x60[30,j]=own.seq60[day,j]
    xday=np.zeros(SHAPES['xday'],np.float32)
    lo=max(0,day-126)
    xday[126-(day-lo):]=own.seqday[lo:day]
    prior, history_ids = _prior(symbol,deadline,mature_prior_store)
    data={'x5':x5[None], 'x60':x60[None], 'xday':xday[None],
          'group_seq':group_seq[None]}
    tab_frame=tabular(data,group=route in GROUP_ROUTES)
    tab=tab_frame.to_numpy(np.float32)
    path=path_summary(data)
    curve_frame=curve_features(data)
    curve=curve_frame.to_numpy(np.float32)
    relative_frame=group_relative_features(data) if route=='B_group' else None
    relative=relative_frame.to_numpy(np.float32) if relative_frame is not None else np.empty((1,0),np.float32)
    if (tab.shape[1] != schema['tabular_width'] or path.shape[1] != schema['path_width'] or
        curve.shape[1] != schema['curve_width'] or relative.shape[1] != schema['relative_width'] or
        list(tab_frame.columns) != schema['tabular_columns'] or
        [f'path_{i:03d}' for i in range(path.shape[1])] != schema['path_columns'] or
        list(curve_frame.columns) != schema['curve_columns'] or
        (list(relative_frame.columns) if relative_frame is not None else []) != schema['relative_columns']):
        raise ValueError('tabular/path/curve/relative width differs from manifest')
    # Removed route inputs are physically absent from X, not merely unused by a model flag.
    X={'x5':x5,'x60':x60,'prior':prior,'xbase':tab[0],'paths':path[0],
       'curve':curve[0],'relative':relative[0]}
    if route not in ('C_no_daily',):
        X['xday']=xday
    if route in GROUP_ROUTES:
        X['group_seq']=group_seq
    latest_available=max((row['_available'] for _,row in used),default=None)
    receipts=[row['_received'] for _,row in used]
    receipt_verified=(bool(used) and all(x is not None and x<=deadline for x in receipts)
                      and all(x['unverified_keys']==0 for x in ignored.values()))
    prior_verified=(metadata.get('mature_prior_scope')=='all_candidates' and
                    bool(metadata.get('mature_prior_snapshot_sha256')))
    actions_verified=(bool(metadata.get('corporate_actions_verified')) and
                      all(s in metadata['split_days'] for s in symbols))
    snapshot_file=snapshot.get('snapshot_file')
    snapshot_verified=False
    if snapshot_file is not None:
        path=Path(snapshot_file).expanduser().resolve(strict=True)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1<<20),b''):
                h.update(block)
        if h.hexdigest()!=snapshot.get('snapshot_sha256'):
            raise ValueError('snapshot file hash mismatch')
        snapshot_verified=bool(snapshot.get('snapshot_hash_verified'))
    lineage={'symbol':symbol,'session_date':date,'sample_id':f'{symbol}:{date}:11:30:v9',
             'cutoff_at':cutoff.isoformat(),'decision_at':anchor.isoformat(),
             'information_deadline_at':deadline.isoformat(),
             'prior_decision_at':deadline.isoformat(),
             'source_ids':sorted(row['_source_id'] for _,row in used),
             'group_version_ids':sorted(set(group_version_ids)),
             'mature_prior_history_ids':history_ids,
             'latest_available_at':latest_available.isoformat() if latest_available is not None else None,
             'latest_received_at':max(receipts).isoformat() if receipt_verified else None,
             'receipt_verified':receipt_verified,'mode':mode,
             'snapshot_sha256':snapshot.get('snapshot_sha256'),
             'snapshot_file':str(Path(snapshot_file).resolve()) if snapshot_file is not None else None,
             'g2_eligible':mode=='live' and receipt_verified and group_verified and
                           prior_verified and actions_verified and snapshot_verified,
             'ignored_versions':ignored}
    return {'X':X,'lineage':lineage,'rejections':[]}
